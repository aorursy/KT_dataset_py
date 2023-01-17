import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler,PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option("display.max_columns", None)

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.multicomp import pairwise_tukeyhsd, tukeyhsd, MultiComparison
from scipy.stats import jarque_bera, shapiro, bartlett
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
def drop_cols(data, cols):
    if len(cols)==0:
        return df
    return data.drop(cols, axis=1)

def missing(data):
    if data.isna().sum().sum()==0:
        return "all missing values treated"
    data = data.isna().sum()/data.shape[0]
    data[data>0].plot(kind='bar', figsize=(16,7))
    all_miss = list(data[data==1].index)
    print("These columns have all the values missing",all_miss)
    plt.title("Missing value plot")
    plt.tight_layout()
    plt.xlabel("Column")
    plt.ylabel("Missing data in %")
    plt.xticks(rotation=90)
    plt.show()

def treat_skew(data, exclude=None, threshold = 1, ):
    cols = list(data.skew()[abs(data.skew())>threshold].index)    
    cols = [ col for col in cols if col not in exclude]
    return cols
df = pd.read_csv("/kaggle/input/londonairbnb/vishal_final.csv")
df = drop_cols(df, ["country","lastreviewdays", "firstreviewdays"])
original_price = df["price"]
Y = np.log(df["price"])
X = df.drop(["id","price"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=1)
x_train.shape, x_test.shape
def feature_select_rf(train=None, test=None, y=y_train, threshold = '0.7*median'):
    selector_rf = SelectFromModel(select_rf, threshold='0.7*median')
    columns = train.columns
    selector_rf.fit(train, y)
    columns_rf = [col for col, flag in zip(columns,selector_rf.get_support()) if flag]
    x_select = train[columns_rf]
    print(f"Feature shape selected : {x_select.shape}")
    test_select = test[columns_rf]
    return (x_select, test_select)

def preprocess_onehot(train, test=None, y=y_train, select=True):
    ct_onehot.fit(train)
    onehot_cols = list(ct_onehot.named_transformers_['onehot'].get_feature_names(dummy))
    all_columns = skew + passthru + onehot_cols
    x_transform = pd.DataFrame(scaler.fit_transform(ct_onehot.transform(train)), columns=all_columns)
    x_transform = pd.DataFrame(numeric_skew.fit_transform(imputer_br.fit_transform(x_transform)), columns=all_columns)
    if test is not None:
        test_transform = pd.DataFrame(scaler.transform(ct_onehot.transform(test)), columns=all_columns)
        test_transform = pd.DataFrame(numeric_skew.transform(imputer_br.transform(test_transform)), columns=all_columns)
        print("Original Data Shape", x_transform.shape, test_transform.shape)
        if select:
            print("Selecting the final features...")
            x_transform, test_transform = feature_select_rf(x_transform, test_transform, y_train)
        return (x_transform, test_transform)
    print("Original Data Shape", x_transform.shape)
    return x_transform
numeric = ["host_acceptance_rate","accommodates","bathrooms","bedrooms","beds","security_deposit","cleaning_fee",
       "guests_included","extra_people","availability_30","availability_60","availability_90","availability_365",
       "number_of_reviews","number_of_reviews_ltm","review_scores_rating","review_scores_accuracy",
       "review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
       "review_scores_value","reviews_per_month","amenity_sum","distance","premium",'neighbourhood_cleansed']
pt = PowerTransformer(method='yeo-johnson')
treatment = pd.DataFrame(X[numeric].skew())
treatment['PT'] = pd.DataFrame(pt.fit_transform(df[numeric]), columns = numeric).skew()

skew = treat_skew(X[numeric], ['host_acceptance_rate'])
categorical = list(set(X.columns).difference(set(numeric)))

dummy = ["experiences_offered","property_type","room_type","bed_type","cancellation_policy","host_response_time"]
passthru = list(set(X.columns).difference(set(dummy+skew)))
pt = PowerTransformer(method='yeo-johnson')

impute_br = BayesianRidge()
imputer_br = IterativeImputer(impute_br, skip_complete=True)

select_rf = RandomForestRegressor(random_state=1, n_jobs=4)

onehot = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

ct_onehot = ColumnTransformer([('pass1','passthrough', skew),
                               ('pass2','passthrough', passthru),
                               ('onehot', onehot,   dummy)    ], remainder='drop')

numeric_skew = ColumnTransformer([('skew', pt, np.arange(len(skew)))],
                                  remainder='passthrough')
x_transform, test_transform = preprocess_onehot(x_train, x_test, select=False)
plt.figure(figsize=(16,16))
corr = X[numeric].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlGnBu', mask=mask)
plt.title("Correlation map for numeric variables")
plt.show()

x_constant = sm.add_constant(x_transform)
test_constant = sm.add_constant(test_transform)
model = sm.OLS(y_train.reset_index(drop=True), x_constant).fit()
model.summary()
import statsmodels.tsa.api as smt
acf = smt.graphics.plot_acf(model.resid, lags=100 , alpha=0.05)
acf.show()
durbin_watson(model.resid)

from statsmodels.compat import lzip
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(model.resid)
lzip(name, test)
shapiro(model.resid)
import scipy.stats as st
plt.figure(figsize=(6,4))
st.probplot(model.resid, dist = "norm", plot=plt)
plt.show()
sns.residplot(model.predict(), model.resid, lowess=True, color='g')
ax.set(xlabel = 'Fitted value', ylabel = 'Residuals', title = 'Residual vs Fitted Plot \n')
plt.show()
sns.distplot(model.resid)
plt.title("Skewness of residuals: %.4f"%stats.skew(model.resid))
plt.show()

# homoskedasticity test
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(model.resid, model.model.exog)
lzip(name, test)

sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

def linearity_test(model, y):
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
linearity_test(model, y_train)
sms.diagnostic.linear_rainbow(model, frac=0.5)
vif_val = pd.DataFrame(index = x_transform.columns)
vif_val["VIF"]= [vif(x_transform.values, i) for i in range(x_transform.shape[1])]
vif_val
#vif_val.loc[numeric,:].sort_values('VIF', ascending=False)

def calc_vif(data, threshold=10):
    vif_max = threshold
    col_vif = np.Inf
    while col_vif>vif_max:
        vif_val = pd.DataFrame(index = data.columns)
        vif_val["VIF"]= [vif(data.values, i) for i in range(data.shape[1])]
        col_vif = vif_val.VIF.max()
        drop_col = list(vif_val[vif_val.VIF == col_vif].index)
        print("Dropping column...", drop_col,"VIF :",col_vif)
        data = data.drop(drop_col, axis=1)
    return list(data.columns)
calc_vif(x_transform[numeric])
vif_filter = ['host_acceptance_rate', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'security_deposit','cleaning_fee',
              'guests_included', 'extra_people', 'availability_30', 'availability_365', 'number_of_reviews', 'number_of_reviews_ltm',
              'review_scores_location', 'amenity_sum', 'distance', 'premium','neighbourhood_cleansed']

onehot_cols = list(set(x_transform.columns).symmetric_difference(numeric))
onehot_cols.remove('property_type_House')
onehot_cols.remove('month_book')
onehot_cols.remove('cancellation_policy_flexible')
onehot_cols.remove('week_book')
onehot_cols.remove('experiences_offered_none')
final_cat = list(model.pvalues[onehot_cols][model.pvalues[onehot_cols]<0.05].index)

x_final = x_transform[vif_filter + final_cat]
test_final = test_transform[vif_filter + final_cat]

x_final_constant = sm.add_constant(x_final)
test_final_constant = sm.add_constant(test_final)
model_final = sm.OLS(y_train.reset_index(drop=True), x_final_constant).fit()
model_final.summary()

