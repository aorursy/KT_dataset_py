import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
np.warnings.filterwarnings('ignore')

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.10f' % x)
# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df = train.append(test).reset_index()
df.head()
train.head()
test.head()
test.isnull().sum().sum()
train.isnull().sum().sum()
df.shape
train.shape
test.shape
df.info()
df.describe([0.25,0.95]).T
df.isnull().sum()
df.isnull().sum().sum()
df.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature","index","Id"],axis=1,inplace=True)
df.isnull().sum()
df.isnull().sum().sum()
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Kategorik Değişken Sayısı: ', len(cat_cols))
cat_cols
dfc = df[cat_cols]
dfc.head()
#We conduct a stand alone observation review for the salary variable
#We suppress contradictory values
Q1 = df.SalePrice.quantile(0.25)
Q3 = df.SalePrice.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["SalePrice"] > upper,"SalePrice"] = upper
df.drop(cat_cols,axis=1,inplace=True)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 10)
df_filled = imputer.fit_transform(df)
df = pd.DataFrame(df_filled,columns = df.columns)
df.isnull().sum().sum()
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 15)
lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]
th = np.sort(df_scores)[15]
th
df = df[df_scores > th]
df = pd.concat([df, dfc], axis=1)
df.shape
df.isnull().sum()
df = df[~(df["MSSubClass"].isnull())]
df.isnull().sum()
# LABEL ENCODING & ONE-HOT ENCODING
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=False):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns

df, new_cols_ohe = one_hot_encoder(df, cat_cols)
df.head()
df.shape
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                   and col not in "Id"
                   and col not in "SalePrice"
                   and col not in like_num]
def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


for col in cols_need_scale:
    df[col] = robust_scaler(df[col])
df.isnull().sum()
df.drop(["LowQualFinSF","3SsnPorch","MiscVal"],axis=1,inplace=True)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

test = test_df[["Id"]]
# train_df tüm veri setimiz gibi davranarak derste ele aldığımız şekilde modelelme işlemini gerçekleştiriniz.
X = train_df.drop('SalePrice', axis=1)
y = train_df[["SalePrice"]]

import statsmodels.api as sm
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols


X_train, X_test, y_train, y_test = train_test_split(X[selected_features_BE], y, test_size=0.20, random_state=46)

# TODO scaler'i burada çalıştırıp deneyebilirsiniz.

models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge()),
          ('Lasso', Lasso()),
          ('ElasticNet', ElasticNet())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)
    
Lasso.fit(X_train, y_train)
y_pred = Lasso.predict(X_test)
    
print(X_train.shape)
print(X_test.shape)
df["SalePrice"].mean()
submission_file_name = "submission_kernel02.csv"
