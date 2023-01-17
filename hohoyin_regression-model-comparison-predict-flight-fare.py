# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from scipy.stats import norm, skew 

from scipy import stats

import warnings

warnings.filterwarnings("ignore")
#Read data set

X = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx',parse_dates = ['Date_of_Journey'] )

test = pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx',parse_dates = ['Date_of_Journey'] )
#Read first few rows of the data set

X.head()
#Rows and columns of the data set

X.shape
#Summary of numerical variables

X.describe()
#Summary of categorical variables

X.describe(include = ['object'])
X.dropna(inplace=True)
X.isnull().any()
#Combining test set and tarining set to conduct features engineering

all_data = pd.concat([X,test],axis = 0)



#X = all_data.iloc[:10682]

#test = all_data.iloc[10682:]
all_data.Total_Stops.value_counts()
no_stops = {'1 stop':1,'non-stop':0,'2 stops':2,'3 stops':3,'4 stops':4}

all_data.Total_Stops = all_data.Total_Stops.map(no_stops)
all_data.Total_Stops.value_counts()
all_data.Date_of_Journey.head()
#Add journey month and day of week, then drop the original date of jounery column

all_data['Journey_month'] = all_data.Date_of_Journey.dt.month

all_data['Journey_day_of_week'] = all_data.Date_of_Journey.dt.dayofweek

all_data['Journey_year'] = all_data.Date_of_Journey.dt.year

all_data['Journey_day'] = all_data.Date_of_Journey.dt.day

all_data.drop('Date_of_Journey',axis = 1,inplace = True)
all_data['Journey_year'].value_counts()
all_data.drop('Journey_year',axis = 1,inplace = True)
all_data[['Dep_Time', 'Arrival_Time']].head()
#add new 4 features in form of hour and mintues of departure time and arrival time.

all_data["Dep_hour"],all_data["Arr_hour"]= pd.to_datetime(all_data['Dep_Time']).dt.hour,pd.to_datetime(all_data['Arrival_Time']).dt.hour



all_data["Dep_min"],all_data["Arr_min"]= pd.to_datetime(all_data['Dep_Time']).dt.minute,pd.to_datetime(all_data['Arrival_Time']).dt.minute
#Drop the original depoarture time and arrival time columns

all_data.drop(['Arrival_Time'],axis =1 ,inplace =True)

all_data.drop(['Dep_Time'],axis =1,inplace =True)
all_data.head()
all_data.Duration.head()
d = list(all_data.Duration)

duration_hour = []

duration_min = []



for time in d:

    if len(time.split()) == 2: #cell with both hour and mintues data

        hour = time.split()[0].rsplit('h')[0]

        mintues = time.split()[1].rsplit('m')[0]

        duration_hour.append(hour)

        duration_min.append(mintues)

    else: 

        #data with only hour or mintue information.

        if 'h' in time.split()[0]:

            hour =  time.split()[0].rsplit('h')[0]

            duration_hour.append(int(hour))

            duration_min.append(0) 

            # 0 mintues in there are no mintues data

        elif 'm'in time.split()[0]:

            mintues =  time.split()[0].rsplit('m')[0]

            duration_hour.append(0)

            duration_min.append(int(mintues))
#Check whether the length of the two lists we just created is same with the data set.

len(duration_hour) == len(duration_min ) == len(all_data)
all_data['duration_hour'] = pd.DataFrame(duration_hour).astype('int32')

all_data['duration_min'] = pd.DataFrame(duration_min).astype('int32')
all_data.drop(['Duration'],axis = 1, inplace = True)
all_data['duration_min'].value_counts()
all_data['duration_hour'].value_counts()
all_data.head()
all_data.head()
#Check again to ensure there are no missing data

all_data.isnull().any()
X = all_data.iloc[:10682]

test = all_data.iloc[10682:]
sns.distplot(X['Price'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(X['Price'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(X['Price'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

X["Price"] = np.log1p(X["Price"])



#Check the new distribution 

sns.distplot(X["Price"] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(X["Price"])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(X["Price"], plot=plt)

plt.show()
#list of categorical and numerical variables

cat_cols = X.dtypes =='object'

cat_cols = list(cat_cols[cat_cols].index)

num_cols = X.dtypes != 'object'

num_cols = list(num_cols[num_cols].index)

num_cols.remove('Price')
def cat_boxplot(col,target,train,a=8,b=6):

    f, ax = plt.subplots(figsize=(a, b))

    fig = sns.boxplot(x=col, y=str(target), data=train)

    #fig.axis(xmin=0, xmax=x);
#Airline and log Price

cat_boxplot('Price',cat_cols[0],X,)
cat_boxplot('Price',cat_cols[1],X)
cat_boxplot('Price',cat_cols[2],X)
cat_boxplot('Price',cat_cols[3],X,a = 15,b= 15)
cat_boxplot('Price',cat_cols[4],X)
def num_plot(col,target,train,a=8,b=6):

    f, ax = plt.subplots(figsize=(a, b))

    fig = sns.scatterplot(x=col, y=str(target), data=train)

    #fig.axis(ymin=0, ymax=x);
num_cols
for i in range(len(num_cols)):

    num_plot(num_cols[i],'Price',X)
for i in range(len(num_cols)):

    cat_boxplot(num_cols[i],'Price',X)
X.head()
import category_encoders as ce



cat_features = ['Additional_Info','Route']

# Create the encoder

target_enc = ce.CatBoostEncoder(cols=cat_features)

target_enc.fit(X[cat_features], X['Price'])



# Transform the features, rename columns with _cb suffix, and join to dataframe

X_CBE = X.join(target_enc.transform(X[cat_features]).add_suffix('_cb'))

test_CBE = test.join(target_enc.transform(test[cat_features]).add_suffix('_cb'))
test_CBE['Route_cb'].value_counts()
X = X_CBE.copy()

test = test_CBE.copy()

X.drop(['Route', "Additional_Info"], axis = 1, inplace = True)

test.drop(['Route', "Additional_Info"], axis = 1, inplace = True)
#Function to create a data frame with number and percentage of missing data in a data frame

def missing_to_df(df):

    #Number and percentage of missing data in training data set for each column

    total_missing_df = df.isnull().sum().sort_values(ascending =False)

    percent_missing_df = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)

    missing_data_df = pd.concat([total_missing_df, percent_missing_df], axis=1, keys=['Total', 'Percent'])

    return missing_data_df

missing_df = missing_to_df(X)

missing_df[missing_df['Total'] > 0]
#Correlation map to see how features are correlated with Price

corrmat = X.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
all_data = pd.concat([X,test],axis = 0)

all_data = pd.get_dummies(all_data)

all_data.shape
X = all_data.iloc[:10682]

test = all_data.iloc[10682:]
test = test.drop('Price',axis = 1)
def get_data_splits(dataframe, valid_fraction=0.1):

    valid_fraction = 0.1

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    

    return train, valid, test
from sklearn.feature_selection import SelectKBest, f_classif



sel_train, valid, _ = get_data_splits(X)

feature_cols = list(sel_train.columns)

feature_cols.remove('Price')

# Keep 10 features

selector = SelectKBest(f_classif, k=30)



X_new = selector.fit_transform(sel_train[feature_cols], sel_train['Price'])
# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=sel_train.index, 

                                 columns=feature_cols)
# Dropped columns have values of all 0s, so var is 0, drop them

selected_columns = selected_features.columns[selected_features.var() != 0]



# Get the valid dataset with the selected features.

selected_columns
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import lightgbm as lgb
y = X.Price
X = X.drop('Price',axis = 1)
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)

    rmse= np.sqrt(-cross_val_score(model, X.values, 

                                   y.values, 

                                   scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lasso_score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(lasso_score.mean(), lasso_score.std()))
enet_score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(enet_score.mean(), enet_score.std()))
krr_score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(krr_score.mean(), krr_score.std()))
gboost_score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(gboost_score.mean(), gboost_score.std()))
lgb_score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(lgb_score.mean(), lgb_score.std()))
models = pd.DataFrame({

    'Model': ['LASSO', 'ENet', 'KRR', 

              'GBoost', 'lgb'],#,'XGBoost'],

    'Mean_Score': [lasso_score.mean(), enet_score.mean(), krr_score.mean(), 

              gboost_score.mean(), lgb_score.mean()]})

#xgb_score.mean()

models.sort_values(by='Mean_Score', ascending=True)
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X[selected_columns].values)

    rmse= np.sqrt(-cross_val_score(model, X[selected_columns].values, 

                                   y.values, 

                                   scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso_score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(lasso_score.mean(), lasso_score.std()))

enet_score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(enet_score.mean(), enet_score.std()))

krr_score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(krr_score.mean(), krr_score.std()))

gboost_score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(gboost_score.mean(), gboost_score.std()))

lgb_score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(lgb_score.mean(), lgb_score.std()))
models_sel = pd.DataFrame({

    'Model': ['LASSO', 'ENet', 'KRR', 

              'GBoost', 'lgb'], #'XGBoost',

    'Mean_Score': [lasso_score.mean(), enet_score.mean(), krr_score.mean(), 

              gboost_score.mean(), lgb_score.mean()]})  #xgb_score.mean(),

models_sel.sort_values(by='Mean_Score', ascending=True)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

GBoost.fit(X_train, y_train)

y_pred = GBoost.predict(X_test)

print('MAE:', mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)))

print('MSE:', mean_squared_error(np.expm1(y_test),np.expm1(y_pred)))

print('RMSE:', np.sqrt(mean_squared_error(np.expm1(y_test),np.expm1(y_pred))))
print('R square error:', r2_score(np.expm1(y_test),np.expm1(y_pred)))
GBoost = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05,

                                   max_depth=20, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
GBoost.fit(X,y)

gboost_pred = (GBoost.predict(X))

y_pred = np.expm1(GBoost.predict(test))

print('mean_squared_error: ',(rmsle(np.expm1(y),np.expm1(gboost_pred))))
submission = pd.DataFrame({'Price':y_pred})

submission.head()