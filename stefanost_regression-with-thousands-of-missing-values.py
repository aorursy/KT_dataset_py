import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")
df.head()
df.info()
#visualize NaN values
sns.heatmap(df.isnull());
#columns with null values
nans = pd.DataFrame(data=[], index=None, 
                          columns=['feature_name','missing_values','percentage_of_total'])
nans['feature_name'] = df.columns[df.isna().sum()>0]
nans['missing_values'] = np.array(df[nans.iloc[:,0]].isna().sum())
nans['percentage_of_total'] = np.round(nans['missing_values'] / df.shape[0] * 100)
nans['var_type']= [df[c].dtype for c in nans['feature_name']]
nans
#make list of columns with up to 50 nan, impute these with column mean
nan_cols = list(nans['feature_name'][nans['missing_values']<=50])
for col in nan_cols:
    mean_ = df[col].mean()
    df[col][df[col].isna()==True] = mean_
#columns with null values
nans = pd.DataFrame(data=[], index=None, 
                          columns=['feature_name','missing_values','percentage_of_total'])
nans['feature_name'] = df.columns[df.isna().sum()>0]
nans['missing_values'] = np.array(df[nans.iloc[:,0]].isna().sum())
nans['percentage_of_total'] = np.round(nans['missing_values'] / df.shape[0] * 100)
nans['var_type']= [df[c].dtype for c in nans['feature_name']]
nans
corr_matr = df.corr()
corr_matr
sns.heatmap(df.corr());
#for later use, to compare with linear regression coefficients
cols=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
feat_names = corr_matr.iloc[1,cols].keys()
corr_coefs = corr_matr.iloc[1,cols].values
#more concise
def impute(df, to_impute, reference):
    index=df[to_impute][(df[to_impute].isna()==True)&
                    (df[reference].isna()==False)].keys()
    #df['Total expenditure'][index]
    var_min = df[reference].min()
    var_max = df[reference].max()
    range_filler =  var_max - var_min
    step = range_filler / 10
    one = df[to_impute][df[reference] < (var_min+step)].mean()
    two = df[to_impute][(df[reference] > (var_min+step))&
              (df[reference] < (var_min+step*2))].mean()
    three = df[to_impute][(df[reference] > (var_min+step*2))&
              (df[reference] < (var_min+step*3))].mean()
    four = df[to_impute][(df[reference] > (var_min+step*3))&
              (df[reference] < (var_min+step*4))].mean()
    five = df[to_impute][(df[reference] > (var_min+step*4))&
              (df[reference] < (var_min+step*5))].mean()
    six = df[to_impute][(df[reference] > (var_min+step*5))&
              (df[reference] < (var_min+step*6))].mean()
    seven = df[to_impute][(df[reference] > (var_min+step*6))&
              (df[reference] < (var_min+step*7))].mean()
    eight = df[to_impute][(df[reference] > (var_min+step*7))&
              (df[reference] < (var_min+step*8))].mean()
    nine = df[to_impute][(df[reference] > (var_min+step*8))&
              (df[reference] < (var_min+step**9))].mean()
    ten = df[to_impute][df[reference] > (var_max-step)].mean()
    
    for i in index:
        if df[reference][i] < (var_min+step):
            df[to_impute][i]=one
        elif df[reference][i] < (var_min+step*2):
                df[to_impute][i]=two
                continue
        elif df[reference][i] < (var_min+step*3):
                df[to_impute][i]=three
                continue
        elif df[reference][i] < (var_min+step*4):
                df[to_impute][i]=four
                continue
        elif df[reference][i] < (var_min+step*5):
                df[to_impute][i]=five
                continue
        elif df[reference][i] < (var_min+step*6):
                df[to_impute][i]=six
                continue
        elif df[reference][i] < (var_min+step*7):
                df[to_impute][i]=seven
                continue
        elif df[reference][i] < (var_min+step*8):
                df[to_impute][i]=eight
                continue
        elif df[reference][i] < (var_min+step**9):
                df[to_impute][i]=nine
                continue
        else:
            df[to_impute][i]=ten
impute(df, 'GDP', 'Total expenditure')
impute(df, 'Total expenditure', 'GDP')
df['GDP'][df['GDP'].isna() == True] =df['GDP'].mean()
df['Total expenditure'][df['Total expenditure'].isna()==True]=df['Total expenditure'].mean()

impute(df, 'Alcohol', 'Schooling')
impute(df, 'Schooling', 'Alcohol')
df['Alcohol'][df['Alcohol'].isna() == True] =df['Alcohol'].mean()
df['Schooling'][df['Schooling'].isna()==True]=df['Schooling'].mean()

impute(df, 'Hepatitis B', 'Diphtheria ')
df['Hepatitis B'][df['Hepatitis B'].isna() == True] =df['Hepatitis B'].mean()

impute(df, 'Population', 'infant deaths')
df['Population'][df['Population'].isna() == True] =df['Population'].mean()

impute(df, 'Income composition of resources', 'Schooling')
df['Income composition of resources'][df['Income composition of resources'].isna() == True] =df['Income composition of resources'].mean()
#are there any missing values left?
df.isna().sum()
#df.to_csv('imputed_data.csv')
df=pd.read_csv('imputed_data.csv')
#one-hot encode categorical variables
dummies=pd.get_dummies(df[['Country','Status']])
dummies.head()
#prepare final dataset, concatenate dummy columns and drop original categorical columns
df=pd.concat([df,dummies],axis=1)
df.drop(columns=['Country','Status'],inplace=True)
df.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#prepare train and test data
x=df.drop(columns='Life expectancy ').values
y=df['Life expectancy '].values
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.2, 
                                     random_state=42, shuffle=True)
# Linear Regression
linreg=LinearRegression()
linreg.fit(x_tr,y_tr)
pred=linreg.predict(x_ts)
print('mse:',mean_squared_error(y_ts,pred))
print('r2_score:',r2_score(y_ts,pred))
#comparison of correlation coefficients and linear regression coefficients, between predictors and target variable
features = pd.DataFrame(data=[], index=None, 
                          columns=['feature_name','correlation_with_target','lin_reg_coefficient'])
features['feature_name']=feat_names
features['correlation_with_target']=corr_coefs
features['lin_reg_coefficient']=np.round(linreg.coef_[0:19],decimals=3)
features
#RANSAC regression
ransac=RANSACRegressor(LinearRegression(),max_trials=120,min_samples=50,
                      loss='absolute_loss',residual_threshold=5.0,
                       random_state=42)
ransac.fit(x_tr,y_tr)
pred=ransac.predict(x_ts)
print('mse:',mean_squared_error(y_ts,pred))
print('r2_score:',r2_score(y_ts,pred))
#Ridge Regression
ridge=Ridge(alpha=0.1)
ridge.fit(x_tr,y_tr)
pred=ridge.predict(x_ts)
print('mse:',mean_squared_error(y_ts,pred))
print('r2_score:',r2_score(y_ts,pred))
#Lasso Regression
lasso=Lasso(alpha=0.1)
lasso.fit(x_tr,y_tr)
pred=lasso.predict(x_ts)
print('mse:',mean_squared_error(y_ts,pred))
print('r2_score:',r2_score(y_ts,pred))
#ElasticNet
ela=ElasticNet(alpha=0.1, l1_ratio=0.5)
ela.fit(x_tr,y_tr)
pred=ela.predict(x_ts)
print('mse:',mean_squared_error(y_ts,pred))
print('r2_score:',r2_score(y_ts,pred))
#Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=42)
rf.fit(x_tr,y_tr)
tr_pred=rf.predict(x_tr)
pred=rf.predict(x_ts)
print('mse train:',mean_squared_error(y_tr,tr_pred))
print('mse test:',mean_squared_error(y_ts,pred))
print('')
print('r2_score train:',r2_score(y_tr,tr_pred))
print('r2_score test:',r2_score(y_ts,pred))
features['Random Forest importances'] = np.round(rf.feature_importances_[0:19], decimals=3)
features