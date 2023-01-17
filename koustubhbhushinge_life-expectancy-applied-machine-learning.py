import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split,GridSearchCV
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingCVRegressor

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
df = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')
df.head()
df.corr()
df.isnull().sum()
df_label_nan = df[df['Life expectancy '].isnull() == True]
df.drop(df[df['Life expectancy '].isnull() == True].index,inplace = True)
df.drop('Population',axis = 1,inplace=True)
df.reset_index(drop = True,inplace = True)
df.info()
df['Alcohol'].fillna(df['Alcohol'].median(),inplace = True)
df[' BMI '].fillna(df[' BMI '].mean(),inplace = True)
df['Polio'].fillna(df['Polio'].mean(),inplace = True)
df['Diphtheria '].fillna(df['Diphtheria '].mean(),inplace = True)
df[' thinness  1-19 years'].fillna(df[' thinness  1-19 years'].mean(),inplace = True)
df[' thinness 5-9 years'].fillna(df[' thinness 5-9 years'].mean(),inplace = True)
df['Total expenditure'].fillna(df['Total expenditure'].mean(),inplace = True)
#Income Schooling
df_ = df[['Income composition of resources','Schooling']]

i = KNNImputer(n_neighbors=36)
df_i = pd.DataFrame(i.fit_transform(df_))

df['Income composition of resources'] = df_i.iloc[:,0]
df['Schooling'] = df_i.iloc[:,1]

#GDP
df_ = df[['GDP','percentage expenditure']]

i = KNNImputer(n_neighbors=36)
df_i = pd.DataFrame(i.fit_transform(df_))

df['GDP'] = df_i.iloc[:,0]

#Hepatitis B
df_ = df[['Hepatitis B','Diphtheria ']]

i = KNNImputer(n_neighbors=36)
df_i = pd.DataFrame(i.fit_transform(df_))

df['Hepatitis B'] = df_i.iloc[:,0]
df.isnull().sum()
df['Year'] = df['Year'].apply(str)
cat = ['Country','Year','Status']

for f in cat:
    df_ = pd.get_dummies(df[f],prefix=f,drop_first=True)
    df.drop(f,axis = 1,inplace = True)
    df = pd.concat([df,df_],axis = 1)
df.head()
df.info()
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat['Life expectancy '])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#Schooling
plt.scatter(df['Schooling'],df['Life expectancy '])
df.drop(df[df['Schooling']<1].index,inplace = True)
plt.scatter(df['Schooling'],df['Life expectancy '])
continuous_features = [feature for feature in df.columns if len(df[feature].unique())>5 and df[feature].dtype != 'object' and 'Year' not in feature and 'Yr' not in feature]
continuous_features,len(continuous_features)
sk = df[continuous_features].apply(lambda x:skew(x)).sort_values(ascending = False)
sk = pd.DataFrame(sk)
sk
ch = [0,0.03,0.05,0.08,0.1,0.13,0.15]
df__ = pd.DataFrame()
for choice in ch:
    df_ = pd.DataFrame(skew(boxcox1p(df[continuous_features],choice)),columns=[choice],index = continuous_features)
    df__ = pd.concat([df__,df_],axis = 1)
    
df__ = pd.concat([pd.DataFrame(skew(df[continuous_features]),columns = ['Org'],index = continuous_features),df__],axis = 1)
df__
skew_result = {}
for i in df__.index:
    min_ = 'Org'
    for j in df__.columns:
        if df__.loc[i,j]>=0 and df__.loc[i,j]<df__.loc[i,min_]:
            min_ = j
            
    skew_result[i] = min_
    

print(skew_result)
skew_result = {k:v for k,v in skew_result.items() if v != 'Org'}
#boxcox1p for other continuous values 
for k,v in skew_result.items():
    df[k] = boxcox1p(df[k],v)
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat['Life expectancy '])>0.4]
plt.figure(figsize=(10,10))
g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.drop('infant deaths',axis = 1,inplace = True)
x = df.drop('Life expectancy ',axis = 1)
y = df['Life expectancy ']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
#Validation
kf = KFold(5, shuffle=True, random_state=42)

def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def rmsle_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x.values, y.values, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lr = make_pipeline(RobustScaler(),LinearRegression(fit_intercept=True,normalize=True))
svr = make_pipeline(RobustScaler(),SVR())
rf = RandomForestRegressor(n_estimators=200)
stk = StackingCVRegressor( regressors=(lr,svr,rf),meta_regressor=rf,use_features_in_secondary = True)
stk.fit(x_train.values,y_train.values)
rmse(y_test,stk.predict(x_test.values))
rmsle_cv(stk).mean()
