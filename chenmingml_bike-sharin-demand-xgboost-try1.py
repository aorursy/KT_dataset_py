#load data, review the field and data type 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import boxcox, inv_boxcox
df_train=pd.read_csv("../input/train.csv", header=0)
df_train.head()
df_train.describe()
sns.boxplot(df_train['count'])
plt.show()
#set CI number
cnt=df_train['count'].values
q99=np.percentile(cnt,[99,95])
q99
df_train=df_train[df_train['count']<q99[0]]
sns.distplot(df_train['count'])
plt.show()
df_train['count']=df_train['count'].apply(lambda x:np.log(x))
sns.distplot(df_train['count'])
plt.show()
df_train['count']
# analysis of all variables categorical data: season, holiday, workingday, weather
i=0
cat_names=['season', 'holiday', 'workingday', 'weather']
for cat in cat_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.countplot(cat, data=df_train)
plt.show()
# analysis of all continuous data: temp, atemp, humidity, windspeed
cont_names=['temp', 'atemp', 'humidity', 'windspeed']
i=0
for name in cont_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.boxplot(name, data=df_train)
plt.show()
from datetime import datetime
df_train['datetime']=df_train['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
time_series_df=df_train
time_series_df.index=df_train['datetime']
#rolling average of count-60d
plt.plot(pd.rolling_mean(time_series_df['count'],60))
plt.show()
i=1
for name_1 in cont_names:
    j=cont_names.index(name_1)
    while (j<len(cont_names)-1):
        plt.subplot(6,1,i)
        plt.title(name_1+' vs '+cont_names[j+1])
        sns.jointplot(x=name_1, y=cont_names[j+1], data=df_train)
        j=j+1
        i=i+1
        plt.show()
from datetime import datetime
#converting string dattime to datetime
#train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
new_df=df_train
new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))
sns.swarmplot(x='hour',y='temp',data=new_df, hue='season')
plt.show()
sns.swarmplot(x='month',y='temp',data=new_df, hue='season')
plt.show()
sns.swarmplot(x='workingday',y='temp',data=new_df, hue='season')
plt.show()
sns.swarmplot(x='year',y='temp',data=new_df, hue='season')
plt.show()
#pairwise covariance of columns
new_df.cov()
sns.heatmap(new_df.corr())
plt.show()
#correlation
new_df.corr()
cat_names=['season', 'holiday', 'workingday', 'weather']
i=1
for name in cat_names:
    plt.subplot(2,2,i)
    sns.barplot(x=name,y='count',data=new_df,estimator=sum)
    i=i+1
    plt.show()
final_df=new_df.drop(['datetime','temp','windspeed','casual','registered','mnth+day','day'], axis=1)
final_df.head()
weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)
year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)
season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)
final_df=final_df.join(weather_df)
final_df=final_df.join(year_df)
final_df=final_df.join(month_df)                     
final_df=final_df.join(hour_df)
final_df=final_df.join(season_df)
                     
final_df.head()
final_df.columns
#training dataset
X=final_df.iloc[:,final_df.columns!='count'].values
y=final_df.iloc[:,6].values
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
def grid_search():
    xgr=xgb.XGBRegressor(max_depth=8, min_child_weight=6, gamma=0.4)
    xgr.fit(X,y)
    #rf=RandomForestRegressor(n_estimators=100,random_state=0)
    #rf.fit(X,Y)
    #parameters=[{'max_depth':[8,9,10,11,12],'min_child_weight':[4,5,6,7,8]}]
    #parameters=[{'gamma':[i/10.0 for i in range(0,5)]}]
    parameters=[{'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}]
    grid_search=GridSearchCV(estimator=xgr, param_grid=parameters, cv=10, n_jobs=-1)
    print ("Grid search fit")
    grid_search=grid_search.fit(X,y)
    print ("best score and params")
    best_accuracy=grid_search.best_score_
    best_parameters=grid_search.best_params_
    print (best_accuracy)
    print (best_parameters)
    
#if __name__ == '__main__':
   #grid_search()    
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=0)
rf.fit(X,y)
imp_list=rf.feature_importances_
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(final_df.columns, rf.feature_importances_):
    feats[feature] = importance #add the name/value pair 
import operator 
sorted_x=sorted(feats.items(), key=operator.itemgetter(1), reverse=True)
print (sorted_x)
xgr=xgb.XGBRegressor(max_depth=8, min_child_weight=6, gamma=0.4,colsample_bytree=0.6, subsample=0.6)
xgr.fit(X,y)
new_df=pd.read_csv('../input/test.csv')
df_submission=pd.DataFrame(new_df['datetime'], columns=['datetime'])
new_df['datetime']=new_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))


new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
#new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))

print (new_df.head())
new_df=new_df.drop(['datetime','temp','windspeed','day'], axis=1)
new_df.head()
#adding dummy varibles to categorical variables
weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)
yr_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)
season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)


new_df=new_df.join(weather_df)
new_df=new_df.join(yr_df)
new_df=new_df.join(month_df)                     
new_df=new_df.join(hour_df)
new_df=new_df.join(season_df)
                     
new_df.head()
X_test=new_df.iloc[:,:].values
X_test.shape
y_predict=xgr.predict(X_test)
y_predict
df_submission['count']=np.exp(y_predict)
df_submission
#df_submission.to_csv('submission_XGB_1.csv',index=False)

#kaggle score 0.41719 (413/3251=12.7%)