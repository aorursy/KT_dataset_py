import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.special import boxcox, inv_boxcox



train_df=pd.read_csv('../input/train.csv')



train_df.describe()
sns.boxplot(train_df['count'])

plt.show()



cnt=train_df['count'].values

q99=np.percentile(cnt,[99])



train_df=train_df[train_df['count']<q99[0]]

sns.distplot(train_df['count'])

plt.show()
from scipy.stats import boxcox

sns.distplot(boxcox(train_df['count'])[0])

plt.show()



sns.distplot(train_df['count'].apply(lambda x:np.log(x)))

plt.show()



sns.distplot(train_df['count'].apply(lambda x:x**0.5))

plt.show()

train_df['count']=train_df['count'].apply(lambda x:np.log(x))




from datetime import datetime



#converting string dattime to datetime





train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



new_df=train_df



new_df['month']=new_df['datetime'].apply(lambda x:x.month)

new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df.cov()

sns.heatmap(new_df.corr())

plt.show()



sns.boxplot('windspeed',data=train_df)

new_df.corr()
'''

humid + temp + month + hour + season + weather

humid + temp + month + hour

humid + temp + month

humid + temp

'''



final_df=new_df.drop(['datetime', 'holiday', 'workingday', 'atemp', 'windspeed', 'casual', 'registered'], axis=1)



#final_df=new_df.drop(['datetime', 'holiday', 'workingday', 'atemp', 'windspeed', 'casual', 'registered', 'season', 'weather'], axis=1)



#final_df=new_df.drop(['datetime', 'holiday', 'workingday', 'atemp', 'windspeed', 'casual', 'registered', 'season', 'weather', 'hour'], axis=1)



#final_df=new_df.drop(['datetime', 'holiday', 'workingday', 'atemp', 'windspeed', 'casual', 'registered', 'season', 'weather', 'hour', 'month'], axis=1)





final_df.head()







weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)

#year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)

month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)

hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)

season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)

                     





final_df=final_df.join(weather_df)

#final_df=final_df.join(year_df)

final_df=final_df.join(month_df)                     

final_df=final_df.join(hour_df)

final_df=final_df.join(season_df)

                     

final_df.head()



'''

#weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)

# year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)

month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)

hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)

#season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)

                     

#final_df=final_df.join(weather_df)

# final_df=final_df.join(year_df)

final_df=final_df.join(month_df)                     

final_df=final_df.join(hour_df)

#final_df=final_df.join(season_df)

'''



'''

#weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)

# year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)

month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)

#hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)

#season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)

                     

#final_df=final_df.join(weather_df)

# final_df=final_df.join(year_df)

final_df=final_df.join(month_df)                     

#final_df=final_df.join(hour_df)

#final_df=final_df.join(season_df)

'''



'''

#weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)

# year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)

#month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)

#hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)

#season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)

                     

#final_df=final_df.join(weather_df)

# final_df=final_df.join(year_df)

#final_df=final_df.join(month_df)                     

#final_df=final_df.join(hour_df)

#final_df=final_df.join(season_df)

'''



X=final_df.iloc[:,final_df.columns!='count'].values

print (X)



Y=final_df.iloc[:,4].values

# Y=final_df.iloc[:,2].values



print (Y)
import xgboost as xg

xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4)

xgr.fit(X,Y)



'''import xgboost as xg

from sklearn.model_selection import GridSearchCV



def grid_search():

    

    xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4)

    xgr.fit(X,Y)



    #rf=RandomForestRegressor(n_estimators=100,random_state=0)

    #rf.fit(X,Y)



    #parameters=[{'max_depth':[8,9,10,11,12],'min_child_weight':[4,5,6,7,8]}]

    #parameters=[{'gamma':[i/10.0 for i in range(0,5)]}]

    parameters=[{'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]}]



    grid_search= GridSearchCV(estimator=xgr, param_grid=parameters, cv=10,n_jobs=-1)



    print (1)

    grid_search=grid_search.fit(X,Y)

    print (2)

    best_accuracy=grid_search.best_score_

    best_parameters=grid_search.best_params_

    print (best_accuracy)

    print (best_parameters)



#if __name__ == '__main__':

   #grid_search()'''
'''

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=0)

rf.fit(X,Y)

imp_list=rf.feature_importances_

feats = {} # a dict to hold feature_name: feature_importance

for feature, importance in zip(final_df.columns, rf.feature_importances_):

    feats[feature] = importance #add the name/value pair

''' 
new_df.head()
new_df=pd.read_csv('../input/test.csv')

test_df=pd.read_csv('../input/test.csv')



new_df['datetime']=new_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



new_df['month']=new_df['datetime'].apply(lambda x:x.month)

new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)



weather_df=pd.get_dummies(new_df['weather'],prefix='w',drop_first=True)

month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)

hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)

season_df=pd.get_dummies(new_df['season'],prefix='s',drop_first=True)





new_df=new_df.join(weather_df)

new_df=new_df.join(month_df)                     

new_df=new_df.join(hour_df)

new_df=new_df.join(season_df)



new_df=new_df.drop(['datetime', 'holiday', 'workingday', 'atemp', 'windspeed'], axis=1)

new_df.head()

                     
import xgboost as xg

xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)

xgr.fit(X,Y)



X_test=new_df.iloc[:,:].values

X_test.shape

#print (new_df.columns)



y_output=xgr.predict(X_test)

y_output



test_df['count'] = pd.Series(np.exp(y_output))

test_df = test_df.drop(['humidity', 'temp', 'season', 'holiday', 'workingday', 'weather', 'atemp', 'windspeed'], axis=1)

test_df.to_csv('sub1.csv', index=False)


