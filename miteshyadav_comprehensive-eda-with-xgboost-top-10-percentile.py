import pandas as pd



train_df=pd.read_csv('../input/train.csv')
train_df.head()

train_df.describe()
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.special import boxcox, inv_boxcox



sns.boxplot(train_df['count'])

plt.show()

#train_df['count']=train_df['count'].apply(lambda x:np.sqrt(x))

#train_df['count']=train_df['count'].apply(lambda x:np.sqrt(x))





cnt=train_df['count'].values

q99=np.percentile(cnt,[99])





train_df=train_df[train_df['count']<q99[0]]

sns.distplot(train_df['count'])

plt.show()
#from scipy.stats import boxcox

train_df['count']=train_df['count'].apply(lambda x:np.log(x))

#train_df['count']=boxcox(train_df['count'])[0]

sns.distplot(train_df['count'])

plt.show()

print (train_df['count'])








cat_names=['season', 'holiday', 'workingday', 'weather']



i=0

for name in cat_names:

    i=i+1

    plt.subplot(2,2,i)

    sns.countplot(name,data=train_df) 

    

plt.show()


cont_names=['temp','atemp','humidity','windspeed']



        

#sns.boxplot(train_df['season'])   

i=0

for name in cont_names:

    i=i+1

    plt.subplot(2,2,i)

    sns.boxplot(name,data=train_df) 

    

plt.show()




from datetime import datetime



train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

time_series_df=train_df

time_series_df.index=train_df['datetime']



import matplotlib.pyplot as plt



#Applying rolling average on a period of 60 days, as the typical weather lasts for around 3 months (20 days in training data of each month)

plt.plot(pd.rolling_mean(time_series_df['count'],60))

plt.show()


i=1

for name_1 in cont_names:

    j=cont_names.index(name_1)





    while(j<len(cont_names)-1):





        plt.subplot(6,1,i)

        plt.title(name_1+' vs '+cont_names[j+1])

        sns.jointplot(x=name_1,y=cont_names[j+1],data=train_df) 

        j=j+1

        i=i+1

        plt.show()

            

    





from datetime import datetime



#converting string dattime to datetime





#train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



new_df=train_df



new_df['month']=new_df['datetime'].apply(lambda x:x.month)

new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)

new_df['day']=new_df['datetime'].apply(lambda x:x.day)

new_df['year']=new_df['datetime'].apply(lambda x:x.year)

#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)

new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))



sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')

plt.show()



new_df.cov()

sns.heatmap(new_df.corr())

plt.show()



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




X=final_df.iloc[:,final_df.columns!='count'].values

print (X)



Y=final_df.iloc[:,6].values



print (Y)




import xgboost as xg

from sklearn.model_selection import GridSearchCV



def grid_search():

    print ('lets go')



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

   #grid_search()





from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100,random_state=0)

rf.fit(X,Y)

imp_list=rf.feature_importances_

feats = {} # a dict to hold feature_name: feature_importance

for feature, importance in zip(final_df.columns, rf.feature_importances_):

    feats[feature] = importance #add the name/value pair 
import operator

sorted_x = sorted(feats.items(), key=operator.itemgetter(1),reverse=True)

print (sorted_x)


import xgboost as xg

xgr=xg.XGBRegressor(max_depth=8,min_child_weight=6,gamma=0.4,colsample_bytree=0.6,subsample=0.6)

xgr.fit(X,Y)





new_df=pd.read_csv('../input/test.csv')

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

#print (new_df.columns)

#def invboxcox(y):

#    return(np.exp(np.log(0.69*y+1)/0.69))
y_output=xgr.predict(X_test)

y_output









op=pd.DataFrame({'count':np.exp(y_output)})

op.to_csv('sub1.csv')

print (np.exp(y_output))