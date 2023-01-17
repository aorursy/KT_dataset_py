import pandas as pd



train_df=pd.read_csv('../input/train.csv')
train_df.head()
train_df.describe()

#Univariate analysis of all variables

#Categorical data--> Season, Holiday, WorkingDay, Weather

import seaborn as sns

import matplotlib.pyplot as plt



#train=sns.load_dataset('data/train.csv')



cat_names=['season', 'holiday', 'workingday', 'weather']



        

#sns.boxplot(train_df['season'])   

i=0

for name in cat_names:

    i=i+1

    plt.subplot(2,2,i)

    sns.countplot(name,data=train_df) 

    

plt.show()





#Univariate analysis for continuous data

cont_names=['temp','atemp','humidity','windspeed']



        

#sns.boxplot(train_df['season'])   

i=0

for name in cont_names:

    i=i+1

    plt.subplot(2,2,i)

    sns.boxplot(name,data=train_df) 

    

plt.show()

# some of the inferences that can be made

#Holiday and working day look  somewhat correlated. Can one of them be removed to avoid multi-collinearity? Let's wait until we calculate thier correlation value

#Not much can be inferred from season data. Majorit of the data fall under 1 and 2, which is clear skies mist/cloudy.

# Temp, Atemp, humidity look normally distributed. However, windspeed has a lot of outliers which will be analysed further.

#doing a brief time-series analysis to see if there's any improvement in count over a period of time

#moving average to be calculated for a period of 3/4 months as that is the no of months in one season





from datetime import datetime



train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

"""

time_series_df=train_df

time_series_df.index=train_df['datetime']



import matplotlib.pyplot as plt



plt.plot(pd.rolling_mean(time_series_df['count'],60))

plt.show()

"""



#As expected the total count grows over a period of time, therefore this dataset needs to incorporate chaanges in seasonality too.

#calculating bivariate analysis on continuous data

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

            

    
#Not much can be inferred about the distribution of these variables except for variable 'temp' and 'atemp' that almost have

#similar context. We would be using the 'temp' and getting rid of the 'atemp' variables for better precision value and avoiding multi-collinearity.
#sns.boxplot(x='season',y='count',data=train_df)

#plt.show()



type(train_df['datetime'][0])
#Let us perfrom some feature engineering. The datetime column can be used to extract data like the month, day, hour which can be

#used in our model for making better predictions.



from datetime import datetime



#converting string dattime to datetime





#train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))



new_df=train_df



new_df['month']=new_df['datetime'].apply(lambda x:x.month)

new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)

new_df['day']=new_df['datetime'].apply(lambda x:x.day)

#new_df['year']=new_df['datetime'].apply(lambda x:x.year)

#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)

new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))



print (new_df.head())





x='2012-11-30 14:00:00'

n=datetime.strptime(x,'%Y-%m-%d %H:%M:%S')

n.month

n.day

n.year
sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')

plt.show()



# a non-linear relationship between temperature and day of the hour according to different seasons is evident from this chart.
#sns.pairplot(train_df)

#plt.show()



#Season 3 and 4 have the highest number of bicycle registrations.

new_df.cov()

sns.heatmap(new_df.corr())

plt.show()



# A lot of inferences that we have already covered could be verified using the following heatmap

#Seas
new_df.corr()

# Visualizing multi-variate distribution of target variable with other categorical data.



cat_names=['season', 'holiday', 'workingday', 'weather']

i=1

for name in cat_names:

    plt.subplot(2,2,i)

    sns.barplot(x=name,y='count',data=new_df,estimator=sum)

    i=i+1

    plt.show()

    

# With weather 1,2 and season 2,3 and working days the bicycle rental count is maximum.
# As per the analysis, we need to get rid off these variables to be inputted in our model: datetime,season,holiday,atemp,holiday

#(Working day) has better correlation with count, 

#weather,working day, hour,year has to be label encoded





final_df=new_df.drop(['datetime','season','holiday','atemp','holiday','windspeed','casual','registered','mnth+day','day'], axis=1)

final_df.head()



#adding dummy varibles

weather_df=pd.get_dummies(new_df['weather'],prefix='w')

#year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)

month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)

hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)

                     





final_df=final_df.join(weather_df)

#final_df=final_df.join(year_df)

final_df=final_df.join(month_df)                     

final_df=final_df.join(hour_df)

                     

final_df.head()

final_df.columns

model_df=final_df.drop(['workingday','month','hour','weather'],axis=1)

model_df.head()
# Now that we have got our guns lock and loaded, it's time to shoot.

#lets begin the modelling process.



X=model_df.iloc[:,model_df.columns!='count'].values





Y=model_df.iloc[:,2].values



print ('oye',X.shape)

#splitting the data into training and test data

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)















# choosing the appropriate model for regression

# we would be trying multiple linear regression, ploy linear regression, SVR, Decision Tree regression and RF regression

# Out of these, we would be choosing the one having the best accuracy and aplying GridSearchCV for optimal hyperparmater tuning. 



#Multiple linear regression



from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,Y_train)



# k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=lr,X=X_train,y=Y_train,scoring='r2',cv=10)

print (accuracies)

print (accuracies.mean())

from sklearn.preprocessing import PolynomialFeatures

# converts X matrix to the power of the degree provided which is 2. i.e converts X to Xsquared.

poly_reg=PolynomialFeatures(degree=2)

X_poly=poly_reg.fit_transform(X_train)

poly_reg.fit(X_poly,Y_train)

lin_reg_2=LinearRegression()

lin_reg_2.fit(X_poly,Y_train)



# k-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=lin_reg_2,X=X_train,y=Y_train,scoring='r2',cv=10)



print (accuracies.mean())

# using Random Forest

from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=300,random_state=0)

rf.fit(X,Y)











accuracies=cross_val_score(estimator=rf,X=X_train,y=Y_train,scoring='r2',cv=5)

print (accuracies)

print (accuracies.mean())
X.shape
#using SVR (Requres Feature scaling)



from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

sc_Y=StandardScaler()

X_temp=sc_X.fit_transform(X)

y_temp=sc_Y.fit_transform(Y.reshape(-1,1))



from sklearn.svm import SVR

svr=SVR(kernel='rbf')

svr.fit(X_temp,y_temp)



accuracies=cross_val_score(estimator=svr,X=X_temp,y=y_temp,scoring='r2',cv=5)

print (accuracies)

print (accuracies.mean())
from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(random_state=0)

dtr.fit(X_train,Y_train)







accuracies=cross_val_score(estimator=dtr,X=X_train,y=Y_train,scoring='r2',cv=5)

print (accuracies)

print (accuracies.mean())

# SVR and RandomForestRegressor are the ones having maximum accuracies.

#Lets try changing the hyperparameters of SVR to come to a more optimal solution



#Grid search (changing the hyperparamters for optimal solution)

from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

sc_Y=StandardScaler()

X_temp=sc_X.fit_transform(X_train)

y_temp=sc_Y.fit_transform(Y_train.reshape(-1,1))





parameters=[{'C':[1,10,100,1000],'kernel':['linear']},

            {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.01,0.001]}

            ]



grid_search= GridSearchCV(estimator=svr, param_grid=parameters, cv=5,n_jobs=-1)



"""

print 1

grid_search=grid_search.fit(X_temp,y_temp)

print 2

best_accuracy=grid_search.best_score_

best_parameters=grid_search.best_params_

print best_accuracy

print best_parameters

"""

import numpy as np

def rmsle(y, y_):

	#np.nan_to_num replaces nan with zero and inf with finite numbers

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

	#taking squares

    calc = (log1 - log2) ** 2

	#taking mean and then square

    return np.sqrt(np.mean(calc))





#predicting test data 



#train_predict_op=rf.predict(X)







#rmsle(Y,train_predict_op)



# FOllowing the same procedure for pre-processsing test data

test_df=pd.read_csv('../input/test.csv')

test_df['datetime']=test_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

new_df=test_df



new_df['month']=new_df['datetime'].apply(lambda x:x.month)

new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)

new_df['day']=new_df['datetime'].apply(lambda x:x.day)

#new_df['year']=new_df['datetime'].apply(lambda x:x.year)

#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)

#new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))



print (new_df.head())



test_df=new_df.drop(['datetime','season','holiday','atemp','holiday','windspeed','day'], axis=1)

test_df.head()

#adding dummy varibles

weather_df=pd.get_dummies(test_df['weather'],prefix='w',drop_first=True)

#yr_df=pd.get_dummies(test_df['year'],prefix='y',drop_first=True)

month_df=pd.get_dummies(test_df['month'],prefix='m',drop_first=True)

hour_df=pd.get_dummies(test_df['hour'],prefix='h',drop_first=True)
test_df=test_df.join(weather_df)

test_df=test_df.join(yr_df)

test_df=test_df.join(month_df)                     

test_df=test_df.join(hour_df)

                     

test_df.head()
test_df=test_df.drop(['workingday','month','hour','weather'],axis=1)
X_test=test_df.iloc[:,:].values

X_test.shape

"""

from sklearn.preprocessing import StandardScaler







Y_scaled=svr.predict(sc_X.transform(X_test))

Y_output=sc_Y.inverse_transform(Y_scaled)

print Y_output



"""



# Using the Ranfom Forest Classifier as it gives the maximum accuracy amongst the rest

y_output=rf.predict(X_test)

y_output





op=pd.DataFrame({'count':y_output})

op.to_csv('sub.csv')



#function to calculate RMSLE

def rmsle(y, y_):

	#np.nan_to_num replaces nan with zero and inf with finite numbers

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

	#taking squares

    calc = (log1 - log2) ** 2

	#taking mean and then square

    return np.sqrt(np.mean(calc))




