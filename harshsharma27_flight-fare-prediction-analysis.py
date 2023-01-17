import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df=pd.read_excel(r'../input/flight-fare-prediction-mh/Data_Train.xlsx')
df.keys()
pd.set_option('display.max_columns',None)    #To Display All Columns in dataframe
df.head()
df.tail()
df.info()
df.describe()
#To Check For Null Values :

df.isnull().sum()
#Wow, we only have a few null values..so can be easily handled..so we can drop these values ..since it wont affect our dataset

#upto large extent
df.dropna(inplace=True)
df.isnull().sum()
df['Airline'].value_counts()
df['Source'].value_counts()
df['Destination'].value_counts()
df['Total_Stops'].value_counts()
df['Duration'].value_counts()
#Since Duration column is in string format , we need to preprocess it for our machine learning model to understand
df['Date_of_Journey'].value_counts()
#These values are basically string type , so we need to convert them into datetime -> timestamp format

#i am gonna use pandas to_datetime function for this 
df['Journey_Of_Day']=pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.day
df['Journey_Of_Month']=pd.to_datetime(df['Date_of_Journey'],format="%d/%m/%Y").dt.month
#since the dates here are of same year ..i.e.,2019 , ..there is no need to specifiy a seprate column feature for the same 
df.head()
#Since we have timestamp format now , we dont need date of journey column..so we can drop it now 

df.drop(['Date_of_Journey'],inplace=True,axis=1)
df
#Similarly , we need to preprocess Departure Time since it is in string format..we need to get hours and minutes separately ..



df['Dep_Hour']=pd.to_datetime(df['Dep_Time']).dt.hour

df['Dep_Min']=pd.to_datetime(df['Dep_Time']).dt.minute
df
#Let us drop the Dep_time now , since we have already preprocessed it :



df.drop('Dep_Time',axis=1,inplace=True)
df
#Similarly , we have to preprocess Arrival Time Data , being text  , to convert into datetime->timestamp format



df['Arrival_Hour']=pd.to_datetime(df['Arrival_Time']).dt.hour

df['Arrival_Min']=pd.to_datetime(df['Arrival_Time']).dt.minute
df
#Now , drop Arrival_time :



df.drop('Arrival_Time',axis=1,inplace=True)
df
#Now we Have Duration , which is differece between arrival and departure time 

#This is a bit tricky pre-processing Technique.

#Since if you take a look at duration column , we have something like '2h 50m' this is a string which consists of 'h' and 'm' as well

#for our model to understand 'h' and 'm' as 'hours' and 'minutes' , we need to apply some iterative logic :









duration=list(df['Duration'])



for i in range(len(duration)):

  if(len(duration[i].split()))!=2:                             #To check if duration has both hours and minutes , if yes , implement the below logic

    if "h" in duration[i]:

      duration[i]=duration[i].strip()+" 0m"

    else:

      duration[i]="0h "+duration[i]             





duration_hours=[]

duration_min=[]



for i in range(len(duration)):

  duration_hours.append(int(duration[i].split(sep='h')[0]))                                #To Get Hours in Duration

  duration_min.append(int(duration[i].split(sep='m')[0].split()[-1]))                      #To Get Minutes in Duration

#For testing only , not involved in actual code



samp="30h"

l1=list(samp)

temp=[]

for i in range(len(list(samp))):

  if (len(list(samp)[i].split()))!=2:

    if "h" in list(samp)[i]:

      list(samp)[i]=list(samp)[i].strip()+" 0m"

      temp.append(list(samp))

    else:

      list(samp)[i]="0h "+list(samp)[i]

      temp.append(list(samp)[i])

##cole=[]

#for i in range(len(list(samp))):

 # cole.append(int(list(samp)[i].split(sep='h')[0]))

duration_hours
duration_min
df['Duration_Hour']=duration_hours

df['Duration_Min']=duration_min
df
df.drop('Duration',axis=1,inplace=True)
df
df['Airline'].value_counts()
#To check The most bought airlines

plt.figure(figsize=(26,12))

plt.title('To check The most bought airlines',fontsize=20)

sns.set_style('darkgrid')

sns.boxplot(y='Price',x="Airline",data=df.sort_values('Price',ascending=False))
#Airline - > Nominal Data -> OneHotEncoding

Airline=df['Airline']

Airline=pd.get_dummies(Airline,drop_first=True)

Airline.head()
df['Source'].value_counts()
#Source VS Price

plt.figure(figsize=(26,12))

plt.title('Source VS Price Plot',fontsize=20)

sns.set_style('darkgrid')

sns.boxplot(y='Price',x="Source",data=df.sort_values('Price',ascending=False))
#Onehotencoding for Source :

Source=df[['Source']]

Source=pd.get_dummies(Source,drop_first=True)

Source.head()
df['Destination'].value_counts()
#Destination VS Price

plt.figure(figsize=(26,12))

plt.title('Destination VS Price Plot',fontsize=20)

sns.set_style('darkgrid')

sns.boxplot(y='Price',x="Destination",data=df.sort_values('Price',ascending=False))
#Destination -> OneHotEncoding:

Destination=df[['Destination']]

Destination=pd.get_dummies(Destination,drop_first=True)

Destination.head()



df['Route'].value_counts()
df['Additional_Info'].value_counts()
#Route and Total stops convey almost same info

#Addtional Info has no significant info for our model , and also it has almost 80% of info as - "No_info"

#So we can drop some unncessary features



df.drop(['Additional_Info','Route'],axis=1,inplace=True)
df
df['Total_Stops'].value_counts()
#Total Stops VS Price

plt.figure(figsize=(26,12))

plt.title('Total Stops VS Price Plot',fontsize=20)

sns.set_style('darkgrid')

sns.boxplot(y='Price',x="Total_Stops",data=df.sort_values('Price',ascending=False))
#The Data is in somewhat Order , therefore . Ordinal data - > Label Encoder should be used :



df.replace({

    "non-stop":0,

    "1 stop":1,

    "2 stops":2,

    "3 stops":3,

    "4 stops":4

},inplace=True)
df.head()
#Now , we have to add all those enocded fetaures to our main dataframe :

df=pd.concat([df,Airline,Source,Destination],axis=1)
df.head()
#Since we have already included encoded values of Airline , Source And Destination , we should drop their textual values



df.drop(['Airline','Source','Destination'],axis=1,inplace=True)
df.head()
#Now Our train data Is Purely converted into Numerical Format :

df.shape
test_data=pd.read_excel(r"../input/flight-fare-prediction-mh/Test_set.xlsx")
test_data.head()
#So Basically , we have to repeat our Pre-processing steps again , like the same steps we did earlier in Our Train set :
test_data.info()
test_data.describe()
test_data.isnull().sum()  #To Check For Null Values
test_data.dropna(inplace=True)
# Date_of_Journey

test_data["Journey_Of_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day

test_data["Journey_Of_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month

test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time

test_data["Dep_Hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour

test_data["Dep_Min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute

test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time

test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour

test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute

test_data.drop(["Arrival_Time"], axis = 1, inplace = True)
# Duration

duration = list(test_data["Duration"])



for i in range(len(duration)):

    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins

        if "h" in duration[i]:

            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute

        else:

            duration[i] = "0h " + duration[i]           # Adds 0 hour



duration_hours = []

duration_mins = []

for i in range(len(duration)):

    duration_hours.append(int(duration[i].split(sep = "h")[0]))    

    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1])) 


# Adding Duration column to test set

test_data["Duration_hours"] = duration_hours

test_data["Duration_mins"] = duration_mins

test_data.drop(["Duration"], axis = 1, inplace = True)
print("Airline")

print("-"*75)

print(test_data["Airline"].value_counts())

Airline = pd.get_dummies(test_data["Airline"], drop_first= True)



print()



print("Source")

print("-"*75)

print(test_data["Source"].value_counts())

Source = pd.get_dummies(test_data["Source"], drop_first= True)



print()



print("Destination")

print("-"*75)

print(test_data["Destination"].value_counts())

Destination = pd.get_dummies(test_data["Destination"], drop_first = True)


# Route and Total_Stops are related to each other

test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)



# Replacing Total_Stops

test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)



# Concatenate dataframe --> test_data + Airline + Source + Destination

data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)



data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)



#So BASICALLY , i have created a new dataframe -> data_test , with all encoded values and necessary features ONLY
data_test.shape
data_test.head()
df.shape
df.columns
#Selecting All Independent Features :



X=df.loc[:,['Total_Stops', 'Journey_Of_Day', 'Journey_Of_Month',

       'Dep_Hour', 'Dep_Min', 'Arrival_Hour', 'Arrival_Min', 'Duration_Hour',

       'Duration_Min', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways',

       'Jet Airways Business', 'Multiple carriers',

       'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara',

       'Vistara Premium economy', 'Source_Chennai', 'Source_Delhi',

       'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',

       'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',

       'Destination_New Delhi']]
X.head()
#Selecting My Target / Dependent Feature :

y=df.iloc[:,1]
y.head()
df.corr()
plt.figure(figsize=(18,18))

sns.heatmap(df.corr(),annot=True)
#With Respect To Target Value:

df.corr()['Price'].sort_values(ascending=False)
#Before Model Selection , we will try to Exlpore The Correlations Between Different Features , For our Output Variable , Which Are Important.

from sklearn.ensemble import ExtraTreesRegressor
selection=ExtraTreesRegressor()
selection.fit(X,y)
important_feat=selection.feature_importances_
pd.Series(important_feat,index=X.columns)
important_feats=pd.Series(important_feat,index=X.columns)
#To Print Top 20 Highly Correlated Features :

important_feats.nlargest(20)
#Graphical Visualisation :

plt.figure(figsize=(12,8))

important_feats.nlargest(20).plot(kind='barh')
#We can see that  , Total_Stops Play AN very important Feature
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
rfc_model=RandomForestRegressor()

rfc_model.fit(X_train,y_train)
pred=rfc_model.predict(X_test)
pred
comp_df=pd.DataFrame({'Actual Values':y_test,'Predicted Values':pred})
comp_df
#To Display The Score Of Our Model On Train Set :

rfc_model.score(X_train,y_train)
#To Display The Score Of Our Model On Test Set :

rfc_model.score(X_test,y_test)
#DistPlot  For Original Values VS Prediccted Values :

sns.distplot(y_test-pred)
plt.scatter(y_test,pred,alpha=0.5)

plt.xlabel('Original Values : y_test')

plt.ylabel('Predicted Values : pred')
from sklearn import metrics





print('MAE:', metrics.mean_absolute_error(y_test, pred))

print('MSE:', metrics.mean_squared_error(y_test, pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print("R2 Score Of Our Model Is : ")

print()

metrics.r2_score(y_test, pred)
#In The Next Section , Let Us Try To Improve The Score oF Our Model , Using HyperParameter Tuning
from sklearn.model_selection import RandomizedSearchCV
test=np.linspace(start=100,stop=1000,num=20)

test
#Defining Hyperparameters Grid:



n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)] 

 

max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=6) ]

min_samples_split=[2, 5, 10, 15,20, 100] 

min_samples_leaf=[1, 2, 5, 10] 

max_features=['auto', 'sqrt']

#Creating Random GRID :



random_grid={

             'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf

}
rf_random=RandomizedSearchCV(estimator=rfc_model,param_distributions=random_grid,

                             scoring='neg_mean_squared_error',n_iter=15,n_jobs=1,cv=5,verbose=2,

                             random_state=42)
rf_random.fit(X_train,y_train)
rf_random.best_params_
pred2=rf_random.predict(X_test)
pred2
plt.figure(figsize=(12,8))

plt.title('Y_test VS RandomisedSearch CV Predictions')

sns.distplot(y_test-pred2)
plt.figure(figsize = (12,8))

plt.scatter(y_test, pred2, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("RCV predictions")

plt.title("Y_test VS Predictions RCV Scatterplot")
print('MAE:', metrics.mean_absolute_error(y_test, pred2))

print('MSE:', metrics.mean_squared_error(y_test, pred2))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred2)))
print("R2 Score Of Our Model Is : ")

print()

metrics.r2_score(y_test, pred2)
#Since our model's performance has been increased ..We can export it now for our future refernces
import pickle
my_file=open('flight_fa.pkl','wb')
pickle.dump(rf_random,my_file)
#Let us test whether our loading of model has been done or not..



model=open('flight_fa.pkl','rb')

forest=pickle.load(model)
pred3=forest.predict(X_test)

metrics.r2_score(y_test,pred3)
# Somewhat our model is performing Quite Well.