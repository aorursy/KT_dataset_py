# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



############# A Note that i have not removed some auxilary print commands which might make the notebook very long so plese comment if not needed ##################
a=pd.read_csv('/kaggle/input/jagritistatsimpact/train.csv',index_col=None)##to take in the data

del a["Id"]## not useful  as doent relate to any order of  any data

print(a)



countries=[]      #initializing the list of countries

data=[]           #initializing the list  of  data in each of the country 





a["Date"]=pd.to_datetime(a["Date"])          #to get it into the right format 

a = a.sort_values(by="Date")                 #arranging everything in chronological order to have it easy to understand the  cummulative data

day0=a["Date"].iloc[0]                       #to get the  first day of  the record to make calculation very easy

#dateno=lambda x: list(x.split(" ")[0].split("-"))

#daypass=lambda x,z: 





a["DayPassed"]=a["Date"]-day0                #To find the no of days passed

for i in range(len(a)):

    a["DayPassed"].iloc[i]=int(str(a["DayPassed"].iloc[i]).split(" ")[0])         #looping the the data to get the int value and as the hourly data is not  given 



    

a



dates=a["Date"]                 #Keeping the date data for later reference

del a["Date"]
def ratecha(a):   ## This function tries to return a approximate slope of the graph seen in the bottom with log applied to no of cases

    main=np.array(a.groupby("DayPassed")["ConfirmedCases"].sum(),dtype=np.float32) ## total no of patients in a country on each day

    if np.count_nonzero(main)==0:

        return (0,60)

    else:

        j=np.log(max(np.max(main),1))/(np.count_nonzero(main))   ## count_nonzero for the no of days that the country is suffering in the day given

        k=60-np.count_nonzero(main)

        return (min(j,20000),k)                     ## min and maxes are used to not too get a NaN or inf in the (Clipping) 



distCountri=a.groupby(['Country/Region'],sort=False)   ## to just get a distribution of countries



for i in distCountri:

    print(i)
residual=a["Province/State"]                   ##this data is not so important as the longitudes and the latitudes can take care about the region of count

del a["Province/State"]

for i in range(len(a)):

    #print(a["Country/Region"].iloc[i])        

    if a["Country/Region"].iloc[i] not in countries:

        countries.append(a["Country/Region"].iloc[i])           ## this to sort the data into country wise data scale I know it can be done by groupby class function but i choose not to

        data.append([])

    j=countries.index(a["Country/Region"].iloc[i])

    data[j].append(a.iloc[i])

print(countries)

for i in range(len(data)):

    data[i]=pd.DataFrame(data[i])            ## To convert the list of data into sub dataframes

data





l=list(zip(distCountri,countries,data))

l.sort(key=lambda x:ratecha(x[0][1]))                   ## by this the no that a country is directly proportional to the sort of rate that cases exists 

countries=[i[1] for i in l]                            ### The above step is of no use but maybe help ful to view the plot

data=[i[2] for i in l ]

distCountries=[i[0] for i in l]

print(countries)

print(data)
pd.read_csv('/kaggle/input/jagritistatsimpact/test.csv')        #just to check the test input

b=a

b["Day0"]=b["DayPassed"]

for i in range(len(b)):

    temp=ratecha((distCountries[countries.index(b["Country/Region"].iloc[i])])[1])## this  is useful for the model training

    b["Country/Region"].iloc[i]=temp[0]              ## stores the slope that I believe

    b["Day0"].iloc[i]=temp[1]                        ## stores the first date that the country was infected will help to get the y intercept of log plot

    

print(b)
data     #final data for representation

         ## see at the bottom    



b
import tensorflow as tf

from tensorflow import keras



y=np.array(b["ConfirmedCases"],dtype=np.float32)

del b["ConfirmedCases"]                               ## the training output

y
X=np.array(b,dtype=np.float32)

X

print(y.shape)
def norm(j):

    j1=(j-np.mean(j))

    return  j1/max(abs(j1))















X=X.T

print(X)

print(X.shape)



###############test######################

y=y.reshape((1,y.shape[0]))

np.random.seed(0)

rng_state = np.random.get_state()

np.random.shuffle(X[0,:])

np.random.set_state(rng_state)

np.random.shuffle(X[1,:])

np.random.set_state(rng_state)

np.random.shuffle(X[2,:])

np.random.set_state(rng_state)

np.random.shuffle(X[3,:])

np.random.set_state(rng_state)

np.random.shuffle(X[4,:])

np.random.set_state(rng_state)

np.random.shuffle(y[0,:])



lat2=X[1,:]*X[1,:]

lat2=lat2.reshape(1,lat2.shape[0])

long2=X[2,:]*X[2,:]

long2=long2.reshape(1,long2.shape[0])

latlong=X[2,:]*X[1,:]

latlong=latlong.reshape(1,latlong.shape[0])

X=np.append(X,lat2,axis=0)

X=np.append(X,long2,axis=0)

X=np.append(X,latlong,axis=0)

## I believe normalization is not neccessary

#X[0,:]=norm(X[0,:])

#X[1,:]=norm(X[1,:])

#X[2,:]=norm(X[2,:])

#X[3,:]=norm(X[3,:])

#X[4,:]=norm(X[4,:])

#X[5,:]=norm(X[5,:])

#X[6,:]=norm(X[6,:])



print(X)

X_train=X

#X_test=X[:,10000:]

print(X_train.shape)

y_train=y

y_tlog=np.log(y_train+1)   ## this is done so as the random tree would not need to learn a exponential functional as it is a exponential function

                           ## As can be seen from the plot as if the relation is made with log of no of Cases it will be straight line

                           ## Which is easier for a random tree to learn than a exponential curve

                           ## 1 is added so that log(0) error should not  occur for which one is also subtracted at the prediction

#y_test=y[:,10000:]
####### These are of no  use though i did try to ensemble them tried a lot of of architecture didnt work well #######

tf.random.set_seed(2)

np.random.seed(2)

model1 = keras.Sequential([

        keras.layers.Dense(14,input_dim=7,activation="relu"),

        keras.layers.Dense(14,activation="relu"),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(7),

        keras.layers.ReLU(),

        keras.layers.Dense(1,activation="linear"),

        

])





model1.compile(loss='mse', optimizer="adam", metrics=['mse','mae'])



tf.random.set_seed(3)

np.random.seed(3)

model2 = keras.Sequential([

        keras.layers.Dense(14,input_dim=7,activation="relu"),

        keras.layers.Dense(14,activation="relu"),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(7),

        keras.layers.ReLU(),

        keras.layers.Dense(1,activation="linear"),

        

])





model2.compile(loss='mse', optimizer="adam", metrics=['mse','mae'])



tf.random.set_seed(13)

np.random.seed(13)

model3 = keras.Sequential([

        keras.layers.Dense(20,input_dim=7,activation="relu"),

        keras.layers.Dense(30,activation="relu"),

        keras.layers.Dense(20),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(20,activation="relu"),

        keras.layers.Dense(15),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(15,activation="relu"),

        keras.layers.Dense(15),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(7),

        keras.layers.ReLU(),

        keras.layers.Dense(1,activation="linear"),

        

])





model3.compile(loss='mse', optimizer="adam", metrics=['mse','mae'])



tf.random.set_seed(13)

np.random.seed(13)

model4 = keras.Sequential([

        keras.layers.Dense(14,input_dim=7,activation="relu"),

        keras.layers.Dense(14,activation="relu"),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(7),

        keras.layers.ReLU(),

        keras.layers.Dense(1,activation="linear"),

        

])





model4.compile(loss='mse', optimizer="adam", metrics=['mse','mae'])



tf.random.set_seed(9)

np.random.seed(9)

model5 = keras.Sequential([

        keras.layers.Dense(14,input_dim=7,activation="relu"),

        keras.layers.Dense(14,activation="relu"),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(7),

        keras.layers.ReLU(),

        keras.layers.Dense(1,activation="linear"),

        

])





model5.compile(loss='mse', optimizer="adam", metrics=['mse','mae'])



tf.random.set_seed(6)

np.random.seed(6)

model6 = keras.Sequential([

        keras.layers.Dense(14,input_dim=7,activation="relu"),

        keras.layers.Dense(14,activation="relu"),

        keras.layers.Dense(10),

        keras.layers.LeakyReLU(alpha=0.2),

        keras.layers.Dense(7),

        keras.layers.ReLU(),

        keras.layers.Dense(1,activation="linear"),

        

])





model6.compile(loss='mse', optimizer="adam", metrics=['mse','mae'])



#keras.metrics.RootMeanSquaredError(name='rmse')



############ End off NN############
#model1.fit(X_train.T,y_train.T,epochs=4000)

#model2.fit(X_train.T,y_train.T,epochs=4000)

#model3.fit(X_train.T,y_train.T,epochs=4000)

#model4.fit(X_train.T,y_train.T,epochs=4000)

#model5.fit(X_train.T,y_train.T,epochs=4000)

#model6.fit(X_train.T,y_train.T,epochs=4000)

from sklearn.ensemble import RandomForestRegressor

regr1 = RandomForestRegressor(n_estimators = 2000, random_state = 5)  ### n_estimators were best at 2000 as it was the best output using a test set

regr1 = regr1.fit(X_train.T,y_tlog.T)   ### The regresoin is to fit the log and not directly the no of cases 

#y_plog = regr1.predict(X_test.T)

#from sklearn import metrics

#y_pred = np.exp(y_plog)-1

#print('Accuracy Score:', metrics.mean_squared_error(y_test.T,y_pred))
#model1.save("/kaggle/working/my_model1.h5")

#model2.save("/kaggle/working/my_model2.h5")

#model3.save("/kaggle/working/my_model3.h5")

#model4.save("/kaggle/working/my_model4.h5")

#model5.save("/kaggle/working/my_model5.h5")

#model6.save("/kaggle/working/my_model6.h5")
#model1 = keras.models.load_model('/kaggle/working/my_model.h5')
#from sklearn.linear_model import LinearRegression

#from sklearn import metrics

#reg=LinearRegression()

#reg.fit(X_train.T,y_train.T)

#pred=reg.predict(X_test.T)

#print(metrics.mean_squared_error(y_test.T,pred))
#model1.fit(X_test.T,y_test.T,epochs=5000)
test=pd.read_csv("/kaggle/input/jagritistatsimpact/test.csv")  ## inputing the test data

test                     
del test["Id"]

residual_test=test["Province/State"]

del test["Province/State"]

test                                             ### deleting the irrelevent stuff
test["Date"]=pd.to_datetime(test["Date"])          #to get it into the right format 



test["DayPassed"]=test["Date"]-day0                #To find the no of days passed

for i in range(len(test)):

    test["DayPassed"].iloc[i]=int(str(test["DayPassed"].iloc[i]).split(" ")[0])         #looping the the data to get the int value and as the hourly data is not  given 



    

test



dates=test["Date"]                 #Keeping the date data for later reference

del test["Date"]



test

testX=test

testX["Day0"]=testX["DayPassed"]

for i in range(len(testX)):

    temp=ratecha((distCountries[countries.index(testX["Country/Region"].iloc[i])])[1])

    testX["Country/Region"].iloc[i]=temp[0]

    testX["Day0"].iloc[i]=temp[1]                                               ## trying to  replicate the data manipulation done to training data

print(testX)
X=np.array(testX,dtype=np.float32)

X

lat2=X[:,1]*X[:,1]

lat2=lat2.reshape(lat2.shape[0],1)

long2=X[:,2]*X[:,2]

long2=long2.reshape(long2.shape[0],1)

latlong=X[:,2]*X[:,1]

latlong=latlong.reshape(latlong.shape[0],1)

X=np.append(X,lat2,axis=1)

X=np.append(X,long2,axis=1)

X=np.append(X,latlong,axis=1)

print(X.shape)
#pred=(model1.predict(X)+model2.predict(X)+model3.predict(X)+model4.predict(X)+model5.predict(X)+model6.predict(X))/6
pred=regr1.predict(X)

print(pred)              ## prediction
for i in pred:

    print(i)
predx=(np.exp(pred)-1)   ### here the prediction is taken to e's  power and subtracting 1 to remove zero error (error analysis :) )  

print(predx)            ## np.around is can be taken so as to make int value out of the float value in order to get better accuracy which is not true 

                        ## its better not to take np.around()

#print(sum(keras.losses.MSE(predx,predi)))
p=pd.read_csv("/kaggle/input/jagritistatsimpact/submission.csv")

p
o=dict({'ConfirmedCases':predx})

odf=pd.DataFrame(predx)             ##trying to convert  

odf[0]
p["ConfirmedCases"]=odf 
p
p.to_csv("/kaggle/working/submission.csv",index=False)   ##saving the file


import matplotlib.pyplot as plt  ### no  use of this
print(np.array(data[0].groupby('DayPassed')['ConfirmedCases'].sum()))

print(np.array( [i for i in set(data[0]['DayPassed'])] ,dtype=np.uint8 ))



for i in range(len(countries)):

    #plt.cle

    data[i].groupby('DayPassed')['ConfirmedCases'].sum().plot()     ## this plots expnential curve of total no of cases in a day vs Daypasses

    #plt.scatter((np.array((data[i].groupby('DayPassed')['ConfirmedCases']).sum())),np.array( [i for i in set(data[0]['DayPassed'])] ,dtype=np.uint8 ) )

    #plt.xlabel('DayPassed')

    #plt.ylabel(countries[i])

    #plt.show()
# the discountinues parts are the lack of data from maybe some province of certain part of the country



## which is asked in test data 
j=1

print(np.array(data[j].groupby('DayPassed')['ConfirmedCases'].sum()))

print(np.array( [i for i in set(data[j]['DayPassed'])] ,dtype=np.uint8 ))



for i in range(len(countries)):

    #plt.cle

    data[i].groupby('DayPassed')['ConfirmedCases'].sum().plot(logx=True)            ## this plots log(total no of cases) in a day vs Day passed
#Applying a log transmform every country became a straight line from the day they started so a exponential function for day passed 

## no of cases proportional to e^(daypassed-day0_for_that_country )*W_for_that_country

## I tried approximating W by the function ratecha()

## W=y2-y1/x2-x1

## W=log(maximum no of no of cases as it would only be on the last day of the data as the curve is not decreasing)/(no of days it took to reach there)

## as W is country specific

# from day0 of each country while give a good data representation of y intercept to predict the straight line without the use of lot of previous data

##as there is no identification when is day0 of the country as there is no very good way that the algorithm can make it remember

##I set that as a parameter manually  i did this being inspired by LSTM
set(b["Day0"])  ## this just to check 
####Gut Feeling###  ##3 this didnt work out so well
predic=np.exp(X[:,0]*(X[:,3]-X[:,4]))

print(pred)
p=pd.read_csv("/kaggle/input/jagritistatsimpact/submission.csv")

p
odf=pd.DataFrame(predic)

odf[0]
p["ConfirmedCases"]=odf
p.to_csv("/kaggle/working/submissiongut.csv",index=False)
#This gut feeling didnt work out that well but got 1.2 accuracy
