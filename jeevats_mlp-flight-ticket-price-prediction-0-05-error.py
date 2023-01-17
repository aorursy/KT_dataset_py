import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


# Read in train data
df_train = pd.read_excel('Data_Train.xlsx')



#descriptive statistics summary
df_train['Price'].describe()

#histogram
sns.distplot(df_train['Price']);

#skewness and kurtosis
print("Skewness: %f" % df_train['Price'].skew())
print("Kurtosis: %f" % df_train['Price'].kurt())







#data cleaning using python
temp=df_train
df_train.columns.values.tolist()


df_train.dropna(inplace = True) 
  
# new df_train frame with split value columns 
new = df_train["Route"].str.split("→", n = 5, expand = True) 


new[5]
# making seperate first name column from new df_train frame 
df_train["Route_Path1"]= new[0] 
df_train["Route_Path2"]= new[1] 
df_train["Route_Path3"]= new[2] 
df_train["Route_Path4"]= new[3] 
df_train["Route_Path5"]= new[4] 
  
# making seperate last name column from new df_train frame 
#df_train["Last Name"]= new[1] 
  
# Dropping old Name columns 
df_train.drop(columns =["Route"], inplace = True) 
  
# df display 
df_train 

le = preprocessing.LabelEncoder()
li=[]
mylist = set(df_train.Airline)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Date_of_Journey)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Source)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Destination)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)



df_train.Route_Path1 = le.fit_transform(df_train.Route_Path1.astype(str))
df_train.Route_Path2 = le.fit_transform(df_train.Route_Path2.astype(str))
df_train.Route_Path3 = le.fit_transform(df_train.Route_Path3.astype(str))
df_train.Route_Path4 = le.fit_transform(df_train.Route_Path4.astype(str))
df_train.Route_Path5 = le.fit_transform(df_train.Route_Path5.astype(str))

mylist = set(df_train.Route_Path1)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Route_Path2)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Route_Path3)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Route_Path4)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_train.Route_Path5)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_train.Dep_Time = le.fit_transform(df_train.Dep_Time.astype(str))

mylist = set(df_train.Dep_Time)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_train.Arrival_Time = le.fit_transform(df_train.Arrival_Time.astype(str))

mylist = set(df_train.Arrival_Time)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_train.Duration = le.fit_transform(df_train.Duration.astype(str))

mylist = set(df_train.Duration)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)


df_train.Total_Stops = le.fit_transform(df_train.Total_Stops.astype(str))
mylist = set(df_train.Total_Stops)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_train.Additional_Info = le.fit_transform(df_train.Additional_Info.astype(str))

mylist = set(df_train.Additional_Info)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)
temp1=df_train




df_train.columns.values.tolist()
li
#df_train.to_excel("./midified.xlsx")
                                    
df_train=temp1
df_train.Airline=df_train.Airline.replace( li[0], list(range(len(li[0]))))
df_train.Date_of_Journey=df_train.Date_of_Journey.replace( li[1], list(range(len(li[1]))))
df_train.Source=df_train.Source.replace( li[2], list(range(len(li[2]))))
df_train.Destination=df_train.Destination.replace( li[3], list(range(len(li[3]))))

df_train.Dep_Time=df_train.Dep_Time.replace( li[9], list(range(len(li[9]))))
df_train.Arrival_Time=df_train.Arrival_Time.replace( li[10], list(range(len(li[10]))))
df_train.Duration=df_train.Duration.replace( li[11], list(range(len(li[11]))))

df_train.Additional_Info=df_train.Additional_Info.replace( li[13], list(range(len(li[13]))))
df_train.Route_Path1=df_train.Route_Path1.replace( li[4], list(range(len(li[4]))))
df_train.Route_Path2=df_train.Route_Path2.replace( li[5], list(range(len(li[5]))))
df_train.Route_Path3=df_train.Route_Path3.replace( li[6], list(range(len(li[6]))))
df_train.Route_Path4=df_train.Route_Path4.replace( li[7], list(range(len(li[7]))))
df_train.Route_Path5=df_train.Route_Path5.replace( li[8], list(range(len(li[8]))))
df_train.Total_Stops=df_train.Total_Stops.replace( li[12], list(range(len(li[12]))))
temp2=df_train
df_train
df_train.to_excel("train_modified.xlsx")
#scatterplot

sns.set()
cols = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
 'Additional_Info', 'Route_Path1', 'Route_Path2', 'Route_Path3', 'Route_Path4', 'Route_Path5']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

df_train = df_train.fillna(df_train.mean())

#standardizing data
Price_scaled = StandardScaler().fit_transform(df_train['Price'][:,np.newaxis]);
low_range = Price_scaled[Price_scaled[:,0].argsort()][:10]
high_range= Price_scaled[Price_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


df_train = temp2

cols = [ 'Price','Airline', 'Date_of_Journey', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
 'Additional_Info', 'Route_Path1', 'Route_Path2', 'Route_Path3', 'Route_Path4', 'Route_Path5']
df_train = df_train[cols]
# Create dummy values
df_train = pd.get_dummies(df_train)
#filling NA's with the mean of the column:
df_train = df_train.fillna(df_train.mean())
# Always standard scale the data before using NN
scale = StandardScaler()
X_train = df_train[['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
 'Additional_Info', 'Route_Path1', 'Route_Path2', 'Route_Path3', 'Route_Path4', 'Route_Path5']]
X_train = scale.fit_transform(X_train)
# Y is just the 'Price' column
y = df_train['Price'].values
seed = 7
np.random.seed(seed)
# split into 67% for train and 33% for test
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.33, random_state=seed)


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(90, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer ='adam', loss = 'mean_squared_error', 
              metrics =[metrics.mae])
    return model

model = create_model()
model.summary()
print(X_train,y_train)
print(X_test,y_test)
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=300, batch_size=32)


regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)

predictions = model.predict(X_test) 
y_test
#model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) 
predictn= regr.predict(X_test)
print_model = model.summary()
print(print_model)

predictn= model.predict(X_test)


from sklearn.metrics import confusion_matrix

rmse = sqrt(mean_squared_error(y_test, predictn))


model.summary()

rmse
#
#from sklearn import metrics as ms
mse=mean_squared_error(y_test, predictn)
print("Mean Square Error",mse)
#print("Accuracy:", ms.accuracy_score(y_test,predictn))
# summarize history for accuracy
plt.plot(y_test)
plt.plot(predictn)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test original', 'prediction'], loc='upper left')
plt.show()
# summarize 


# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
import pickle
path= "300_120.pkl"
with open(path, 'wb') as f:
        pickle.dump(model, f)
        print("Done Pickiling")
        #print("Pickled clf at {}".format(path))
"""with open("./two_200_multiple_layer_epoch.pkl", 'rb') as f:
            regressor = pickle.load(f)"""
df_test = pd.read_excel('Test_set.xlsx')


df_test.dropna(inplace = True) 
  
# new df_test frame with split value columns 
new = df_test["Route"].str.split("→", n = 5, expand = True) 
# making seperate first name column from new df_test frame 
df_test["Route_Path1"]= new[0] 
df_test["Route_Path2"]= new[1] 
df_test["Route_Path3"]= new[2] 
df_test["Route_Path4"]= new[3] 
df_test["Route_Path5"]= new[4] 
  
# making seperate last name column from new df_test frame 
#df_test["Last Name"]= new[1] 
  
# Dropping old Name columns 
df_test.drop(columns =["Route"], inplace = True) 
  
# df display 
df_test 

le = preprocessing.LabelEncoder()
li=[]
mylist = set(df_test.Airline)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Date_of_Journey)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Source)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Destination)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)



df_test.Route_Path1 = le.fit_transform(df_test.Route_Path1.astype(str))
df_test.Route_Path2 = le.fit_transform(df_test.Route_Path2.astype(str))
df_test.Route_Path3 = le.fit_transform(df_test.Route_Path3.astype(str))
df_test.Route_Path4 = le.fit_transform(df_test.Route_Path4.astype(str))
df_test.Route_Path5 = le.fit_transform(df_test.Route_Path5.astype(str))

mylist = set(df_test.Route_Path1)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Route_Path2)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Route_Path3)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Route_Path4)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

mylist = set(df_test.Route_Path5)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_test.Dep_Time = le.fit_transform(df_test.Dep_Time.astype(str))

mylist = set(df_test.Dep_Time)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_test.Arrival_Time = le.fit_transform(df_test.Arrival_Time.astype(str))

mylist = set(df_test.Arrival_Time)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_test.Duration = le.fit_transform(df_test.Duration.astype(str))

mylist = set(df_test.Duration)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)


df_test.Total_Stops = le.fit_transform(df_test.Total_Stops.astype(str))
mylist = set(df_test.Total_Stops)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)

df_test.Additional_Info = le.fit_transform(df_test.Additional_Info.astype(str))

mylist = set(df_test.Additional_Info)
myset = list(mylist)
myval=sorted(myset)
li.append(myval)
temp1=df_test

                                    
df_test=temp1
df_test.Airline=df_test.Airline.replace( li[0], list(range(len(li[0]))))
df_test.Date_of_Journey=df_test.Date_of_Journey.replace( li[1], list(range(len(li[1]))))
df_test.Source=df_test.Source.replace( li[2], list(range(len(li[2]))))
df_test.Destination=df_test.Destination.replace( li[3], list(range(len(li[3]))))

df_test.Dep_Time=df_test.Dep_Time.replace( li[9], list(range(len(li[9]))))
df_test.Arrival_Time=df_test.Arrival_Time.replace( li[10], list(range(len(li[10]))))
df_test.Duration=df_test.Duration.replace( li[11], list(range(len(li[11]))))

df_test.Additional_Info=df_test.Additional_Info.replace( li[13], list(range(len(li[13]))))

df_test.Route_Path1=df_test.Route_Path1.replace( li[4], list(range(len(li[4]))))
df_test.Route_Path2=df_test.Route_Path2.replace( li[5], list(range(len(li[5]))))
df_test.Route_Path3=df_test.Route_Path3.replace( li[6], list(range(len(li[6]))))
df_test.Route_Path4=df_test.Route_Path4.replace( li[7], list(range(len(li[7]))))
df_test.Route_Path5=df_test.Route_Path5.replace( li[8], list(range(len(li[8]))))
df_test.Total_Stops=df_test.Total_Stops.replace( li[12], list(range(len(li[12]))))

temp2=df_test
df_test

df_test.columns.values.tolist()
df_test.to_excel("test_modified.xlsx")
cols = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Dep_Time', 'Arrival_Time', 'Duration', 'Total_Stops',
 'Additional_Info', 'Route_Path1', 'Route_Path2', 'Route_Path3', 'Route_Path4', 'Route_Path5']

df_test['Arrival_Time'] = np.log1p(df_test['Arrival_Time'])
df_test = pd.get_dummies(df_test)
df_test = df_test.fillna(df_test.mean())
X_test = df_test[cols].values

# Always standard scale the data before using NN
scale = StandardScaler()
X_test = scale.fit_transform(X_test)


prediction = model.predict(X_test)



#prediction=prediction.astype(int)
prediction
submission = pd.DataFrame()
submission['Price'] = prediction[:,0]



submission.to_excel("output.xlsx")
plt.plot(submission.Price)
plt.show()
