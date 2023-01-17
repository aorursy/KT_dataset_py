import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')
test = pd.read_csv('../input/used-cars-price-prediction/test-data.csv')
print(train.shape)
train.head()
print(test.shape)
test.head()
train = train.rename(columns = {'Unnamed: 0': 'id'})
train.head()
#train = train.drop(["Unnamed: 0"], axis = 1)
#train.head()

#train_data = train_data.iloc[:,1:]
train.info()
train.describe()
train.isnull().any()
train.isnull().sum()
(train.isnull().sum() / len(train)) * 100
print("Shape of train data Before dropping any Row: ",train.shape)
train = train[train['Mileage'].notna()]
print("Shape of train data After dropping Rows with NULL values in Engine : ",train.shape)
train = train[train['Power'].notna()]
print("Shape of train data After dropping Rows with NULL values in Power  : ",train.shape)
train = train[train['Seats'].notna()]
print("Shape of train data After dropping Rows with NULL values in Seats  : ",train.shape)
train = train.reset_index(drop=True)
train.columns
print(train['Name'].unique())
print(train['Location'].unique())
print(train['Fuel_Type'].unique())
print(train['Transmission'].unique())
print(train['Owner_Type'].unique())
train['Name'].value_counts()
train['Name'] = train['Name'].str.split().str.get(0)
train.head()
train['Name'].unique()
train.Name[train.Name == 'Isuzu'] = 'ISUZU'
train['Name'].unique()
train['Fuel_Type'].value_counts()
train['Fuel_Type'] = train['Fuel_Type'].replace(['Diesel','Petrol', 'CNG', 'LPG'], [0, 1, 2, 3])
train.head()
train['Transmission'].value_counts()
train['Transmission'] = train['Transmission'].replace(['Manual','Automatic'], [0, 1])
train.head()
train['Owner_Type'].value_counts()
train['Owner_Type'] = train['Owner_Type'].replace(['First','Second', 'Third', 'Fourth & Above'], [0, 1, 2, 3])
train.head()
train['Mileage'].value_counts()
mileage_split = train['Mileage'].str.split(" ")

train['Mileage'] = mileage_split.str.get(0)
train.head()

#train.Mileage = train.Mileage.str.split().str.get(0).astype('float')
train['Engine'].value_counts()
train['Engine'] = train['Engine'].str.strip(' CC').astype(float)
train.head()

#train.Engine = train.Engine.str.split().str.get(0).astype('int', errors='ignore')
train['Power'].value_counts()
power_split = train['Power'].str.split(" ")

train['Power'] = power_split.str.get(0)
train.head()

#train.Power = train.Power.str.split().str.get(0).astype('float', errors='ignore')
train['Power'].str.contains("null")
position = []
for i in range(train.shape[0]):
    if train['Power'][i]=='null':
        position.append(i)
        
train = train.drop(train.index[position])
train = train.reset_index(drop=True) 

train.head()
train['Name'].value_counts()
train['Name'].describe()
# 원핫 인코딩

name = pd.get_dummies(train['Name'], prefix='Name')
name.head()
train = pd.concat([train,name], axis=1)
train.head()
train['Location'].value_counts()
train['Location'].describe()
# 원핫 인코딩

location = pd.get_dummies(train['Location'], prefix='Location')
location.head()
train = pd.concat([train,location], axis=1)
train.head()
f, ax = plt.subplots(figsize=(15,8))
sns.distplot(train['Price'])
plt.xlim([0,160])
var = "Name"
plt.figure(figsize=(20, 10))
sns.catplot(x=var, kind="count", palette="ch:.25", height=8, aspect=2, data=train);
plt.xticks(rotation=90);
#sns.countplot(data=train, x="Fuel_Type", hue="Kilometers_Driven")

var = "Location"
plt.figure(figsize=(20, 10))
sns.catplot(x=var, kind="count", palette="ch:.25", height=8, aspect=2, data=train);
plt.xticks(rotation=90);
plt.figure(figsize=(20,20))
sns.heatmap(train.corr(),annot=True,cmap='coolwarm')

plt.show()
train.corr()
train.columns
X = train.iloc[:,3:]
X.drop(["Kilometers_Driven"],axis=1,inplace=True)
X.drop(["New_Price"],axis=1,inplace=True)
X.head()
X.shape
y = train.loc[:,['Kilometers_Driven']]
y.head()
y.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=31)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
model_1 = DecisionTreeRegressor(max_depth=7)
model_2 = RandomForestRegressor(n_estimators=3000, max_depth=7, max_features=0.7, random_state=31, n_jobs=-1)
model_3 = LinearRegression()
model_4 = GradientBoostingRegressor()
import numpy as np
from sklearn import metrics
from sklearn.metrics import make_scorer

def rmse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)
    
    difference = predict - actual
    difference = np.square(difference)
    
    mean_difference = difference.mean()
    
    return score

rmse_scorer = make_scorer(rmse)
model_1.fit(X_train, y_train)
prediction = model_1.predict(X_test)

print("Accuracy on Traing set: ",model_1.score(X_train,y_train))
print("Accuracy on Testing set: ",model_1.score(X_test,y_test))
np.sqrt(metrics.mean_squared_error(y_test, prediction))
model_2.fit(X_train, y_train)
prediction = model_2.predict(X_test)

print("Accuracy on Traing set: ",model_2.score(X_train,y_train))
print("Accuracy on Testing set: ",model_2.score(X_test,y_test))
np.sqrt(metrics.mean_squared_error(y_test, prediction))
model_3.fit(X_train, y_train)
prediction = model_3.predict(X_test)

print("Accuracy on Traing set: ",model_3.score(X_train,y_train))
print("Accuracy on Testing set: ",model_3.score(X_test,y_test))
np.sqrt(metrics.mean_squared_error(y_test, prediction))
model_4.fit(X_train, y_train)
prediction = model_4.predict(X_test)

print("Accuracy on Traing set: ",model_4.score(X_train,y_train))
print("Accuracy on Testing set: ",model_4.score(X_test,y_test))
np.sqrt(metrics.mean_squared_error(y_test, prediction))
model = DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=7, min_samples_split=2, min_samples_leaf=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

print("Accuracy on Traing set: ",model.score(X_train,y_train))
print("Accuracy on Testing set: ",model.score(X_test,y_test))
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, prediction))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, prediction))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print('R Squared Error          : ', metrics.r2_score(y_test, prediction))
#import numpy as np
#from sklearn import metrics
#from sklearn.metrics import make_scorer
#
#def rmse(predict, actual):
#    predict = np.array(predict)
#    actual = np.array(actual)
#    
#    difference = predict - actual
#    difference = np.square(difference)
#    
#    mean_difference = difference.mean()
#    
#    return score
#
#rmse_scorer = make_scorer(rmse)
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
#
#max_depth_list = [3, 6, 9, 12, 15, 18, 21]
#max_features_list = [0.1, 0.3, 0.5, 0.7, 0.9]
#
#hyperparameters_list = []
#for max_depth in max_depth_list:
#    for max_features in max_features_list:
#        model = DecisionTreeRegressor(max_depth = max_depth, 
#                                      max_features = max_features, 
#                                      random_state=31)
#        
#        score = cross_val_score(model, X_train, y_train, cv=5,
#                                scoring=rmse_scorer).mean()
#        
#        hyperparameters_list.append({
#            'score':score,
#            'max_depth':max_depth,
#            'max_feature':max_features
#        })
#        
#        print('현재 Score = {0:5f}'.format(score))
#        
#hyperparameters_list
#result = pd.DataFrame.from_dict(hyperparameters_list)
#result = result.sort_values(by="score")
#result.head()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train)
y_train_scale = sc.fit_transform(y_train)
X_test_scale = sc.fit_transform(X_test)
y_test_scale = sc.fit_transform(y_test)
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
act = 'relu'
opt = 'Adam'
los = 'mean_squared_error'

model5 = Sequential()
model5.add(Dense(128, activation = act))
model5.add(Dense(128, activation = act))
model5.add(Dense(128, activation = act))
model5.add(Dense(1, activation = act))
model5.compile(optimizer = opt, loss = los, metrics = ['mse'])


#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
batch_size = 20
epoch = 10
history = model5.fit(X_train_scale, y_train_scale, epochs = epoch, batch_size = batch_size, verbose = 1, validation_data=(X_test_scale, y_test_scale))
score = model5.evaluate(X_test_scale, y_test_scale, batch_size=128)
print('\nAnd the Score is ', score[1] * 100, '%')
