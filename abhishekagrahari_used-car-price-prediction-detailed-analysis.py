import datetime

import pandas as pd

import numpy as np

import math

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

from pandas import DataFrame
train = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')

train = train.reindex(np.random.permutation(train.index))

print("TRAIN SHAPE: ",train.shape)

train.info()

train.head()
train.isnull().sum()
train.describe()
print(train['Location'].unique())

print(train['Fuel_Type'].unique())

print(train['Transmission'].unique())

print(train['Owner_Type'].unique())
print("Shape of train data Before dropping any Row: ",train.shape)

train = train[train['Mileage'].notna()]

print("Shape of train data After dropping Rows with NULL values in Mileage: ",train.shape)

train = train[train['Engine'].notna()]

print("Shape of train data After dropping Rows with NULL values in Engine : ",train.shape)

train = train[train['Power'].notna()]

print("Shape of train data After dropping Rows with NULL values in Power  : ",train.shape)

train = train[train['Seats'].notna()]

print("Shape of train data After dropping Rows with NULL values in Seats  : ",train.shape)
train = train.reset_index(drop=True)
train.head()
for i in range(train.shape[0]):

    train.at[i, 'Brand'] = train['Name'][i].split()[0]

    train.at[i, 'Mileage'] = train['Mileage'][i].split()[0]

    train.at[i, 'Engine'] = train['Engine'][i].split()[0]

    train.at[i, 'Power'] = train['Power'][i].split()[0]

    train.at[i, 'Model']=''

    for j in np.arange(1,len(train['Name'][i].split())):

        train.at[i, 'Model']=train.at[i, 'Model'] + train['Name'][i].split()[j]

train.drop(train.columns[0], axis=1, inplace=True)

print(train['Brand'].unique())
train.loc[train.Brand=='Isuzu','Brand']='ISUZU'
train.head()
col_name='Model'

first_col=train.pop(col_name)

train.insert(0,col_name,first_col)

col_name='Brand'

first_col=train.pop(col_name)

train.insert(0,col_name,first_col)
train.head()
train['Mileage'] = train['Mileage'].astype(float)

train['Engine'] = train['Engine'].astype(float)
x = 'n'

count = 0

position = []

for i in range(train.shape[0]):

    if train['Power'][i]=='null':

        x = 'Y'

        count = count + 1

        position.append(i)

print(x)

print(count)

print(position)

train = train.drop(train.index[position])

train = train.reset_index(drop=True)
train.shape
train['Power'] = train['Power'].astype(float)

#Converting power values to float
x = 'n'

count = 0

position = []

for i in range(train.shape[0]):

    if train['Mileage'][i]==0.0:

        x = 'Y'

        count = count + 1

        position.append(i)

print(x)

print(count)

print(position)
train = train.drop(train.index[position])

train = train.reset_index(drop=True)
train.shape
train.head()
import seaborn as sns

sns.boxplot(x=train['Mileage'])
sns.boxplot(x=train['Year'])
sns.boxplot(x=train['Kilometers_Driven'])
x = 'n'

count = 0

position = []

for i in range(train.shape[0]):

    if train['Kilometers_Driven'][i]>5000000:

        x = 'Y'

        count = count + 1

        position.append(i)

print(x)

print(count)

print(position)
train = train.drop(train.index[position])

train = train.reset_index(drop=True)
sns.boxplot(x=train['Engine'])
sns.boxplot(x=train['Power'])
train['Seats'].value_counts()
train.shape
train.describe()
train.head()
train.shape
train.dtypes
train.isnull().sum()
print(train['Location'].unique())

print(train['Fuel_Type'].unique())

print(train['Transmission'].unique())

print(train['Owner_Type'].unique())

print(train['Year'].unique())
plt.figure(figsize=(20,10))

plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.2)



plt.subplot(141)

plt.title('Location',fontsize=20)

train['Location'].value_counts().plot.pie(autopct="%1.1f%%")



plt.subplot(142)

plt.title('Fuel_Type',fontsize=20)

train['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%')



plt.subplot(143)

plt.title('Transmission',fontsize=20)

train['Transmission'].value_counts().plot.pie(autopct='%1.1f%%')



plt.subplot(144)

plt.title('Owner_Type',fontsize=20)

train['Owner_Type'].value_counts().plot.pie(autopct='%1.1f%%')

plt.show()

fig = plt.figure(figsize=(20,18))

fig.subplots_adjust(hspace=0.2, wspace=0.2)

fig.add_subplot(2,2,1)

g1 = sns.countplot(x='Brand', data=train)

loc,labels = plt.xticks()

g1.set_xticklabels(labels,rotation=90)

fig.add_subplot(2,2,2)

g2 = sns.countplot(x='Fuel_Type', data=train)

loc,labels = plt.xticks()

g2.set_xticklabels(labels,rotation=0)

fig.add_subplot(2,2,3)

g3 = sns.countplot(x='Seats', data=train)

loc,labels = plt.xticks()

g3.set_xticklabels(labels,rotation=0)

fig.add_subplot(2,2,4)

g4 = sns.countplot(x='Owner_Type', data=train)

loc,labels = plt.xticks()

g4.set_xticklabels(labels,rotation=0)

plt.show()
sns.distplot(train['Price'])
sns.distplot(train['Kilometers_Driven'])
train['Price'].describe()
df_vis_1 = pd.DataFrame(train.groupby('Owner_Type')['Price'].mean())

df_vis_1.plot.bar()
df_vis_1 = pd.DataFrame(train.groupby('Fuel_Type')['Price'].mean())

df_vis_1.plot.bar()
df_vis_1 = pd.DataFrame(train.groupby('Transmission')['Price'].mean())

df_vis_1.plot.bar()
df_vis_2 = pd.DataFrame(train.groupby('Brand')['Price'].mean())

df_vis_2.plot.bar()
plt.figure(figsize=(20,8))

plt.subplot(1,2,1)

sns.distplot(train['Price'])



plt.subplot(1,2,2)

sns.boxplot(y=train['Price'])

plt.show()
fig = plt.figure(figsize=(15,15))

fig.subplots_adjust(hspace=0.2, wspace=0.2)

ax1 = fig.add_subplot(2,2,1)

plt.xlim([0, 100000])

p1 = sns.scatterplot(x="Kilometers_Driven", y="Price", data=train)

loc, labels = plt.xticks()

ax1.set_xlabel('Kilometer')



ax2 = fig.add_subplot(2,2,2)

p2 = sns.scatterplot(x="Mileage", y="Price", data=train)

loc, labels = plt.xticks()

ax2.set_xlabel('Mileage')



ax3 = fig.add_subplot(2,2,3)

p3 = sns.scatterplot(x="Engine", y="Price", data=train)

loc, labels = plt.xticks()

ax3.set_xlabel('Engine')



ax4 = fig.add_subplot(2,2,4)

p4 = sns.scatterplot(x="Power", y="Price", data=train)

loc, labels = plt.xticks()

ax4.set_xlabel('Power')



plt.show()
plt.figure(figsize=(18,18))

sns.heatmap(train.corr(),annot=True,cmap='RdYlGn')

plt.show()
train.corr() # Correlation coefficients .
fig = plt.figure(figsize=(18,6))

ax1 = fig.add_subplot(1,3,1)

sns.boxplot(x='Seats', y='Price', data=train)

ax1.set_title('Seats vs Price')



ax2 = fig.add_subplot(1,3,2)

sns.boxplot(x='Transmission', y='Price', data=train)

ax2.set_title('Transmission vs Price')



ax3 = fig.add_subplot(1,3,3)

sns.boxplot(x='Fuel_Type', y='Price', data=train)

ax3.set_title('Fuel vs Price')



plt.show()

#year vs price

plt.title("Year vs Price")

plt.xlabel("Year")

plt.ylabel("Price")

plt.scatter(train.Year,train.Price)

#fuel type vs price

plt.title("Fuel_Type vs Price")

plt.xlabel("Fuel_Type")

plt.ylabel("Price")

plt.scatter(train.Fuel_Type,train.Price)

#transmission vs price

plt.title("Transmission vs Price")

plt.xlabel("Transmission")

plt.ylabel("Price")

plt.scatter(train.Transmission,train.Price)

#owner type vs price

plt.title("Owner_Type vs Price")

plt.xlabel("Owner")

plt.ylabel("Price")

plt.scatter(train.Owner_Type,train.Price)

plt.show()
print(train['Location'].unique())

print(train['Fuel_Type'].unique())

print(train['Transmission'].unique())

print(train['Owner_Type'].unique())

print(train['Year'].unique())

print(train['Brand'].unique())
train['Brand'].value_counts()
train_df = pd.DataFrame(train.groupby('Brand')['Price'].mean())

train_df=train_df.sort_values(by=['Price'])

train_df.reset_index(inplace=True)

print(train_df)
train.replace({"First":1,"Second":2,"Third": 3,"Fourth & Above":4},inplace=True)
train.replace({'Ambassador':1,'Datsun':2, 'Chevrolet':3, 'Fiat':4, 'Tata':5, 'Maruti':6, 'Nissan':7, 'Volkswagen':8, 

                'Honda':9, 'Hyundai':10, 'Renault':11, 'Ford':12, 'Skoda':13, 'Mahindra':14, 'Force':15, 'Mitsubishi':16, 'Toyota':17, 

                'ISUZU':18, 'Jeep':19, 'Volvo':20, 'BMW':21, 'Audi':22, 'Mercedes-Benz':23, 'Mini':24, 'Jaguar':25, 'Land':26, 'Porsche':26,

                'Bentley':27, 'Lamborghini':28},inplace=True)
var = 'Fuel_Type'

Fuel_Type = train[[var]]

Fuel_Type = pd.get_dummies(Fuel_Type,drop_first=False)



var = 'Transmission'

Transmission = train[[var]]

Transmission = pd.get_dummies(Transmission,drop_first=False)

train= pd.concat([train,Transmission,Fuel_Type],axis=1)
train.drop(['Transmission','Transmission_Manual','Model','Fuel_Type_Diesel','Fuel_Type','Location'], axis=1, inplace=True)
curr_time = datetime.datetime.now()

train['Year'] = train['Year'].apply(lambda x : curr_time.year - x)
train = train[['Brand', 'Year','Kilometers_Driven', 'Fuel_Type_CNG', 'Fuel_Type_LPG', 'Fuel_Type_Petrol', 'Transmission_Automatic',

      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']]
train.head()
corrMatrix = train.corr()

plt.figure(figsize=(10,7))

sns.heatmap(corrMatrix, annot=True, linewidths=3, linecolor='black')

plt.show()
train.dtypes
X = train.loc[:,['Brand', 'Year','Kilometers_Driven', 'Fuel_Type_CNG', 'Fuel_Type_LPG', 'Fuel_Type_Petrol', 'Transmission_Automatic',

      'Owner_Type', 'Power', 'Seats']]

print(X.shape)

y = train.loc[:,['Price']]
from sklearn.ensemble import ExtraTreesRegressor

selection= ExtraTreesRegressor()

selection.fit(X,y.values.ravel())

plt.figure(figsize = (12,8))

feat_importances = pd.Series(selection.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3,random_state=25)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
import pandas as pd

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)

X_test=scaler.transform(X_test)
from sklearn.linear_model import LinearRegression

model1 = LinearRegression()

model1.fit(X_train, y_train)

y_pred1= model1.predict(X_test)

print("Accuracy on Traing set: ",model1.score(X_train,y_train)*100,'%')

print("Accuracy on Testing set: ",model1.score(X_test,y_test)*100,'%')
from sklearn import metrics

from sklearn.metrics import r2_score



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))

print("R2 score : %f" % r2_score(y_test,y_pred1))
test_values=list(list(model1.coef_)[0])

test_keys=list(X.columns)

res={}

for key in test_keys:

    for value in test_values:

        res[key]=value

        test_values.remove(value)

        break

print(res)
print(model1.intercept_)
plt.scatter(y_pred1,y_test)

plt.xlabel('Predicted Price of the Car')

plt.ylabel('Actual Price of the Car')
from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor()

model2.fit(X_train, y_train.values.ravel())

y_pred2= model2.predict(X_test)

print("Accuracy on Traing set: ",model2.score(X_train,y_train))

print("Accuracy on Testing set: ",model2.score(X_test,y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))

print("R2 score : %f" % r2_score(y_test,y_pred2))
plt.scatter(y_pred2,y_test,)

plt.xlabel('Predicted Price of the Car')

plt.ylabel('Actual Price of the Car')
from xgboost import XGBRegressor



model3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model3.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)

y_pred3 = model3.predict(X_test)

print("Accuracy on Traing set: ",model3.score(X_train,y_train)*100,'%')

print("Accuracy on Testing set: ",model3.score(X_test,y_test)*100,'%')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred3))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))

print("R2 score : %f" % r2_score(y_test,y_pred3))
plt.scatter(y_pred3,y_test,)

plt.xlabel('Predicted Price of the Car')

plt.ylabel('Actual Price of the Car')
# find optimal alpha with grid search

from sklearn.linear_model import Ridge

from sklearn import svm, datasets

from sklearn.model_selection import GridSearchCV

model4 = Ridge(alpha=1).fit(X_train, y_train)

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid = dict(alpha=alpha)

model4 = GridSearchCV(estimator=model4, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)

grid_result = model4.fit(X_train, y_train)

print('Best Params: ', grid_result.best_params_)
model4 = Ridge(alpha=10.0)

model4.fit(X_train, y_train)

y_pred4 = model4.predict(X_test)

print("Accuracy on Traing set: ",model4.score(X_train,y_train)*100,'%')

print("Accuracy on Testing set: ",model4.score(X_test,y_test)*100,'%')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred4))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred4))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred4)))

print("R2 score : %f" % r2_score(y_test,y_pred4))
test_values=list(list(model4.coef_)[0])

test_keys=list(X.columns)

res={}

for key in test_keys:

    for value in test_values:

        res[key]=value

        test_values.remove(value)

        break

print(res)
plt.scatter(y_pred4,y_test,)

plt.xlabel('Predicted Price of the Car')

plt.ylabel('Actual Price of the Car')
from sklearn.linear_model import Lasso

model5 = Lasso(alpha=1).fit(X_train, y_train)

alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

param_grid = dict(alpha=alpha)



model5 = GridSearchCV(estimator=model5, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)

grid_result = model5.fit(X_train, y_train)

print('Best Params: ', grid_result.best_params_)
model5 = Lasso(alpha=0.01)

model5.fit(X_train, y_train)

y_pred5 = model5.predict(X_test)

print("Accuracy on Traing set: ",model5.score(X_train,y_train)*100,'%')

print("Accuracy on Testing set: ",model5.score(X_test,y_test)*100,'%')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred5))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred5))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred5)))

print("R2 score : %f" % r2_score(y_test,y_pred5))
#list(list(model5.coef_)[0])



test_values=list(model5.coef_)

test_keys=list(X.columns)

res={}

for key in test_keys:

    for value in test_values:

        res[key]=value

        test_values.remove(value)

        break

print(res)
plt.scatter(y_pred5,y_test,)

plt.xlabel('Predicted Price of the Car')

plt.ylabel('Actual Price of the Car')