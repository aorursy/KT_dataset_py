import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder 

%matplotlib inline



dataset = pd.read_csv('../input/House_prediction.csv')

data=pd.DataFrame.copy(dataset)
le = LabelEncoder() 



dataset['city']= le.fit_transform(dataset['city']) 



dataset['furniture']= le.fit_transform(dataset['furniture']) 



dataset['animal']= le.fit_transform(dataset['animal']) 



dataset['floor']= le.fit_transform(dataset['floor']) 



dataset_scaled = preprocessing.scale(dataset)

arr=dataset.to_numpy()

dataset.describe()
dataset.describe()
plt.figure(figsize =(10,10))

plt.subplot(2, 1, 1)

ax = seabornInstance.distplot(data['rent amount (R$)'],kde =True)

plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(dataset['rooms'])
plt.figure(figsize=(15,10))

plt.tight_layout()

seabornInstance.distplot(dataset['bathroom'])
dataset.plot(x='city',y='rent amount (R$)',style='o')

plt.title('')  

plt.xlabel('')  

plt.ylabel('')  
dataset.plot(x='bathroom',y='rent amount (R$)',style='o')

plt.title('HOA vs city ')  

plt.xlabel('City')  

plt.ylabel('Hoa')  
g = seabornInstance.FacetGrid(data, row = 'city')

g = g.map(plt.hist,'rent amount (R$)')
dataset.plot(x='furniture',y='fire insurance (R$)',style='o')

plt.title('Fire insurance vs furniture')  

plt.xlabel('furniture')  

plt.ylabel('Fire insurance')  
dataset.plot(x='fire insurance (R$)',y='rent amount (R$)',style='o')

plt.title('rent vs fire insurance ')  

plt.xlabel('fire insurance')  

plt.ylabel('rent')  
def categorize(col):

    numerical,category=[],[]

    for i in col:

        if data[i].dtype ==object:

            category.append(i)

        else:

            numerical.append(i)

    print("The numerical features {}:".format(numerical))

    print("The categorical features {}:".format(category))

    return category,numerical

categorical,numerical =categorize(dataset.columns)

plt.figure(figsize =(8,8))

cor = data[numerical].corr()

seabornInstance.heatmap(cor,annot =True)
y=dataset['rent amount (R$)'].values.reshape(-1,1)



regressor = LinearRegression(fit_intercept=False)  



X=dataset[['city','area','rooms','bathroom','parking spaces','floor','animal','furniture','fire insurance (R$)','rent amount (R$)']]



print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.06, random_state=0)



X_scaled = preprocessing.scale(X_train)

regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)



df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})



df

df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
y=dataset['rent amount (R$)'].values.reshape(-1,1)



regressor = LinearRegression(fit_intercept=False)  



X=dataset[['city','area','rooms','bathroom','parking spaces','floor','animal','furniture']]



print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.06, random_state=0)



X_scaled = preprocessing.scale(X_train)

regressor.fit(X_train, y_train) 

y_pred = regressor.predict(X_test)



df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})



df
df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))