import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()
# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
car= pd.read_csv("../input/vehicle-dataset-from-cardekho/car data.csv")
# Let's see how our dataset looks like



car.head()
# Let's see how many rows and columns do we have in the dataset.



car.shape
# Let's see some summary



car.describe()
car.info()
# To check if there are any outliers



car.describe(percentiles=[0.25,0.5,0.75,0.9,0.95,0.99])
# To check if there are any missing values in the dataset



import missingno as mn

mn.matrix(car)
# It's important to know how many years old the car is.



car['car_age']= 2020-car['Year']
# It's time to drop the Year column after the needed info is derived.



car.drop('Year',axis=1,inplace=True)
car.head()
sns.pairplot(car)
sns.heatmap(car.corr(),annot=True,cmap='summer')
car.columns
sns.barplot('Seller_Type','Selling_Price',data=car,palette='twilight')
sns.barplot('Transmission','Selling_Price',data=car,palette='spring')
sns.barplot('Fuel_Type','Selling_Price',data=car,palette='summer')
sns.regplot('Selling_Price','Present_Price',data=car)
sns.regplot('Selling_Price','Kms_Driven',data=car)
sns.barplot('Owner','Selling_Price',data=car,palette='ocean')
plt.figure(figsize=(10,5))

sns.barplot('car_age','Selling_Price',data=car)
car.columns
fuel = pd.get_dummies(car['Fuel_Type'])

transmission = pd.get_dummies(car['Transmission'],drop_first=True)

seller= pd.get_dummies(car['Seller_Type'],drop_first=True)
fuel.drop('CNG',axis=1,inplace=True)
car= pd.concat([car,fuel,transmission,seller],axis=1)
car.head()
car.drop(['Fuel_Type','Seller_Type','Transmission'],axis=1,inplace=True)
#The column car name doesn't seem to add much value to our analysis and hence dropping the column



car= car.drop('Car_Name',axis=1)
car.head()
from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train, df_test = train_test_split(car, test_size = 0.3, random_state = 100)
num_vars=['Selling_Price','Present_Price','Kms_Driven']
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()
df_train[num_vars]= scaler.fit_transform(df_train[num_vars])

df_test[num_vars]= scaler.transform(df_test[num_vars])
y_train = df_train.pop('Selling_Price')

X_train = df_train
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
lm= LinearRegression()

lm.fit(X_train, y_train)



rfe= RFE(lm,10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train_rfe = X_train[col]
import statsmodels.api as sm

X_train_rfe= sm.add_constant(X_train_rfe)
model = sm.OLS(y_train,X_train_rfe).fit()

model.summary()
X_train1= X_train_rfe.drop('Petrol',axis=1)
X_train2= sm.add_constant(X_train1)

model1= sm.OLS(y_train,X_train2).fit()

model1.summary()
X_train3= X_train2.drop('Owner',axis=1)
X_train4= sm.add_constant(X_train3)

model2= sm.OLS(y_train,X_train4).fit()

model2.summary()
X_train5= X_train4.drop('Kms_Driven',axis=1)
X_train6= sm.add_constant(X_train5)

model3= sm.OLS(y_train,X_train6).fit()

model3.summary()
X_train_new= X_train6.drop('const',axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = model3.predict(X_train6)
fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  
#Dividing the test set into features and target.



y_test = df_test.pop('Selling_Price')

X_test = df_test
# Predicting the values by extracting the columns that our final model had



X_test_pred= X_test[X_train_new.columns]



X_test_pred= sm.add_constant(X_test_pred)
y_pred= model3.predict(X_test_pred)
fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16) 
df = pd.DataFrame({'Actual':y_test,"Predicted":y_pred})

df.head()
from sklearn.metrics import r2_score

R2 = r2_score(y_test,y_pred)

R2