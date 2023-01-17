import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



%matplotlib inline
df = pd.read_csv("/kaggle/input/used-cars-price-prediction/train-data.csv")
df.head()
df.rename(columns={'Unnamed: 0':'ID'},inplace=True)

df.head()
df['Engine']=df['Engine'].str.replace('CC','')

df['Mileage']=df['Mileage'].str.replace('kmpl','')

df['Mileage']=df['Mileage'].str.replace('km/kg','')

df['Power']=df['Power'].str.replace('bhp','')

df.head()
df.describe(include='all')
df.isnull().sum()
(df.isnull().sum() / len(df)) * 100

df.groupby('Seats')['ID'].nunique()
df['Seats'].mode()
df["Seats"].fillna(value = 5.0, inplace=True)

df.Seats[df.Seats == 0.0] = 5.0

df['Seats'].isna().sum()
df.groupby('Mileage')['ID'].nunique()
df.Mileage[df.Mileage == '0.0'] = np.nan

df['Mileage'] = df['Mileage'].astype(float)

df['Mileage'].mode()


df.Mileage.isnull().sum()
df.Mileage.describe()
df.Mileage=df.Mileage[df.Mileage>4]

sns.distplot(df.Mileage)
df['Mileage'].fillna(value = 17.0, inplace = True)

df.Mileage.isnull().sum()
df.groupby('Engine')['ID'].nunique()
df.Engine = df.Engine.astype(float)
sns.distplot(df.Engine)
df.Engine.describe()
q=df.Engine.quantile(0.99) # 99 percentile of Engine value

df.Engine=df.Engine[df.Engine<q]

sns.distplot(df.Engine)
df.Engine.mode()
df.Engine.fillna(value =1197 , inplace = True)

df.Engine.isnull().sum()
df['Power'] = df['Power'].str.split(' ').str[0]

# including nan rows there is data in this column of 'null' value

df.Power[df.Power == 'null'] = np.NaN

df['Power'].isnull().sum()
df['Power'] = df['Power'].astype(float)
sns.distplot(df.Power)
q=df.Power.quantile(0.99)

df.Power=df.Power[df.Power<q]

sns.distplot(df.Power)
df.Power.mode()
df['Power'].fillna(value = 74, inplace = True)

df.Power.isnull().sum()
df['Name'] = df['Name'].str.split(' ').str[0]

df.groupby('Name')['ID'].nunique()
df.Name[df.Name == 'ISUZU'] = 'Isuzu'
sns.pairplot(data=df,y_vars='Price',x_vars=['Kilometers_Driven','Mileage','Engine','Power'])
df['Price_log']=np.log(df.Price)

del df['Price']
del df['New_Price']

del df['Location']
df.describe(include='all')
del df['ID']
df.dtypes
df.Year=df.Year.astype(float)

df.Kilometers_Driven=df.Kilometers_Driven.astype(float)

df.dtypes
df_dummies=pd.get_dummies(df,drop_first=True)

df_dummies.head()
df_dummies.columns
col=['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',

       'Name_Audi', 'Name_BMW', 'Name_Bentley', 'Name_Chevrolet',

       'Name_Datsun', 'Name_Fiat', 'Name_Force', 'Name_Ford', 'Name_Honda',

       'Name_Hyundai', 'Name_Isuzu', 'Name_Jaguar', 'Name_Jeep',

       'Name_Lamborghini', 'Name_Land', 'Name_Mahindra', 'Name_Maruti',

       'Name_Mercedes-Benz', 'Name_Mini', 'Name_Mitsubishi', 'Name_Nissan',

       'Name_Porsche', 'Name_Renault', 'Name_Skoda', 'Name_Smart', 'Name_Tata',

       'Name_Toyota', 'Name_Volkswagen', 'Name_Volvo', 'Fuel_Type_Diesel',

       'Fuel_Type_Electric', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',

       'Transmission_Manual', 'Owner_Type_Fourth & Above', 'Owner_Type_Second',

       'Owner_Type_Third']
target=df_dummies['Price_log']

inputs=df_dummies[col]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2,random_state=365)
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(x_train,y_train)
y_hat=reg.predict(x_train)
plt.scatter(y_train,y_hat)
sns.distplot(y_train-y_hat)

plt.title('Residuals')
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary=pd.DataFrame(inputs.columns.values,columns=['Features'])

reg_summary['Weights']=reg.coef_.round(5)

reg_summary
y_hat_test=reg.predict(x_test)
plt.scatter(y_test,y_hat_test,alpha=0.4)

plt.xlabel('Targets(y_test)')

plt.ylabel('Prediction(y_hat_test)')

plt.show()