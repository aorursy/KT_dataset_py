#Import Library ต่างๆ

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import statsmodels.api as sm



# machine learning

import sklearn.datasets as datasets

# เช็คData ที่อยู่ใน folder input

print(os.listdir("../input"))
# อ่าน CSV files

population = pd.read_csv('../input/country_population.csv')

fertility = pd.read_csv('../input/fertility_rate.csv')

life = pd.read_csv('../input/life_expectancy.csv')

population.dropna(how='any',inplace=True)

life.dropna(how='any',inplace=True)

fertility.dropna(how='any',inplace=True)
# ตัวอย่างการ Drop Colums

temp_pop = population

temp_pop.drop(columns=['Country Name','Country Code', 'Indicator Name', 'Indicator Code'],axis =1, inplace=True)
# จำนวนทั้งหมดของประชากรตชในปีนั้นๆ

pop_sum=temp_pop.sum()

pop_sum=pd.DataFrame(pop_sum).reset_index()

pop_sum.columns= ['Year','Total Population']

pop_sum.head()

#  Stack overflow method 

#การทำ Linear Plot ของประชากรทั้งโลกในแต่ละปี

plt.figure(figsize=(35,10))

plt.plot(pop_sum['Year'], pop_sum['Total Population'])

plt.title('Global Population from 1960 to 2016')

plt.xticks(np.arange(1960,2017))

plt.show()

#ลบ Columns ที่ไม่ต้องการ

temp_fert= fertility

temp_fert.head()

temp_fert.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)

new_fert =temp_fert.dropna()
#ดูHead ของ Fertility

#no empty values 

# new_fert.isnull()

new_fert.head()
# ค่าต่างๆและค่าเฉลี่ยของFert

fert_mean = new_fert.mean()

fert_mean = pd.DataFrame(fert_mean).reset_index()

fert_mean.columns=['Year', 'Fertility']

fert_mean.describe()
# fert_mean.plot()

plt.figure(figsize=(30,10))

plt.plot(fert_mean['Year'], fert_mean['Fertility'])

plt.xticks(np.arange(1960,2017))

plt.title('Fertility from 1960 to 2016')

plt.show()

df_life = life

# df_life.head()

df_life.drop(['Country Name','Country Code','Indicator Name', 'Indicator Code'],axis =1, inplace = True)

new_life = df_life.dropna()

new_life.head()

# new_life.isnull().values.any()# no empty values 

life_mean =  new_life.mean()

life_mean = pd.DataFrame(life_mean).reset_index()

life_mean.columns= ['Year', 'Life expectancy']

# life_mean.plot()

life_mean.head()
plt.figure(figsize=(35,10))

plt.plot(life_mean['Year'], life_mean['Life expectancy'])

plt.xticks(np.arange(1960,2017))

plt.title('Life Expectancy from 1960 to 2016')

plt.show()

# world_data = pd.merge(pd.merge(pop_sum, life_mean, on='Year'), fert_mean, on='Year')

first_data = pd.merge(pop_sum,life_mean, on='Year')

world_data = pd.merge(first_data, fert_mean, on='Year')

world_data.columns

plt.figure(figsize=(35,10))

plt.plot(world_data['Year'],world_data['Total Population'],world_data['Fertility'])

plt.xticks(np.arange(1960,2017))

plt.title('Fertility Rate VS Total Populations from 1960 to 2016')

plt.show()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

world_data[['Total Population', 'Life expectancy', 'Fertility']] = scaler.fit_transform(world_data[['Total Population', 'Life expectancy', 'Fertility']])

world_data.head()
# ค่าต่างๆและค่าเฉลี่ยของFert

new_fert.dropna(how='any',inplace=True)

fert_mean = new_fert.mean()

fert_mean = pd.DataFrame(fert_mean).reset_index()

fert_mean.columns=['Year', 'Fertility']

fert_mean.head()
fert_mean['Year'] = fert_mean.Year.astype(float)
from sklearn.cross_validation import train_test_split

X = fert_mean[['Year']]

y = fert_mean[['Fertility']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20)

model = sm.OLS(y_train, X_train).fit()

predictions = model.predict(X_test)
regressor = LinearRegression()  

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
df = pd.DataFrame(y_pred, columns = ['ouput'])

df.head()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
plt.figure(figsize=(20,10))

plt.scatter(X_test, y_test, color = "white")

plt.plot(X_test, predictions, color = "pink")

plt.title("Regression Model")

plt.xlabel("Year")

plt.ylabel('Fertility Rates')

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")

model.summary()
from sklearn.ensemble import RandomForestRegressor

datardf = RandomForestRegressor().fit(X_train,y_train) # Fitting the model.

predictions = datardf.predict(X_test) # Test set is predicted.

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions)) 

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
plt.figure(figsize=(20,10))

plt.scatter(X_test, y_test, color = "white")

plt.plot(X_test, predictions, color = "pink")

plt.title("Regression Model")

plt.xlabel("Year")

plt.ylabel('Fertility Rates')

ax = plt.axes()

# Setting the background color

ax.set_facecolor("black")

model.summary()
predictions = datardf.predict([[2017]]) # Test set is predicted.

df = pd.DataFrame(predictions, columns = ['Predicted 2017'])

df
predictions = datardf.predict([[2017]]) # Test set is predicted.

predictions