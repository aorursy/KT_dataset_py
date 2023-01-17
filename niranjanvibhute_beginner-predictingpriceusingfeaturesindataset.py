#####################################################PROJECT####################################################################

# Importing packages 

import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import seaborn as sns

import matplotlib.pyplot as plt
                      #Data preparation#

#Reading of data from package-'pandas

#I give name to my data set as 'car' to further analysis

car=pd.read_csv("../input/bmw-pricing-challenge/bmw_pricing_challenge.csv")
# Below code shows row ,columns 

car.shape
#The below code gives the information of dataset

car.info()
#the below code gives the first 5 record of the dataset

car.head()
# The below code describe the dataset

car.describe(include='all')
#Linearity

#Relationship between the independent and dependent variables through the scatter plot

for c in car.columns[:-1]:

    plt.title("{} vs. \nprice".format(c))

    plt.scatter(x=car[c],y=car['price'],color='blue',edgecolor='k')

    plt.grid(True)

    plt.xlabel(c,fontsize=14)

    plt.ylabel('price')

    plt.show()
# Visualization or data comparing(Model_Key Vs Engine_power)

model_key = car['model_key'].unique()

engine_power = []

for each in model_key:

    x = car[car['model_key']==each]

    engine_power.append(sum(x['engine_power']/len(x)))



plt.figure(figsize=(10,20))

sns.barplot(x=engine_power,y=model_key)

plt.xlabel("EnginePower")

plt.ylabel("Model_key")

plt.title("Engine Power according to Model", color="Red")

plt.show()
# Visualization or data comparing(Model_key Vs Price)

model_key = car['model_key'].unique()

price = []

for each in model_key:

    x = car[car['model_key']==each]

    price.append(sum(x['price']/len(x)))



plt.figure(figsize=(10,20))

sns.barplot(x=price,y=model_key)

plt.xlabel("price")

plt.ylabel("Model_key")

plt.title("Price according to Model", color="Red")

plt.show()
# Visualization or data comparing(Paint_color vs price)

paint_color = car['paint_color'].unique()

price = []

for each in paint_color:

    x = car[car['paint_color']==each]

    price.append(sum(x['price']/len(x)))



plt.figure(figsize=(15,10))

sns.barplot(x=paint_color,y=price)

plt.xlabel("Paint_Color")

plt.ylabel("Price")

plt.title("Price according to colors", color="Red")

plt.show()
# BarGraph between fuel and Price

fuel = car['fuel'].unique()

price = []

for each in fuel:

    x = car[car['fuel']==each]

    price.append(sum(x['price']/len(x)))



plt.figure(figsize=(15,7))

sns.barplot(x=fuel,y=price)

plt.xlabel("Fuel")

plt.ylabel("Price")

plt.title("Price according to fuel", color="Red")

plt.show()
# BarGraph b/w car_type and price

car_type = car['car_type'].unique()

price = []

for each in car_type:

    x = car[car['car_type']==each]

    price.append(sum(x['price']/len(x)))



plt.figure(figsize=(15,7))

sns.barplot(x=car_type,y=price)

plt.xlabel("Car_type")

plt.ylabel("Price")

plt.title("Price according to Car_type", color="Red")

plt.show()
#pie chart related mileage

mileage = []

labels = ['below10K','10K-50K','50K-100K','100K-150K','150K-200K','200K+']

colors = ['white','brown','orange','yellow','green','pink']

explode = [0,0,0,0,0,0]

for each in range(1,21):

    each = each*10000

    if(each==10000):

        x = car[car['mileage']<10000]

        mileage.append(len(x))

    elif(each==50000):

        x = car[(car['mileage']>=10000) & (car['mileage']<50000)]

        mileage.append(len(x))

    elif(each==100000):

        x = car[(car['mileage']>=50000) & (car['mileage']<100000)]

        mileage.append(len(x))

    elif(each==150000):

        x = car[(car['mileage']>=100000) & (car['mileage']<150000)]

        mileage.append(len(x))

    elif(each==200000):

        x = car[(car['mileage']>=150000) & (car['mileage']<200000)]

        mileage.append(len(x))

        x = car[car['mileage']>=200000]

        mileage.append(len(x))

plt.figure(figsize=(7,5))

plt.pie(mileage,explode=explode, labels=labels,colors=('white','brown','orange','yellow','green','pink'),autopct='%1.2f%%')

plt.title('Mileage',color="red")

plt.show()
# fuel_type Vs Car count

plt.figure(figsize=(10,5))

sns.countplot(car['fuel'])

plt.title('Fuel',color="red")

plt.xlabel("Fuel Type",color='brown')

plt.ylabel("Car Count",color='brown')

plt.show()
#Converting the data set into numerical for further analysis

#label encoder of fuel table

from sklearn.preprocessing import LabelEncoder

ENC = LabelEncoder()

car["Fuel"] = ENC.fit_transform(car['fuel'])

car[['fuel','Fuel']].head(5)



car.drop('fuel',axis=1,inplace=True)



#Lable encoder of car type

b = LabelEncoder()

car["Car_type"] = b.fit_transform(car['car_type'])

car[['car_type','Car_type']].head(5)



car.drop('car_type',axis=1,inplace=True)



#Lable encoder of car model_key

b = LabelEncoder()

car["Model_key"] = b.fit_transform(car['model_key'])

car[['model_key','Model_key']].head(5)



car.drop('model_key',axis=1,inplace=True)



#Lable encoder of paint_color

b = LabelEncoder()

car["Paint_color"] = b.fit_transform(car['paint_color'])

car[['Paint_color','paint_color']].head(5)



car.drop('paint_color',axis=1,inplace=True)



#Lable encoder of feature1

b = LabelEncoder()

car["Feature1"] = b.fit_transform(car['feature_1'])

car[['Feature1','feature_1']].head(5)



car.drop('feature_1',axis=1,inplace=True)



#Lable encoder of feature1

b = LabelEncoder()

car["Feature2"] = b.fit_transform(car['feature_2'])

car[['Feature2','feature_2']].head(5)



car.drop('feature_2',axis=1,inplace=True)



#Lable encoder of feature1

b = LabelEncoder()

car["Feature3"] = b.fit_transform(car['feature_3'])

car[['Feature3','feature_3']].head(5)



car.drop('feature_3',axis=1,inplace=True)



#Lable encoder of feature4

b = LabelEncoder()

car["Feature4"] = b.fit_transform(car['feature_4'])

car[['Feature4','feature_4']].head(5)



car.drop('feature_4',axis=1,inplace=True)



#Lable encoder of feature5

b = LabelEncoder()

car["Feature5"] = b.fit_transform(car['feature_5'])

car[['Feature5','feature_5']].head(5)



car.drop('feature_5',axis=1,inplace=True)



#Lable encoder of feature6

b = LabelEncoder()

car["Feature6"] = b.fit_transform(car['feature_6'])

car[['Feature6','feature_6']].head(5)



car.drop('feature_6',axis=1,inplace=True)



#Lable encoder of feature7

b = LabelEncoder()

car["Feature7"] = b.fit_transform(car['feature_7'])

car[['Feature7','feature_7']].head(5)



car.drop('feature_7',axis=1,inplace=True)



#Lable encoder of feature8

b = LabelEncoder()

car["Feature8"] = b.fit_transform(car['feature_8'])

car[['Feature8','feature_8']].head(5)



car.drop('feature_8',axis=1,inplace=True)
#Conversion of object(string) to date type

car['Sold_at'] = pd.to_datetime(car['sold_at'], format= '%Y/%m/%d')
#Conversion of object(string) to date type

car['Registration_date'] = pd.to_datetime(car['registration_date'], format= '%Y/%m/%d')
# create one more column AgeOfCar

import datetime

from dateutil.relativedelta import relativedelta

from datetime import date

 

 

car['AgeOfCar'] = car['Sold_at'].sub(car['Registration_date'], axis=0)

car.head()
#max/min of price/AgeOfCar



print(max(car['price']))

print(min(car['price']))

print(max(car['AgeOfCar']))

print(min(car['AgeOfCar']))
#checking relation between new coloumn AgeOfCar vs price

x=car['price']

y=car['AgeOfCar']

plt.plot(x, y,  'o',color='blue')
#conversion of date to numerical for further analysis

#Lable encoder of sold_at

b = LabelEncoder()

car["SoldAt"] = b.fit_transform(car['Sold_at'])

car[['SoldAt','Sold_at']].head(5)



car.drop('Sold_at',axis=1,inplace=True)


#Lable encoder of Registration_date

b = LabelEncoder()

car["RegistrationDate"] = b.fit_transform(car['Registration_date'])

car[['RegistrationDate','Registration_date']].head(5)



car.drop('Registration_date',axis=1,inplace=True)


#Lable encoder of AgeOfCar

b = LabelEncoder()

car["AgeofCar"] = b.fit_transform(car['AgeOfCar'])

car[['AgeofCar','AgeOfCar']].head(5)



car.drop('AgeOfCar',axis=1,inplace=True)
car.drop(['registration_date','sold_at'],axis=1,inplace=True)

car.head()


#to get corelation matics with relevent dataset

print(car.corr())

#by graph correaltional coefficient

sns.heatmap(car.corr())
#for analysis purpose 

car1=car.drop(['maker_key'],axis=1)

X=car1.drop(['price','Fuel','Paint_color','Feature7'],axis=1)

Y=car1['price']
#splitting a data to train test  to create a model and testing on model 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=0)
# Multiple linear agression to the price with all features 

#fitting multiple linear regression

from sklearn.linear_model import LinearRegression

LReg=LinearRegression()

LReg.fit(x_train,y_train)
#predicting the test result

y_pred=LReg.predict(x_test)

print(y_pred)
#calaculating coefficeint and intercept

print(LReg.coef_)#b1,b2,b3,.......,b11

#calaculate intercept

print(LReg.intercept_)#b0
#evaluvatig the model

#calculating the R value

from sklearn.metrics import r2_score

r2_score(y_test , y_pred)
#r2 comparing to this 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

reg=LinearRegression()

reg=reg.fit(X,Y)

y_pred=reg.predict(X)

r2_score=reg.score(X,Y)

print(r2_score)


#Applying Random Forest Regressor

X = car1.drop(['price'],axis=1)

Y= car1['price']
#test_train_split

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,Y,test_size=0.20, 

                                                                          random_state=42,

                                                                          shuffle=True)
#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor

n_estimators=30

# Fit regression model

# Estimate the score on the entire dataset, with no missing values

model = RandomForestRegressor(random_state=42,max_depth=7,max_features=9,n_estimators=n_estimators)

model.fit(x_training_set, y_training_set)
# Appling model to the data and getting R^2.

from sklearn.metrics import mean_squared_error, r2_score

model_score = model.score(x_training_set,y_training_set)

# Have a look at R sq to give an idea of the fit ,

# Explained variance score: 1 is perfect prediction

print("coefficient of determination R^2 of the prediction.: ",model_score)

y_predicted = model.predict(x_test_set)



# The mean squared error

print("Mean squared error: %.2f"% mean_squared_error(y_test_set, y_predicted))

# Explained variance score: 1 is perfect prediction

print('Test Variance score: %.2f' % r2_score(y_test_set, y_predicted))
# So let's run the model against the test data



from sklearn.model_selection import cross_val_predict









fig, ax = plt.subplots()



ax.scatter(y_test_set, y_predicted, edgecolors=(0, 0, 0))



ax.plot([y_test_set.min(), y_test_set.max()], [y_test_set.min(), y_test_set.max()], 'k--', lw=4)



ax.set_xlabel('Actual')



ax.set_ylabel('Predicted')



ax.set_title("Ground Truth vs Predicted")



plt.show()