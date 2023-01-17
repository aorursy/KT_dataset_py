import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
pd.pandas.set_option('display.max_columns', None)
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor
df=pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv")
df.head()
df=df.drop(["car_ID"],axis=1)
#Data Shape
df.shape
df.columns
df['CarName'].unique()
df.isnull().sum()
#info of dataset
df.info()
CarCompany = df['CarName'].apply(lambda x : x.split(' ')[0])
df.insert(3,"CarCompany",CarCompany)
df=df.drop(["CarName"],axis=1)
df.head()
print(df["drivewheel"].unique())
print(df["fuelsystem"].unique())
print(df["enginetype"].unique())
print(df["carbody"].unique())
CarCompany.unique()
def replace_name(a,b):
    df.CarCompany.replace(a,b,inplace=True)

replace_name("maxda","mazda")
replace_name("nissan","Nissan")
replace_name("porcshce","porsche")
replace_name("vokswagen","volkswagen")
replace_name("vw","volkswagen")
replace_name("toyouta","toyota")



print(df["CarCompany"].unique())
df["cylindernumber"].value_counts()
def convert_feature(x):
    return x.map({"two":2,"three":3,"four":4,"five":5,"six":6,"eight":8,"twelve":12})
                  
df["cylindernumber"]=df[["cylindernumber"]].apply(convert_feature)
df["doornumber"].value_counts()
def number(x):
    return x.map({"two":2,"four":4})
df["doornumber"]=df[["doornumber"]].apply(number)
# Data Description
df.describe()
df_numeric = df.select_dtypes(include =['int64','float64'])
df_numeric.head()
df_numeric.shape
df_categorical=df.select_dtypes(include=["object"])
df_categorical.head()
plt.figure(figsize=(30,35))
sns.pairplot(df_numeric)
plt.show()

plt.figure(figsize=(30, 6))

plt.subplot(1,3,1)
plt1 = df.CarCompany.value_counts().plot(kind='bar')
plt.title('Manufacturer')
plt1.set(xlabel = 'Manufacturer', ylabel='Frequency of company')

plt.subplot(1,3,2)
plt1 = df.fueltype.value_counts().plot(kind='bar')
plt.title('Fuel type')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')

plt.subplot(1,3,3)
plt1 = df.carbody.value_counts().plot(kind='bar')
plt.title('Car Type ')
plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')

plt.show()
plt.figure(figsize=(25, 6))

df1 = pd.DataFrame(df.groupby(['CarCompany'])['price'].mean().sort_values(ascending = False))
df1.plot.bar()
plt.title('Car Company vs Price')
plt.figure(figsize=(25,6))

plt.subplot(1,2,1)
plt.title('Engine Type Histogram')
sns.countplot(df.enginetype, palette=("Blues_d"))
plt.show()
df1 = pd.DataFrame(df.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))
df1.plot.bar(figsize=(8,6))
plt.title('Engine Type vs price')
plt.show()
plt.figure(figsize=(30,10))

plt.subplot(1,4,1)
plt.title('Aspiration')
sns.countplot(x="aspiration", data=df)

plt.subplot(1,4,2)
plt.title('Number of Cylinders')
sns.countplot(x="cylindernumber", data=df)

plt.subplot(1,4,3)
plt.title('Fuel System')
sns.countplot(x="fuelsystem", data=df)

plt.subplot(1,4,4)
plt.title('Drive Wheel')
sns.countplot(x="drivewheel", data=df)

plt.show()
plt.figure(figsize=(30,10))

plt.subplot(1,3,1)
plt.title('Symboling')
sns.countplot(x="symboling", data=df)

plt.show()
plt.figure(figsize=(20,5))


plt.subplot(1,3,1)
plt.title('Fuel Type')
sns.boxplot(x="fueltype", y="price", data=df)

plt.subplot(1,3,2)
plt.title('Aspiration')
sns.boxplot(x="aspiration", y="price", data=df)

plt.subplot(1,3,3)
plt.title('doornumber')
sns.boxplot(x="doornumber", y="price", data=df)

plt.show()
plt.figure(figsize=(20,5))


plt.subplot(1,3,1)
plt.title('Carbody')
sns.boxplot(x="carbody", y="price", data=df)

plt.subplot(1,3,2)
plt.title('Drivewheel')
sns.boxplot(x="drivewheel", y="price", data=df)

plt.subplot(1,3,3)
plt.title('Engine Location')
sns.boxplot(x="enginelocation", y="price", data=df)

plt.show()
plt.figure(figsize = (20,12))
sns.boxplot(x = 'CarCompany', y = 'price', data = df)

plt.figure(figsize=(20,8))


plt.subplot(1,3,1)
plt.title('Type of Engine')
sns.boxplot(x="enginetype", y="price", data=df)

plt.subplot(1,3,2)
plt.title('Cylinder number')
sns.boxplot(x="cylindernumber", y="price", data=df)

plt.subplot(1,3,3)
plt.title('Fuel System')
sns.boxplot(x="fuelsystem", y="price", data=df)

plt.show()
plt.figure(figsize=(20,12))

plt.subplot(1,3,1)
plt.title('Symboling')
sns.boxplot(x="symboling", y="price", data=df)
###For Numerical Features

def scatter(x,fig):
    plt.subplot(7,2,fig)
    plt.scatter(df[x],df["price"])
    plt.title(x+' vs Price')
    plt.xlabel(x)
    plt.ylabel("Price")
    
plt.figure(figsize=(15,30))

scatter('symboling',1)
scatter( 'wheelbase',2)
scatter('carlength',3)
scatter('carwidth',4)
scatter('carheight',5)
scatter('curbweight',6)
scatter('enginesize',7)
scatter('boreratio',8)
scatter('stroke',9)
scatter('compressionratio',10)
scatter('horsepower',11)
scatter('peakrpm',12)
scatter('citympg',13)
scatter('highwaympg',14)

plt.tight_layout()



plt.figure(figsize = (20,20))
sns.heatmap(df.corr(), annot = True ,cmap = 'YlGnBu')
plt.show()
#creating dummies
cars_dummies = pd.get_dummies(df_categorical, drop_first = True)
cars_dummies.shape
df_car = pd.concat([df, cars_dummies], axis =1)
df_car=df_car.drop(["CarCompany","fueltype","aspiration","carbody","drivewheel","enginelocation","enginetype","fuelsystem"],axis=1)

df_car.info()
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_car, train_size = 0.7, test_size = 0.3, random_state = 100)
df_train.shape
df_test.shape
scaler = StandardScaler()

col_list=["symboling","doornumber","wheelbase","carlength","carwidth","carheight","curbweight","cylindernumber","enginesize",
          "boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg","price"]
df_train[col_list] = scaler.fit_transform(df_train[col_list])
df_train.describe()
y_train = df_train.pop('price')
X_train = df_train
lr = LinearRegression()
lr.fit(X_train,y_train)

# Subsetting training data for 15 selected columns
rfe = RFE(lr,15)
rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
cols = X_train.columns[rfe.support_]
cols

X1 = X_train[cols]
X1_sm = sm.add_constant(X1)

lr_1 = sm.OLS(y_train,X1_sm).fit()
print(lr_1.summary())
#VIF
vif = pd.DataFrame()
vif['Features'] = X1.columns
vif['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
lr2 = LinearRegression()

rfe2 = RFE(lr2,10)
rfe2.fit(X_train,y_train)
lr2 = LinearRegression()

rfe2 = RFE(lr2,10)
rfe2.fit(X_train,y_train)
supported_cols = X_train.columns[rfe2.support_]
supported_cols

X2 = X_train[supported_cols]
X2_sm = sm.add_constant(X2)

model_2 = sm.OLS(y_train,X2_sm).fit()
print(model_2.summary())

#VIF
vif = pd.DataFrame()
vif['Features'] = X2.columns
vif['VIF'] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X3 = X2.drop(['CarCompany_peugeot'], axis =1)
X3_sm = sm.add_constant(X3)

Model_3 = sm.OLS(y_train,X3_sm).fit()
print(Model_3.summary())
#VIF
vif = pd.DataFrame()
vif['Features'] = X3.columns
vif['VIF'] = [variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
X4 = X3.drop(['enginelocation_rear'], axis =1)
X4_sm = sm.add_constant(X4)

Model_4 = sm.OLS(y_train,X4_sm).fit()
print(Model_4.summary())
#VIF
vif = pd.DataFrame()
vif['Features'] = X4.columns
vif['VIF'] = [variance_inflation_factor(X4.values, i) for i in range(X4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif
y_train_pred = Model_4.predict(X4_sm)
y_train_pred.head()
Residual = y_train- y_train_pred
sns.distplot(Residual, bins =15)
df_test[col_list] = scaler.transform(df_test[col_list])
y_test = df_test.pop('price')
X_test = df_test
final_cols = X4.columns
X_test_model4= X_test[final_cols]
X_test_model4
X_test_sm = sm.add_constant(X_test_model4)
X_test_sm
y_test_pred = Model_4.predict(X_test_sm)
y_test_pred.head()

c = [i for i in range(1,63,1)]
plt.plot(c, y_test,color = 'Blue')
plt.plot(c, y_test_pred,color = 'red')
plt.xlabel("VOC")
plt.ylabel("VOC")


plt.scatter(y_test, y_test_pred)
plt.xlabel('y_test')
plt.ylabel('y_test_pred')
r_squ = r2_score(y_test,y_test_pred)
r_squ