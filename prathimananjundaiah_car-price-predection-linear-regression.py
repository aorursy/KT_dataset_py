import warnings
warnings.filterwarnings('ignore')
#import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# reading the csv file and set index to carid
df=pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv").set_index("car_ID")
#display rowsa and columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
df.head()
# to check the shape of the dataframe
df.shape
# to check the information of the data frame
df.info()
#to check the statistics of the data frame
df.describe()
# to check the columns of the data frame
df.columns
#to convert int to object 
df["symboling"]=df["symboling"].map({-3:"safe",-2:"safe",-1:"safe",0:"moderate",1:"moderate",2:"risk",3:"risk"})
df.info()
# to split carname to company name
df["CompanyName"]=df["CarName"].str.split(" ").str[0]
# to count the company names
df["CompanyName"].value_counts()
# replacing to proper company  spelling names
df["CompanyName"].replace({'maxda':'mazda','vw':'volkswagen','porcshce':'porsche','Nissan':'nissan','vokswagen':'volkswagen',
                             'toyouta':'toyota','alfa-romero':'alfa-romeo'},inplace=True)
#scatter plot for numerical variable
col=("wheelbase","carlength","carwidth","carheight","curbweight","enginesize","fuelsystem","boreratio","stroke","compressionratio","horsepower","peakrpm","citympg","highwaympg")
plt.figure(figsize=(20,15))
for i in range(0,len(col)):
    plt.subplot(4,4,i+1)
    sns.scatterplot(x=col[i],
            y="price",data=df)
plt.show()
# categoricalvalue by using box plot
plt.figure(figsize=(20,15))
col=("symboling","fueltype","aspiration","doornumber","carbody","drivewheel","enginelocation","enginetype","cylindernumber","fuelsystem")
for i in range(0,len(col)):
    plt.subplot(4,4,(i+1))
    sns.boxplot(x=col[i],y="price",data=df)
plt.show()
plt.figure(figsize=(25,15))
sns.boxplot(x="CompanyName",y="price",data=df)
plt.show()
# convert categorical values to numerical values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["fueltype"]=le.fit_transform(df['fueltype'])
df["aspiration"]=le.fit_transform(df['aspiration'])
df["doornumber"]=le.fit_transform(df['doornumber'])
df["enginelocation"]=le.fit_transform(df['enginelocation'])
df.head()
df.drop(["CarName",'CompanyName'],axis=1,inplace=True)
df.head()
# converting categorical variables to dummy variables 
df = pd.get_dummies(df)
df.head()
# after dummy variable convertion from object variables to integer variable
df.info()
from sklearn.model_selection import train_test_split
#creation of train and test data set  as 70:30
df_train,df_test=train_test_split(df,train_size=0.7,test_size=.3,random_state=100)
print(df_train.shape)
print(df_test.shape)
# checking the train head data set
df_train.head()

# checking statistical train datframe
df_train.describe()

#scaling by using min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
# Some variable are out of scale
li=["wheelbase","carlength",'carwidth','carheight',"curbweight",'enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']
df_train[li]=scaler.fit_transform(df_train[li])
df_train.head()
# to cjeck the statstical dataa of the train data
df_train.describe()
# to find co-relation on train set
plt.figure(figsize=(40,40))
sns.heatmap(df_train.corr(),cmap='YlGnBu')
#finding the corelation with respect to price
cor=df_train.corr().iloc[[17]]
cor
y_train=df_train.pop("price")
X_train=df_train
#import the sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# creating object for linear regression
lm=LinearRegression()
# fitting data to X and y train
lm.fit(X_train,y_train)
#selecting the top 15 features
rfe=RFE(lm,15)
rfe=rfe.fit(X_train,y_train)
#listing the raking
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#listing the top 15 features
col=X_train.columns[rfe.support_]
col
import statsmodels.api as sm
#Creating X_train_rfe which will contain only the top 15 selected columns from the X_train dataset.
X_train_rfe=X_train[col]
# training the model
X_train_rfe=sm.add_constant(X_train_rfe)
#Applying the linearRegression model on the X_train_rfe and fitting the training dataset.
lr_1=sm.OLS(y_train,X_train_rfe).fit()
lr_1.summary()
X_train_new = X_train_rfe.drop(columns=['highwaympg'])
X_train_lm = sm.add_constant(X_train_new)

lr_2 = sm.OLS(y_train, X_train_lm).fit()
lr_2.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_train_new = X_train_new.drop(columns=['const'])
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(columns=['enginetype_rotor'])

X_train_lm = sm.add_constant(X_train_new)

lr_3 = sm.OLS(y_train, X_train_lm).fit()
lr_3.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(columns=['curbweight'])

X_train_lm = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train, X_train_lm).fit()
lr_4.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(columns=['carwidth'])

X_train_lm = sm.add_constant(X_train_new)

lr_5 = sm.OLS(y_train, X_train_lm).fit()
lr_5.summary()
X_train_new = X_train_new.drop(columns=['carbody_convertible'])

X_train_lm = sm.add_constant(X_train_new)

lr_6 = sm.OLS(y_train, X_train_lm).fit()
lr_6.summary()
X_train_new = X_train_new.drop(columns=['enginelocation'])

X_train_lm = sm.add_constant(X_train_new)

lr_7 = sm.OLS(y_train, X_train_lm).fit()
lr_7.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(columns=['stroke'])

X_train_lm = sm.add_constant(X_train_new)

lr_8 = sm.OLS(y_train, X_train_lm).fit()
lr_8.summary()
X_train_new = X_train_new.drop(columns=['boreratio'])

X_train_lm = sm.add_constant(X_train_new)

lr_9 = sm.OLS(y_train, X_train_lm).fit()
lr_9.summary()
X_train_new = X_train_new.drop(columns=['cylindernumber_three'])

X_train_lm = sm.add_constant(X_train_new)

lr_10 = sm.OLS(y_train, X_train_lm).fit()
lr_10.summary()
X_train_new = X_train_new.drop(columns=['enginetype_ohc'])

X_train_lm = sm.add_constant(X_train_new)

lr_11 = sm.OLS(y_train, X_train_lm).fit()
lr_11.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new = X_train_new.drop(columns=['horsepower'])

X_train_lm = sm.add_constant(X_train_new)

lr_12 = sm.OLS(y_train, X_train_lm).fit()
lr_12.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_lm.shape
y_train_price=lr_12.predict(X_train_lm)
# Plotting histogram of the error terms
fig = plt.figure(figsize=(5,5))
sns.distplot((y_train - y_train_price))
fig.suptitle('Error Terms')
plt.xlabel('Errors')
df_test.describe()
#creating a list which will contain all the variables which are out of scale.
li = ['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke',
     'compressionratio','horsepower','peakrpm','citympg','highwaympg','price']

#performing fit_transform() on the columns present in the above list.
df_test[li] = scaler.transform(df_test[li])
df_test.head()
#creating X and ytest
X_test=df_test
y_test=df_test.pop("price")
# making predection on lr_12 model
X_test_new=X_test[X_train_new.columns]
# adding constant
X_test_new=sm.add_constant(X_test_new)
# making predection
y_pred=lr_12.predict(X_test_new)
fig = plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)