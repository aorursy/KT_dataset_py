#importing basic libraries
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#importing sklearn libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#importing statsmodel api module
import statsmodels.api as sm

#importing warnings
from warnings import filterwarnings
filterwarnings('ignore')
sb.set_style("darkgrid")

pd.options.display.max_columns = 40
#Reading dataset
df=pd.read_csv("../input/car-price-prediction/CarPrice_Assignment.csv")
df.info()
df.head()
#Checking the Null Values %age
round(((df.isna().sum() / df.shape[0]) * 100),2)
#Splitting the CarName COlumn
df.CarName=df.CarName.apply(lambda x:x.split(" ")[0])
#Checking the unique values
df.CarName.unique()
df.CarName.replace({"maxda":"mazda","vokswagen":"volkswagen","vw":"volkswagen","Nissan":"nissan","porcshce":"porsche","toyouta":"toyota"},inplace=True)
df.doornumber.unique()
#mapping the door number from object to numerical
df.doornumber=df.doornumber.replace({"two":2,"four":4}).astype(np.int64)
df.drivewheel.unique()
df.cylindernumber.unique()
#mapping the object columns to numericals
df.cylindernumber=df.cylindernumber.map({"four":4,"six":6,"five":5,"three":3,"twelve":12,"two":2,"eight":8}).astype(np.int64)
df.cylindernumber.unique()
plt.figure(figsize=(20,10))

#FUELTYPE

# fueltye vs price distribution
plt.subplot(2,3,1)
plt.title("fueltype vs price ")
sb.boxplot(df.fueltype,df.price)
#count 
plt.subplot(2,3,4)
plt.title("NO. of cars with specified fueltype ")
sb.countplot(df.fueltype)

#ASPIRATION

#aspiration vs price disribution
plt.subplot(2,3,2)
plt.title("Aspiration used in the car vs price")
sb.boxplot(df.aspiration,df.price)
#count
plt.subplot(2,3,5)
plt.title("Count of specific Aspiration type")
sb.countplot(df.aspiration)

#DOORNUMBER

#doornumber vs price distribution
plt.subplot(2,3,3)
plt.title("no. of doors vs price")
sb.boxplot(df.doornumber,df.price)
#count
plt.subplot(2,3,6)
plt.title("Count of the cars with specific door number")
sb.countplot(df.doornumber)
plt.figure(figsize=(20,4))
plt.subplot(1,2,1)
plt.title("Count of type of carbody bought")
sb.countplot(df.carbody)
plt.subplot(1,2,2)
plt.title("type of carbody vs price")
sb.boxplot(df.carbody,df.price)
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.title("No. of the cars bought vs engine-location")
sb.countplot(df.enginelocation)
plt.subplot(1,2,2)
plt.title("No of cars vs Engine type")
sb.countplot(df.enginetype)
plt.figure(figsize=(7,4))
plt.title("No. of cyliner used in the cars")
sb.countplot(df.cylindernumber)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("No of cars for particular fuelsystem")
sb.countplot(df.fuelsystem)
plt.subplot(1,2,2)
plt.title("fuelsystm vs car prices")
sb.barplot(df.fuelsystem,df.price)
int_vars=df.select_dtypes(exclude="object")
int_vars.head()
nvars=int_vars.drop(labels=["car_ID","cylindernumber","price"],axis=1)
cols=list(nvars.columns)
#sb.pairplot(data=df,x_vars=cols,y_vars=df.price)
sb.pairplot(data=df,x_vars=cols[:5],y_vars="price")
sb.pairplot(data=df,x_vars=cols[6:11],y_vars="price")
sb.pairplot(data=df,x_vars=cols[12:],y_vars="price")
plt.figure(figsize=(15,7))
sb.heatmap(data=df.corr(),annot=True)
sb.distplot(df.groupby(by="CarName")["price"].mean().sort_values(ascending=False),bins=3)
def car(x):
    if x>5000 and x<=15000:
        return "Low"
    elif x>15000 and x<=25000:
        return "Medium"
    else :
        return "Hign"
df["car_cat"]=df.price.apply(car)
df.head()
#GETTING DUMMIES
dummy_vars=["fueltype","aspiration","carbody","drivewheel","enginelocation","enginetype","fuelsystem","car_cat"]
dummies=pd.get_dummies(data=df[dummy_vars],drop_first=True)
#Dropiing the columns that are of no use to us
df.drop(labels=["car_ID","CarName","fueltype","aspiration","carbody","drivewheel","enginelocation","enginetype","fuelsystem","car_cat"],axis=1,inplace=True)
final_df=pd.concat([df,dummies],axis=1)
final_df.head()
#SPLITTING THE DATASET

df_train,df_test=train_test_split(final_df,test_size=0.3,random_state=42)
#SCALING THE FEATUREs
scale=MinMaxScaler()
df_train.columns
cols=['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth',
       'carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
       'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg',
       'highwaympg', 'price', 'fueltype_gas', 'aspiration_turbo',
       'carbody_hardtop', 'carbody_hatchback', 'carbody_sedan',
       'carbody_wagon', 'drivewheel_fwd', 'drivewheel_rwd',
       'enginelocation_rear', 'enginetype_dohcv', 'enginetype_l',
       'enginetype_ohc', 'enginetype_ohcf', 'enginetype_ohcv',
       'enginetype_rotor', 'fuelsystem_2bbl', 'fuelsystem_4bbl',
       'fuelsystem_idi', 'fuelsystem_mfi', 'fuelsystem_mpfi',
       'fuelsystem_spdi', 'fuelsystem_spfi', 'car_cat_Low', 'car_cat_Medium']
df_train[cols]=scale.fit_transform(df_train[cols])
y_train=df_train.pop("price")
X_train=df_train
X_train.shape
y_train.shape
lm=LinearRegression()
lm.fit(X_train,y_train)
rfe=RFE(lm,10)
rfe=rfe.fit(X_train,y_train)
rfe.ranking_
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
final_cols=X_train.columns[rfe.support_].to_list()
#Checking the final columns 
final_cols
X_train.columns[~rfe.support_]
################################### ITERATION 1 ############################################
X_train_rfe=X_train[final_cols]
#Making the model using statsmodel (sm)
X_train_rfe=sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()
lm.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X=X_train_rfe.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
X_train_2=X_train_rfe.drop(["citympg"],axis=1)
###################################     ITERATION 2   ###########################
X_train_2=sm.add_constant(X_train_2)
lm2 = sm.OLS(y_train,X_train_2).fit()
lm2.summary()
vif = pd.DataFrame()
X=X_train_2.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
X_train_3=X_train_2.drop(["carbody_wagon"],axis=1)
##################################      TRERATION 3    #####################
X_train_3=sm.add_constant(X_train_3)
lm3 = sm.OLS(y_train,X_train_3).fit()
lm3.summary()
vif = pd.DataFrame()
X=X_train_3.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
X_train_4=X_train_3.drop(["highwaympg"],axis=1)
#######################################       ITERATION 4    ###################
X_train_4=sm.add_constant(X_train_4)
lm4 = sm.OLS(y_train,X_train_4).fit()
lm4.summary()
vif = pd.DataFrame()
X=X_train_4.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
############################## ITERATION 5 ##############################
X_train_5=X_train_4.drop(["compressionratio"],axis=1)
X_train_5=sm.add_constant(X_train_5)
lm5 = sm.OLS(y_train,X_train_5).fit()
lm5.summary()
vif = pd.DataFrame()
X=X_train_5.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
##################################  iTERATION 6 ##############################
X_train_6=X_train_5.drop(["carwidth"],axis=1)
X_train_6=sm.add_constant(X_train_6)
lm6 = sm.OLS(y_train,X_train_6).fit()
lm6.summary()
vif = pd.DataFrame()
X=X_train_6.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
sb.heatmap(X_train_6.corr(),annot=True)
###########################################################################################################
X_train_7=X_train_6.drop(["car_cat_Medium"],axis=1)
X_train_7=sm.add_constant(X_train_7)
lm7 = sm.OLS(y_train,X_train_7).fit()
lm7.summary()
vif = pd.DataFrame()
X=X_train_7.drop("const",axis=1)
vif['featues'] = X.columns
vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
#preidting the values
y_train_pred=lm7.predict(X_train_7)
#Finding residuals
res = y_train - y_train_pred
plt.axvline(x=0,color="red")

#distribution of error terms
plt.xlabel("errors",fontsize=15)
sb.distplot(res,rug=True,color='red')
r2_score (y_train, y_train_pred)
#Taking the columns
print(X_train_7.columns)
var=['price', 'enginesize', 'carbody_hardtop', 'fuelsystem_4bbl',
       'car_cat_Low']
df_test[cols]=scale.transform(df_test[cols])
df_test_var=df_test[var]
df_test_var=sm.add_constant(df_test_var)
y_test=df_test_var.pop("price")
X_test_var=df_test_var
X_test_var.head()
y_test_pred=lm7.predict(X_test_var)
res2 = y_test-y_test_pred
plt.axvline(x=0,color="blue")
sb.distplot(res2,rug=True)
#R2 score of the test data
r2_score (y_test, y_test_pred)
#ploting the y_test and Y-pred values
plt.xlabel("y_test",fontsize=15)
plt.ylabel("y_pred",fontsize=15)
plt.scatter(y_test_pred,y_test)
