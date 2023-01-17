import pandas as pd

import numpy as np

pd.options.display.max_columns=None

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import zscore

import warnings

warnings.filterwarnings('ignore')
# Loading the dataset whcih having city and region columns

house=pd.read_csv('/kaggle/input/house-price/innercityn.csv')

y=house['price']
# Shape and Size of the dataset

print("The shape of the dataset",house.shape)

print("The size of the dataset",house.size)
#Check the null values in the dataset

house.isnull().sum()
#Description of the dataset

house.describe().transpose()
#Checking the datatypes of the dataset

house.info()
house.head()
#Extracting the year and Month from the dayhours feature

house['dayhours']=house['dayhours'].apply(lambda x:x.rstrip('T0'))

house['year']=house['dayhours'].apply(lambda x:x[0:4]) # The Year which house was sold

house['month']=house['dayhours'].apply(lambda x:x[4:6])

house.drop('dayhours',axis=1,inplace=True) #droping the dayhours feature
#Dropping the Cid feature

house.drop('cid',axis=1,inplace=True)

house.drop('zipcode',axis=1,inplace=True) #Already we extracted region column from zipcode
#Creating the Two columns

house['year']=house['year'].astype('int64')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

house['age']=house['year']-house['yr_built'] # Age of the house

house['rage']=house['year']-house['yr_renovated'] # The Age of the house after renovation done

house.drop(['yr_built','yr_renovated','year'],axis=1,inplace=True) # Removal of these columns
#Some of the columns shows datatype as numerical but it should be categorical so,changed columns to object

house[['room_bed','ceil','coast','sight','condition','quality','furnished','room_bath','age','rage']]=house[['room_bed','ceil','coast','sight','condition','quality','furnished','room_bath','age','rage']].astype('object')
#Extracting num cols and we are not applying zscore for price feature

num_cols=house.select_dtypes(['int64','float64']).columns

num_cols1=num_cols[1:]

num_cols1
#Exrtacting numerical features to house_num dataframe

house_num=house[num_cols1]
#Applying zscore to numerical features

house_num[num_cols1]=house_num[num_cols1].apply(zscore)

house_num.head()
# EXtracting the outliers and replacing with them null values

for i in range(len(num_cols1)):  # number of columns

    for j in range(len(house_num)):  # number of rows

        if abs(house_num[num_cols1[i]][j])>3:  # condition to extract outliers

            house_num[num_cols1[i]].replace({house_num[num_cols1[i]][j]:np.nan},inplace=True)
#Chceking the null values in the after replacement of outliers with null

house_num.isnull().sum()
#Dropping the numerical columns in the orginal house dataset

house.drop(['living_measure', 'lot_measure', 'ceil_measure', 'basement', 'lat',

       'long', 'living_measure15', 'lot_measure15', 'total_area'],axis=1,inplace=True)
#Concating house_num data to original house data

housef=pd.concat([house,house_num],axis=1)

housef.head()
housef.drop(['City','price'],axis=1,inplace=True)
#Applying the label Encoder for some specific columns

labels=['age','room_bath','rage','ceil']

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in labels:

    housef[i]=le.fit_transform(house[i])
#Get dummies for Region column

housef=pd.get_dummies(housef,columns=['Region'],drop_first=True)
!pip install impyute
from impyute.imputation.cs import mice # Importing the MICE
allcols=housef.columns # Getting all columns
housef[allcols]=housef[allcols].astype('float64') # For MICE every feature should be in Numerical
#Applying MICE to the whole dataset

house_mice=mice(housef)
house_mice.columns=housef.columns
# For MICE we have converted all features to numerical, so we are again converting to original datatype

cat_cols=['room_bed', 'room_bath', 'ceil', 'coast', 'sight', 'condition',

       'quality', 'furnished', 'month', 'age', 'rage',

       'Region_North East', 'Region_North West', 'Region_South East',

       'Region_South West']

house_mice[cat_cols]=house_mice[cat_cols].astype('object')
house_mice.info()
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
Xtrain,Xtest,Ytrain,Ytest=train_test_split(house_mice,y,test_size=0.2,random_state=10)
Xtrain.shape
lr=LinearRegression()

lr.fit(Xtrain,Ytrain)

ytrain_lr=lr.predict(Xtrain)

ypred_lr=lr.predict(Xtest)

a=lr.score(Xtrain,Ytrain) # TRAINING ACCRACY

b=lr.score(Xtest,Ytest)  # TEST ACCURACY

print("The Training Accuracy is",a*100) 

print("The Test Accuracy is ",b*100)

rmse_lr=np.sqrt(mean_squared_error(Ytrain,ytrain_lr)) # TRAIN RMSE

rmse_lr1=np.sqrt(mean_squared_error(Ytest,ypred_lr)) #TEST RMSE

print("The Train RMSE for Linear Regression is",rmse_lr)

print("The Test RMSE for Linear Regression is",rmse_lr1)
df=pd.DataFrame({'columns':Xtrain.columns})

#lr.coef_

df['Coeffciants']=lr.coef_

df.head()
Xtrain.columns
rf=RandomForestRegressor()

rf.fit(Xtrain,Ytrain)

ytrain_rf=rf.predict(Xtrain)

ypred_rf=rf.predict(Xtest)

c=rf.score(Xtrain,Ytrain) # TRAIN ACCURAY

d=rf.score(Xtest,Ytest) #TEST ACCURACY

print("The Training Accuracy is",c*100) 

print("The Test Accuracy is ",d*100) 

rmse_rf=np.sqrt(mean_squared_error(Ytrain,ytrain_rf)) # TRANING RMSE

rmse_rf1=np.sqrt(mean_squared_error(Ytest,ypred_rf)) # TEST RMSE

print("The TRAIN RMSE for Random is",rmse_rf)

print("The TEST RMSE for Random Forest Regression is",rmse_rf1)
gb=GradientBoostingRegressor()

gb.fit(Xtrain,Ytrain)

ypred_gb=gb.predict(Xtest)

ytrain_gb=gb.predict(Xtrain)

e=gb.score(Xtrain,Ytrain) # TRAIN ACCURACY

f=gb.score(Xtest,Ytest) #TEST ACCURACY

print("The Training Accuracy is",e*100) 

print("The Test Accuracy is ",f*100) 

rmse_gb=np.sqrt(mean_squared_error(Ytrain,ytrain_gb)) #TRAIN RMSE

rmse_gb1=np.sqrt(mean_squared_error(Ytest,ypred_gb)) # TEST RMSE

print("The TRAIN RMSE for Gradient Regression is",rmse_gb)

print("The TEST RMSE for Gradient Regression is",rmse_gb1)
gb=GradientBoostingRegressor(random_state=100)

bg=BaggingRegressor(base_estimator=gb)

bg.fit(Xtrain,Ytrain)

ypred_bg=bg.predict(Xtest)

ytrain_bg=bg.predict(Xtrain)

g=bg.score(Xtrain,Ytrain)

h=bg.score(Xtest,Ytest)

print("The Training Accuracy is",g*100)  #RAIN ACCURACY

print("The Test Accuracy is ",h*100)  # TEST ACCURACY

rmse_bg=np.sqrt(mean_squared_error(Ytrain,ytrain_bg)) #TRAIN RMSE

rmse_bg1=np.sqrt(mean_squared_error(Ytest,ypred_bg)) #TEST RMSE

print("The TRAIN RMSE for Bagging Regression is",rmse_bg)

print("The TEST RMSE for Bagging Regression is",rmse_bg1)
models=[ypred_lr,ypred_rf,ypred_gb,ypred_bg]

r2_score=[]

adr2_score=[]

for i in models:

    SS_Residual = sum((Ytest-i)**2)

    SS_Total = sum((Ytest-np.mean(Ytest))**2)

    r_squared = 1 - (float(SS_Residual))/SS_Total

    r2_score.append(r_squared*100)

    adjusted_r_squared = 1 - (1-r_squared)*(len(Ytest)-1)/(len(Ytest)-Xtrain.shape[1]-1)

    adr2_score.append(adjusted_r_squared*100)
bestmodel=pd.DataFrame({'Model':['LR','RF','GB','BG']})

bestmodel['Train RMSE']=[round(rmse_lr),round(rmse_rf),round(rmse_gb),round(rmse_bg)]

bestmodel['Test RMSE']=[round(rmse_lr1),round(rmse_rf1),round(rmse_gb1),round(rmse_bg1)]

bestmodel['Train R2_Score %']=[round(a*100,2),round(c*100,2),round(e*100,2),round(g*100,2)]

bestmodel['Test R2_Score %']=[round(b*100,2),round(d*100,2),round(f*100,2),round(h*100,2)]

bestmodel['Adj_Score']=adr2_score
bestmodel
from sklearn.feature_selection import RFE



lr=LinearRegression()

# .fit(X_train,y_train)

rfe = RFE(lr,14)

rfe.fit(Xtrain,Ytrain)

print(rfe.support_)

print(rfe.ranking_)

idc_rfe = pd.DataFrame({"rfe_support" :rfe.support_,

                       "columns" :Xtrain.columns ,

                       "ranking" : rfe.ranking_,

                      })

cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()
house_rfe=house_mice[cols]
Xtrain,Xtest,Ytrain,Ytest=train_test_split(house_rfe,y,test_size=0.3,random_state=10)
lr=LinearRegression()

lr.fit(Xtrain,Ytrain)

ytrain_lr=lr.predict(Xtrain)

ypred_lr=lr.predict(Xtest)

a=lr.score(Xtrain,Ytrain) # TRAINING ACCRACY

b=lr.score(Xtest,Ytest)  # TEST ACCURACY

print("The Training Accuracy is",a*100) 

print("The Test Accuracy is ",b*100)

rmse_lr=np.sqrt(mean_squared_error(Ytrain,ytrain_lr)) # TRAIN RMSE

rmse_lr1=np.sqrt(mean_squared_error(Ytest,ypred_lr)) #TEST RMSE

print("The Train RMSE for Linear Regression is",rmse_lr)

print("The Test RMSE for Linear Regression is",rmse_lr1)
df1=pd.DataFrame({'columns':Xtrain.columns})

df1['Coef_']=lr.coef_

df1
rf=RandomForestRegressor()

rf.fit(Xtrain,Ytrain)

ytrain_rf=rf.predict(Xtrain)

ypred_rf=rf.predict(Xtest)

c=rf.score(Xtrain,Ytrain) # TRAIN ACCURAY

d=rf.score(Xtest,Ytest) #TEST ACCURACY

print("The Training Accuracy is",c*100) 

print("The Test Accuracy is ",d*100) 

rmse_rf=np.sqrt(mean_squared_error(Ytrain,ytrain_rf)) # TRANING RMSE

rmse_rf1=np.sqrt(mean_squared_error(Ytest,ypred_rf)) # TEST RMSE

print("The TRAIN RMSE for Random is",rmse_rf)

print("The TEST RMSE for Random Forest Regression is",rmse_rf1)
gb=GradientBoostingRegressor()

gb.fit(Xtrain,Ytrain)

ypred_gb=gb.predict(Xtest)

ytrain_gb=gb.predict(Xtrain)

e=gb.score(Xtrain,Ytrain) # TRAIN ACCURACY

f=gb.score(Xtest,Ytest) #TEST ACCURACY

print("The Training Accuracy is",e*100) 

print("The Test Accuracy is ",f*100) 

rmse_gb=np.sqrt(mean_squared_error(Ytrain,ytrain_gb)) #TRAIN RMSE

rmse_gb1=np.sqrt(mean_squared_error(Ytest,ypred_gb)) # TEST RMSE

print("The TRAIN RMSE for Gradient Regression is",rmse_gb)

print("The TEST RMSE for Gradient Regression is",rmse_gb1)
gb=GradientBoostingRegressor()

bg=BaggingRegressor(base_estimator=gb)

bg.fit(Xtrain,Ytrain)

ypred_bg=bg.predict(Xtest)

ytrain_bg=bg.predict(Xtrain)

g=bg.score(Xtrain,Ytrain)

h=bg.score(Xtest,Ytest)

print("The Training Accuracy is",g*100)  #RAIN ACCURACY

print("The Test Accuracy is ",h*100)  # TEST ACCURACY

rmse_bg=np.sqrt(mean_squared_error(Ytrain,ytrain_bg)) #TRAIN RMSE

rmse_bg1=np.sqrt(mean_squared_error(Ytest,ypred_bg)) #TEST RMSE

print("The TRAIN RMSE for Bagging Regression is",rmse_bg)

print("The TEST RMSE for Bagging Regression is",rmse_bg1)
models=[ypred_lr,ypred_rf,ypred_gb,ypred_bg]

r2_score=[]

adr2_score=[]

for i in models:

    SS_Residual = sum((Ytest-i)**2)

    SS_Total = sum((Ytest-np.mean(Ytest))**2)

    r_squared = 1 - (float(SS_Residual))/SS_Total

    r2_score.append(r_squared*100)

    adjusted_r_squared = 1 - (1-r_squared)*(len(Ytest)-1)/(len(Ytest)-Xtrain.shape[1]-1)

    adr2_score.append(round(adjusted_r_squared*100,2))
bestmodel=pd.DataFrame({'Model':['LR','RF','GB','BG']})

bestmodel['Train RMSE']=[round(rmse_lr),round(rmse_rf),round(rmse_gb),round(rmse_bg)]

bestmodel['Test RMSE']=[round(rmse_lr1),round(rmse_rf1),round(rmse_gb1),round(rmse_bg1)]

bestmodel['Train R2_Score %']=[round(a*100,2),round(c*100,2),round(e*100,2),round(g*100,2)]

bestmodel['Test R2_Score %']=[round(b*100,2),round(d*100,2),round(f*100,2),round(h*100,2)]

bestmodel['Adj_Score']=adr2_score

bestmodel