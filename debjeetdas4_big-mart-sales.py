import pandas as pd

import numpy as np



data= pd.read_csv("../input/bigmart-sales-data/Train.csv")

print(data.shape)

print(data.info())
# Replacing the duplicates with its original categories



data["Item_Fat_Content"] = data["Item_Fat_Content"].str.replace("reg","Regular")

data["Item_Fat_Content"] = data["Item_Fat_Content"].str.replace("LF","Low Fat")

data["Item_Fat_Content"] = data["Item_Fat_Content"].str.replace("low fat","Low Fat")

data["Item_Fat_Content"].unique()
# Percentage of null values in each column:

data.isnull().sum()/len(data)*100
!pip install missingpy
# Imputing the nan values using Knn Imputer for Item_Weight Column

from missingpy import KNNImputer

kn= KNNImputer(weights='distance')

new_weight= kn.fit_transform(data["Item_Weight"].values.reshape(-1,1))

data["Item_Weight"]=new_weight
# Substituting the missing values with mean for Item_Weight Column

data["Item_Weight"]=data["Item_Weight"].fillna(np.mean(data["Item_Weight"]))
#Imputing nan values using fillna "backwordFill" method and analysing the nature of Item_Outlet_Sales after Imputation.



data["Outlet_Size"]=data["Outlet_Size"].fillna(method="bfill")

data.pivot_table(index="Outlet_Size",values="Item_Outlet_Sales")
data.isnull().sum()/len(data)*100
# This shoe the correlation through HeatMap

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(10,5))

sns.heatmap(data.corr(),annot=True,cmap="Blues")

plt.show()
# substituting the outletType with the its frequency of each category

OutletType=data["Outlet_Type"].value_counts()

data["Outlet_Type"]=data["Outlet_Type"].map(OutletType)

data["Outlet_Type"]
# substituting the Outlet_Size with the its frequency of each category

OutletSize=data["Outlet_Size"].value_counts()

data["Outlet_Size"]=data["Outlet_Size"].map(OutletSize)
# substituting the Item_Fat_Content with the its frequency of each category

Item_Fat_Content=data["Item_Fat_Content"].value_counts()

data["Item_Fat_Content"]=data["Item_Fat_Content"].map(Item_Fat_Content)
#converting date to age of Outlet

data["Outlet_Establishment_Year"]= data["Outlet_Establishment_Year"].apply(lambda x : 2009 - x)
# substituting the Outlet_Establishment_Year with the its frequency of each category

Outlet_Establishment_Year=data["Outlet_Establishment_Year"].value_counts()

data["Outlet_Establishment_Year"]=data["Outlet_Establishment_Year"].map(Outlet_Establishment_Year)
# substituting the Outlet_Identifier with the its frequency of each category

Outlet_Identifier=data["Outlet_Identifier"].value_counts()

data["Outlet_Identifier"]=data["Outlet_Identifier"].map(Outlet_Identifier)
# substituting the Item_Identifier with the its frequency of each category

Item_Identifier=data["Item_Identifier"].value_counts()

data["Item_Identifier"]=data["Item_Identifier"].map(Item_Identifier)
# substituting the Item_Identifier with the its frequency of each category

Item_Type=data["Item_Type"].value_counts()

data["Item_Type"]=data["Item_Type"].map(Item_Type)
DV = data["Item_Outlet_Sales"]
IV=data[["Item_Type","Item_MRP","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Type","Outlet_Size","Item_Fat_Content"]]
# Log transformed Dependent Variable

Log_DV = data["Item_Outlet_Sales"].apply(lambda x : np.log(x+1))
# Log transformed Independent Variable

Log_IV=data[["Item_Type","Item_MRP","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Type","Outlet_Size","Item_Fat_Content"]].apply(lambda x : np.log(x+1)) 
# This graph shows the skewness of the features after Log transform

import seaborn as sns

sns.distplot(Log_IV.skew(),hist=False)
# skewness means there might be outliers in the dataset too.

# Thus checking the numerical variables wether it has outliers



box_data=data[["Item_Weight","Item_Visibility","Item_MRP"]]

sns.boxplot(data=box_data,orient="v")
#Linear regression

#After log transformation on Independent and Dependent Variables



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt

from sklearn.linear_model import LinearRegression



train_x,test_x,train_y,test_y=train_test_split(Log_IV,Log_DV,test_size=0.2,random_state=3)

lr= LinearRegression()

lr.fit(train_x,train_y)

y_pred=lr.predict(test_x)

r2scores= r2_score(test_y,y_pred)

rmses= sqrt(mean_squared_error(test_y,y_pred))

print("r2scores : ",r2scores)

print("rmses : ",rmses)
# plotting the coef_ bar graph



%matplotlib inline

iv=train_x.columns

coef1= lr.coef_

coef_col1=pd.Series(coef1,iv).sort_values()

print(coef_col1)

coef_col1.plot(kind="bar",title="Modal coeff")
# Residue plot after Log transformed Variables



residue = test_y - y_pred

import matplotlib.pyplot as plt

plt.scatter(test_y,residue)

plt.axhline(y=0)
#Linear regression

#Without Log Transformation



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt

from sklearn.linear_model import LinearRegression

trainx,testx,trainy,testy=train_test_split(IV,DV,test_size=0.2,random_state=3)

LR= LinearRegression()

LR.fit(trainx,trainy)

y_predict=LR.predict(testx)

r2score= r2_score(testy,y_predict)

rmse= sqrt(mean_squared_error(testy,y_predict))

print("r2scores : ",r2score)

print("rmses : ",rmse)
# Residue plot after Log transformed Variables

res = testy - y_predict

import matplotlib.pyplot as plt

plt.scatter(testy,res)

plt.axhline(y=0)
# Ridge

# after log transformation applied on IV & DV



from sklearn.linear_model import Ridge , Lasso

list1=[0.0018,0.002,0.005,0.08,0.09,0.1,0.5]

list2=[]

for i in list1:

    ridge_reg = Ridge(alpha=i,normalize=True)

    trainxr,testxr,trainyr,testyr=train_test_split(Log_IV,Log_DV,test_size=0.2,random_state=3)

    ridge_reg.fit(trainxr,trainyr)

    y_pred_r=ridge_reg.predict(testxr)

    r2score_r= r2_score(testyr,y_pred_r)

    list2.append(r2score_r)



ridge_rscore_df=pd.DataFrame({"ALPHA":list1,"R2SCORE":list2})

ridge_rscore_df 
# plotting the coef_ bar graph for ridge



%matplotlib inline

riv=trainxr.columns

coef2= ridge_reg.coef_

coef_col2=pd.Series(coef2,riv).sort_values()

print(coef_col2)

coef_col2.plot(kind="bar",title="Modal coeff")
# Residue plot for Ridge

resr = testyr - y_pred_r

import matplotlib.pyplot as plt

plt.scatter(testyr,resr)

plt.axhline(y=0)
# Lasso 

# after log transformation applied on IV & DV





list1L=[0.00185,0.002,0.005,0.08,0.09,0.1,0.5]

list2L=[]

for i in list1L:

    lasso_reg = Lasso(alpha=i)

    trainxl,testxl,trainyl,testyl=train_test_split(Log_IV,Log_DV,test_size=0.2)

    lasso_reg.fit(trainxl,trainyl)

    y_pred_l=lasso_reg.predict(testxl)

    r2score_l= r2_score(testyl,y_pred_l)

    list2L.append(r2score_l)



lasso_rscore_df=pd.DataFrame({"ALPHA":list1L,"R2SCORE":list2L})

lasso_rscore_df
# plotting the coef_ bar graph for Lasso



%matplotlib inline

lasso_reg1 = Lasso(alpha=0.002)

trainxl1,testxl1,trainyl1,testyl1=train_test_split(Log_IV,Log_DV,test_size=0.2)

lasso_reg1.fit(trainxl1,trainyl1)

y_pred_l1=lasso_reg1.predict(testxl1)

r2score_l1= r2_score(testyl1,y_pred_l1)

list2L.append(r2score_l1)

liv=trainxl1.columns

coef3= lasso_reg1.coef_

coef_col3=pd.Series(coef3,liv).sort_values()

print(coef_col3)

coef_col3.plot(kind="bar",title="Modal coeff")

import pandas as pd

testdata = pd.read_csv("../input/bigmart-sales-data/Test.csv")
# Replacing the duplicates with its original categories



testdata["Item_Fat_Content"] =testdata["Item_Fat_Content"].str.replace("reg","Regular")

testdata["Item_Fat_Content"] =testdata["Item_Fat_Content"].str.replace("LF","Low Fat")

testdata["Item_Fat_Content"] =testdata["Item_Fat_Content"].str.replace("low fat","Low Fat")

testdata["Item_Fat_Content"].unique()
# Imputing the nan values using Knn Imputer for the Item_weight col.

from missingpy import KNNImputer

kn1=KNNImputer(weights='distance')

new_weight1=kn1.fit_transform(testdata["Item_Weight"].values.reshape(-1,1))

testdata["Item_Weight"]=new_weight1
#Imputing nan values using fillna "backwordFill" method and analysing the nature of Item_Outlet_Sales after Imputation.



testdata["Outlet_Size"]=testdata["Outlet_Size"].fillna(method="ffill")
# Null values in the test dataset

testdata.isnull().sum()/len(testdata)*100
# substituting the outletType with the its frequency of each category

OutletType1=testdata["Outlet_Type"].value_counts()

testdata["Outlet_Type"]=testdata["Outlet_Type"].map(OutletType1)

testdata["Outlet_Type"]



# substituting the Outlet_Size with the its frequency of each category

OutletSize1=testdata["Outlet_Size"].value_counts()

testdata["Outlet_Size"]=testdata["Outlet_Size"].map(OutletSize1)



# substituting the Item_Fat_Content with the its frequency of each category

Item_Fat_Content1=testdata["Item_Fat_Content"].value_counts()

testdata["Item_Fat_Content"]=testdata["Item_Fat_Content"].map(Item_Fat_Content1)



#converting date to age of Outlet

testdata["Outlet_Establishment_Year"]= testdata["Outlet_Establishment_Year"].apply(lambda x : 2009 - x)





# substituting the Outlet_Establishment_Year with the its frequency of each category

Outlet_Establishment_Year1=testdata["Outlet_Establishment_Year"].value_counts()

testdata["Outlet_Establishment_Year"]=testdata["Outlet_Establishment_Year"].map(Outlet_Establishment_Year1)



# substituting the Outlet_Identifier with the its frequency of each category

Outlet_Identifier1=testdata["Outlet_Identifier"].value_counts()

testdata["Outlet_Identifier"]=testdata["Outlet_Identifier"].map(Outlet_Identifier1)



# substituting the Item_Identifier with the its frequency of each category

Item_Identifier1=testdata["Item_Identifier"].value_counts()

testdata["Item_Identifier"]=testdata["Item_Identifier"].map(Item_Identifier1)



# substituting the Item_Identifier with the its frequency of each category

Item_Type1=testdata["Item_Type"].value_counts()

testdata["Item_Type"]=testdata["Item_Type"].map(Item_Type1)





# Log transformed Dependent Variable

Log_DV1 = data["Item_Outlet_Sales"].apply(lambda x : np.log(x+1))



# Log transformed Independent Variable

Log_IV1 =testdata[["Item_Type","Item_MRP","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Type","Outlet_Size","Item_Fat_Content"]].apply(lambda x : np.log(x+1)) 



#Linear regression

#After log transformation on Independent and Dependent Variables



from sklearn.metrics import r2_score, mean_squared_error

from math import sqrt

from sklearn.linear_model import LinearRegression



lr1= LinearRegression()

lr1.fit(Log_IV,Log_DV)

y_pred1=lr1.predict(Log_IV1)
# creating a new column in the test dataset with predicted values

testdata["pred_sales"]=np.exp(y_pred1+1)
# Final dataset with predicted outlet Sales values in the pred_sales column

testdata["pred_sales"]