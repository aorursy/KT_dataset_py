import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



CarDekho_Data=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
CarDekho_Data.head()
CarDekho_Data.shape
CarDekho_Data.info()
CarDekho_Data.drop("Car_Name",axis=1,inplace=True)
CarDekho_Data.isnull().sum().sort_values(ascending=False)
CarDekho_Data.describe()
CarDekho_Data.columns
CarDekho_Data.duplicated().any()
duplicateRowsDF = CarDekho_Data[CarDekho_Data.duplicated()]
duplicateRowsDF
CarDekho_Data["Owner"]=CarDekho_Data["Owner"].astype("object")
#Fuel type

figsize=(5,7)

sns.countplot(CarDekho_Data["Fuel_Type"])
#Seller Type

figsize=(5,7)

sns.countplot(CarDekho_Data["Seller_Type"])
#Transmission 

figsize=(5,7)

sns.countplot(CarDekho_Data["Transmission"])
#Car Name 

figsize=(5,7)

sns.countplot(CarDekho_Data["Owner"])
CarDekho_Data.columns
Year_trans=CarDekho_Data.groupby(["Year","Owner"])["Present_Price"].mean().reset_index()



plt.figure(figsize=(20,10))

sns.barplot(x="Year",y="Present_Price",hue="Owner",data=Year_trans)

plt.xlabel("Year")

plt.ylabel("Average Present price")

plt.show()
Year_trans_sell=CarDekho_Data.groupby(["Year","Owner"])["Selling_Price"].mean().reset_index()



plt.figure(figsize=(20,10))

sns.barplot(x="Year",y="Selling_Price",hue="Owner",data=Year_trans_sell)

plt.xlabel("Year")

plt.ylabel("Average Selling Price")

plt.show()
Owner_kms=CarDekho_Data.groupby(["Owner"])["Kms_Driven"].mean().reset_index()



plt.figure(figsize=(10,5))

sns.barplot(x="Owner",y="Kms_Driven",data=Owner_kms)

plt.xlabel("Owner Type")

plt.ylabel("Average Kms driven")

plt.show()
Seller_kms=CarDekho_Data.groupby(["Seller_Type"])["Kms_Driven"].mean().reset_index()



plt.figure(figsize=(10,5))

sns.barplot(x="Seller_Type",y="Kms_Driven",data=Seller_kms)

plt.xlabel("Seller Type")

plt.ylabel("Average Kms driven")

plt.show()
sns.boxplot(x="Transmission",y="Selling_Price",data=CarDekho_Data,)
sns.boxplot(x="Seller_Type",y="Selling_Price",data=CarDekho_Data)
CarDekho_Data
CarDekho_Data.info()
Categorical_data=[]

Numerical_data=[]



for i,c in enumerate(CarDekho_Data.dtypes):

  if c==object:

    Categorical_data.append(CarDekho_Data.iloc[:,i])



  else:

    Numerical_data.append(CarDekho_Data.iloc[:,i])
Categorical_data=pd.DataFrame(Categorical_data).transpose()

Numerical_data=pd.DataFrame(Numerical_data).transpose()
Categorical_data.head()
Numerical_data.head()
from sklearn.preprocessing import LabelEncoder

LE=LabelEncoder()
for i in Categorical_data:

  Categorical_data[i]=LE.fit_transform(Categorical_data[i])
Categorical_data.head()
CarDekho_Data_Proces=pd.concat([Categorical_data,Numerical_data],axis=1)
CarDekho_Data_Proces.head()
plt.figure(figsize=(10,7))

plt.title="Correlation Matrix"

sns.heatmap(CarDekho_Data_Proces.corr(),annot=True)
CarDekho_Data_Proces.boxplot(figsize=(15,5))
CarDekho_Data_Proces["log_Kms_Driven"]= np.log10(CarDekho_Data_Proces["Kms_Driven"])
CarDekho_Data_Proces.boxplot(figsize=(15,5))
#This fuction is used to download the files from google collab.

#CarDekho_Data_Proces.to_csv("CarDekho.csv")

#files.download("CarDekho.csv")
CarDekho_Data_Proces.head()
from scipy import stats #importing the stats based library.
stats.ttest_1samp(a=CarDekho_Data_Proces["Selling_Price"],popmean=5.5) 
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score,accuracy_score

X=CarDekho_Data_Proces.drop(["Selling_Price"],axis=1)

Y=CarDekho_Data_Proces["Selling_Price"]
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=28)
print(X_train.shape,Y_train.shape)
from sklearn.preprocessing import StandardScaler

Sc=StandardScaler()

X_train=Sc.fit_transform(X_train)

X_test=Sc.fit_transform(X_test)
le=LinearRegression()

le.fit(X_train,Y_train)

y_pred=le.predict(X_test)
from sklearn.metrics import r2_score
r2_score(Y_test,y_pred)
x=CarDekho_Data_Proces.drop(["Selling_Price"],axis=1) #Independent variables

y=CarDekho_Data_Proces["Selling_Price"] #Dependent Variables
import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std
model=sm.OLS(endog=y,exog=x).fit()

model.summary()
x=CarDekho_Data_Proces.drop(["Selling_Price","Owner"],axis=1) #Independent variables

y=CarDekho_Data_Proces["Selling_Price"] #Dependent Variables
model=sm.OLS(endog=y,exog=x).fit()

model.summary()