import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import  LinearRegression
from sklearn.model_selection import cross_validate
sns.set()
data=pd.read_csv("../input/ecommerce-customer-device-usage/Ecommerce Customers")
data.head()
data.describe()
data.info()
data.isnull().sum()
numeric_data = data.select_dtypes(include=[np.number])
# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in numeric_data:
    if plotnumber<=9 :     # as there are 9 columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(numeric_data[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show()

# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in numeric_data:
    if plotnumber<=9 :     # as there are 9 columns in the data
        ax = plt.subplot(3,3,plotnumber)
        sns.scatterplot(numeric_data[column],numeric_data["Yearly Amount Spent"])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.show()
sns.heatmap(numeric_data.corr(),annot=True)

data["Avatar"].unique()
data["Avatar"].value_counts(sort=True)

labelencoder = LabelEncoder()
data["Avatar"]=labelencoder.fit_transform(data["Avatar"])
data.info()

sns.heatmap(data.corr(),annot=True)
x=data[["Avg. Session Length","Time on App","Length of Membership"]]
y=data["Yearly Amount Spent"]
scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)

linearmodel=LinearRegression()
cross_validate(linearmodel,X_scaled,y,cv=4)
linearmodel.fit(X_scaled,y)
linearmodel.score(X_scaled,y)
