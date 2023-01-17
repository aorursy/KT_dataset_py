#Importing the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

#Loading the dataset
data = pd.read_csv("../input/insurance.csv")
data.head()
#Checking the summary of the dataset
data.describe()
#Checking for Null values in the dataset
data.isnull().sum()
# Making copy of the dataset
df = data.copy()
#Encoding the features
from sklearn.preprocessing import LabelEncoder
#smoker
labelencoder_smoker = LabelEncoder()
df.smoker = labelencoder_smoker.fit_transform(df.smoker)
#sex
labelencoder_sex = LabelEncoder()
df.sex = labelencoder_sex.fit_transform(df.sex)
#region
labelencoder_region = LabelEncoder()
df.region = labelencoder_region.fit_transform(df.region)

df.head()
df.corr()['charges'].sort_values()
%matplotlib inline
plt.hist(df.charges,bins = 10,alpha=0.5,histtype='bar',ec='black')
plt.title("Frequency Distribution of the charges")
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()
sns.boxplot(x=data.region,y=data.charges,data=data)
plt.title("Medical charges per region")
plt.show()
sns.boxplot(x=data.smoker,y=data.charges,data=data)
plt.title("Medical charges of Smokers and Non-Smokers")
plt.show()
f = plt.figure(figsize=(12,5))
ax = f.add_subplot(121)
sns.distplot(df[df.smoker==1]['charges'],color='c',ax=ax)
ax.set_title('Medical charges for the smokers')

ax = f.add_subplot(122)
sns.distplot(df[df.smoker==0]['charges'],color='b',ax=ax)
ax.set_title('Medical charges for non-smokers')
plt.show()
sns.boxplot(x=data.sex,y=data.charges,data=data)
plt.title("Charges by Gender")
plt.show()
plt.subplot(1,2,1)
sns.distplot(df[df.smoker==1]['age'],color='red')
plt.title("Distribution of Smokers")

plt.subplot(1,2,2)
sns.distplot(df[df.smoker==0]['age'],color='green')
plt.title("Distribution of Non-Smokers")
plt.show()
sns.lmplot(x="bmi",y='charges',hue='smoker',data=data)

sns.lmplot(x='age',y='charges',hue='smoker',data=data,palette='inferno_r')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Considering all the features

X = df.iloc[:,:6].values
Y = df.iloc[:,6].values

#Splitting the dataset into train and test set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

# Training
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.score(X_test,Y_test))

#Not considering region
X1 = df.iloc[:,[0,1,2,3,4]].values
Y1 = df.iloc[:,6].values

X_train,X_test,Y_train,Y_test = train_test_split(X1,Y1,test_size=0.25)

#Training
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

print(regressor.score(X_test,Y_test))
from sklearn.preprocessing import PolynomialFeatures

poly_reg  = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_poly,Y,test_size=0.25)

lin_reg = LinearRegression()
lin_reg  = lin_reg.fit(X_train,Y_train)

print(lin_reg.score(X_test,Y_test))
pred_train = lin_reg.predict(X_train)
pred_test = lin_reg.predict(X_test)

plt.scatter(pred_train,pred_train - Y_train,label='Train data',color='mediumseagreen')
plt.scatter(pred_test,pred_test-Y_test,label="Test data",color='darkslateblue')
plt.legend(loc = 'upper right')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.show()
