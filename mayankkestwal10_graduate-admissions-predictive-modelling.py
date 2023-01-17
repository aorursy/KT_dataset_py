
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualisation
import matplotlib.pyplot as plt #data visualisation
import warnings  #Filter warnings
warnings.filterwarnings("ignore")

#Reference for dataset files
import os
print(os.listdir("../input/"))


#Reading the dataset, creating a DataFrame from Admission_Predict.csv
df = pd.read_csv('../input/Admission_Predict.csv')
df.head()
#Removing Serial No. column from our DataFrame
df.drop(columns=['Serial No.'],inplace=True)
df.head()
df.info(),
print("Shape",df.shape) #shape of dataset => (400,8)
df.describe() #statistical inferences
df.isnull().sum() #To check missing values without using describe()
#columns depict no missing count
#Data Visualisation
#Comparing every feature with the target variable
plt.figure(figsize=(17,7))
sns.scatterplot(df['CGPA'],df['Chance of Admit '],hue=df['Chance of Admit '])
plt.figure(figsize=(17,7))
sns.scatterplot(df['GRE Score'],df['Chance of Admit '],hue=df['Chance of Admit '])
plt.figure(figsize=(17,7))
sns.scatterplot(df['TOEFL Score'],df['Chance of Admit '],hue=df['Chance of Admit '])
plt.figure(figsize=(17,7))
sns.barplot(df['University Rating'],df['Chance of Admit '])
plt.figure(figsize=(17,7))
sns.barplot(df['SOP'],df['Chance of Admit '])
plt.figure(figsize=(17,7))
sns.barplot(df['LOR '],df['Chance of Admit '])
#Finding out how features are correlated to each other in our data
corr = df.corr()
sns.heatmap(corr,annot=True,cmap='Blues') #creates a heatmap that tells the correlation among all the features
#Comparing these potential features among each others
plt.figure(figsize=(17,7))
sns.scatterplot(df['GRE Score'],df['TOEFL Score'],hue=df['University Rating'],palette=['red','blue','green','yellow','purple'])
plt.figure(figsize=(17,7))
sns.scatterplot(df['GRE Score'],df['CGPA'],hue=df['University Rating'],palette=['red','blue','green','yellow','purple'])
plt.figure(figsize=(17,7))
sns.scatterplot(df['CGPA'],df['TOEFL Score'],hue=df['University Rating'],palette=['red','blue','green','yellow','purple'])
#Reading Test dataset
testdf = pd.read_csv('../input/Admission_Predict_Ver1.1.csv',skiprows=400) #skipping first 400 rows
test_org = testdf.copy() #reserve original test DataFrame
testdf.columns=['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','Chance of Admit']
testdf.drop(testdf[['Serial No.','Chance of Admit']],axis=1,inplace=True) #drop Serial No., and Chance of Admit
testdf.head()
#Preparing data for our model
x = df.iloc[:,:-1].values
y = df.iloc[:,7].values
testX = testdf.iloc[:,:].values
testY = test_org.iloc[:,8].values
#Importing libraries
from sklearn.preprocessing import StandardScaler #Feature Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)
testX = scaler.fit_transform(testX)
#Importing Linear Regression model from sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)  #Fitting training data
#Prediction on test data
predict = regressor.predict(testX)
regressor.score(x,y)
#Checking mean squared error
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
mse = mean_squared_error(predict,testY)
mae = mean_absolute_error(predict,testY)
msle = mean_squared_log_error(predict,testY)
print("mean_squared_error : %f\nmean_absolute_error : %f\nmean_squared_log_error : %f"%(mse,mae,msle))
from sklearn.tree import DecisionTreeRegressor
treeRegressor = DecisionTreeRegressor(criterion='mse',random_state=0,max_depth=6)
treeRegressor.fit(x,y)
predict = treeRegressor.predict(testX)
treeRegressor.score(x,y)
mse = mean_squared_error(predict,testY)
mae = mean_absolute_error(predict,testY)
msle = mean_squared_log_error(predict,testY)
print("mean_squared_error : %f\nmean_absolute_error : %f\nmean_squared_log_error : %f"%(mse,mae,msle))
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=110,criterion='mse',max_depth=6,random_state=0)
rf.fit(x,y)
predict = rf.predict(testX)
rf.score(x,y)
mse = mean_squared_error(predict,testY)
mae = mean_absolute_error(predict,testY)
msle = mean_squared_log_error(predict,testY)
print("mean_squared_error : %f\nmean_absolute_error : %f\nmean_squared_log_error : %f"%(mse,mae,msle))
