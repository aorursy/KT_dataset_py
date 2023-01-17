import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#load data
df = pd.read_excel(r"../input/analysis-of-breaking-of-machineries/Combined_Cycle_powerplant.xlsx")
df.shape
df.columns
df.head()
df.info()
df.describe()
#Check for missing values
df.duplicated().sum()
#Drop duplicates
df.drop_duplicates(inplace=True)
#check for missing values
df.isnull().sum()
#it comapres all column with all the columns and shows the graph
sns.pairplot(df)
plt.show()
cor=df.corr()
plt.figure(figsize=(9,7))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()
#Separate features and label
x = df[['AT','V','AP','RH']]
y = df[['PE']]
# split data into train and test
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
# we have to split the data into 80% as train and 20% as test so we have specified test_size as 0.2
print(x.shape)
print(xtr.shape)
print(xts.shape)
print(y.shape)
print(ytr.shape)
print(yts.shape)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#train the model with the training data
model.fit(xtr,ytr)
new_data=np.array([[13.97,39.16,1016.05,84.6]])
model.predict(new_data)
#get prediction of xts
ypred = model.predict(xts)
#calculating r2score
from sklearn.metrics import r2_score
r2_score(yts,ypred)
#To find the error
from sklearn.metrics import mean_squared_error
mean_squared_error(yts,ypred)
import joblib
#from sklearn.externals import joblib
joblib.dump(model,r"C:\Users\HP\Desktop\p.practice\ML AI COURSE\ccpp_model.pkl")
