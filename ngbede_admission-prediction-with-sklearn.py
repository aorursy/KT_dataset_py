#Importing Neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.metrics as skl
from sklearn import preprocessing
from math import sqrt
admin = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
admin.head()
admin.drop("Serial No.",axis=1,inplace=True)
admin.columns
admin.corr()
admin.describe()
score = ((admin["GRE Score"] + (340 * (admin["TOEFL Score"])/120)) / 2) 
score = pd.DataFrame(score)
score.describe()
admin["GRE Score"] = score
admin.head()
admin.drop("TOEFL Score",axis=1,inplace=True)
admin.head()
columns = list(admin.columns)
columns
for cols in columns:
    print(cols,"\n",admin[cols].unique(),"\n")
columns.pop() # remove chance_admit
columns
cat = list(columns[1:4])
cat.append("Research")
def boxplot(data,cols,y="Chance of Admit "):
    for col in cols:
        plt.figure(figsize=(10,10))
        sns.set_style("darkgrid")
        sns.boxplot(data[col],data[y])
        plt.title("Boxplot of Chance Of Admission Vs. "+col.title(),size=20)
        plt.xlabel(col.title(),size=15)
        plt.ylabel(y.title(),size=15)
        plt.show()
boxplot(admin,cat)
num = ['GRE Score','CGPA']
def joint(data,cols,kin,y="Chance of Admit "):
    for col in cols:
        sns.set_style('whitegrid')
        plt.figure(figsize=(10,10))
        sns.jointplot(data[col],data[y],kind=kin)
        plt.title(kin.title()+" plot of Chance Of Admission Vs. "+col.title(),size=20)
        plt.xlabel(col.title(),size=15)
        plt.ylabel(y.title(),size=15)
        plt.show()
joint(admin,num,"scatter")
joint(admin,num,"kde")
joint(admin,num,"hex")
features = admin["University Rating"]
LB = preprocessing.LabelEncoder()
LB.fit(features)
features = LB.transform(features)
OH = preprocessing.OneHotEncoder()
OH.fit(features.reshape(-1,1))
enc = OH.fit(features.reshape(-1,1))
features = enc.transform(features.reshape(-1,1)).toarray()
features.shape
temp = admin[["GRE Score","CGPA"]].values
x_values = np.concatenate([features,temp],axis=1)
x_values
y_values = admin["Chance of Admit "].values
x_train,x_test,y_train,y_test = train_test_split(x_values,y_values,test_size=100,random_state=89)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
x_values.shape
scaler = preprocessing.StandardScaler().fit(x_train[:,5:])
x_train[:,5:] = scaler.transform(x_train[:,5:])
x_test[:,5:] = scaler.transform(x_test[:,5:])
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train)
pred = LR.predict(x_test)
pred[:5]
print("R2-Score:",skl.r2_score(pred,y_test))
print("Root Mean Squared Error:",sqrt(skl.mean_squared_error(pred,y_test)))
print("Median Absolute Error:",skl.median_absolute_error(pred,y_test))
print("Mean absolute error:",skl.mean_absolute_error(pred,y_test))
