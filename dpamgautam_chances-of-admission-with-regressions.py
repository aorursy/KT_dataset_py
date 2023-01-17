import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sys

import seaborn as sns

import os

print(os.listdir("../input"))

df = pd.read_csv("../input/Admission_Predict.csv", sep=",")
print("there are {} columns in the dataset.".format(len(df.columns)))

print(df.columns)

print("there are {} samples in the dataset.".format(df.shape[0]))
df = df.rename(columns={'Chance of Admit ':'Chance of Admit'})

print(df.columns)
df.info()
df.head(5)
fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), ax=ax, annot=True)

plt.show()
print("Not having research : ", len(df[df.Research == 0]))

print("having research : ", len(df[df.Research == 1]))
x = ["Not having research", "Having research"]

y = np.array( [len(df[df.Research==0]), len(df[df.Research==1])] )

plt.bar(x,y)

plt.xlabel("candidate type")

plt.ylabel("number of candidate")

plt.show()
df["TOEFL Score"].plot(kind='hist', bins=200, figsize=(4,4))

plt.show()
df['GRE Score'].plot(kind='hist', bins=200, figsize=(4,4))

plt.show()
# university ratings VS cgpa analysis



plt.scatter(df["University Rating"], df['CGPA'])

plt.show()
# cgpa VS gre score



plt.scatter(df["GRE Score"], df["CGPA"])

plt.show()
# cgpa VS toefl score



plt.scatter(df["TOEFL Score"], df["CGPA"])

plt.show()
# gre score VS toefl score



plt.scatter(df["TOEFL Score"], df["GRE Score"])

plt.show()
# preparing dataset for regression

# the column "Serial No." is of no use, so we will delete it for now



df = pd.read_csv("../input/Admission_Predict.csv", sep=',')



# save serial no. for future use (if in case)

serialNo = df["Serial No."].values



df.drop(['Serial No.'], axis=1, inplace=True)



df = df.rename(columns={"Chance of Admit ": "Chance of Admit"})
df.columns
y = df["Chance of Admit"].values

x = df.drop(["Chance of Admit"], axis=1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
import warnings

warnings.filterwarnings("ignore")
# normalization

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train, x_test = sc.fit_transform(x_train), sc.fit_transform(x_test)
# Linear Regression



from sklearn.linear_model import LinearRegression

lr = LinearRegression()



lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)



from sklearn.metrics import r2_score



print("r2 score of test set is : ", r2_score(y_test, y_pred_lr))

print("real value of y_test[0] is : " + str(y_test[0]) + " and predicted is : " + str(lr.predict(x_test[[0],:])))

print("real value of y_test[1] is : " + str(y_test[1]) + " and predicted is : " + str(lr.predict(x_test[[1],:])))
# Random Forest Regression



from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()



rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)



print("r2 score of test set is : ", r2_score(y_test, y_pred_rf))

print("real value of y_test[0] is : " + str(y_test[0]) + " and predicted is : " + str(rf.predict(x_test[[0],:])))

print("real value of y_test[1] is : " + str(y_test[1]) + " and predicted is : " + str(rf.predict(x_test[[1],:])))
# Decision Tree Regression



from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()



dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)



print("r2 score of test set is : ", r2_score(y_test, y_pred_dt))

print("real value of y_test[0] is : " + str(y_test[0]) + " and predicted is : " + str(dt.predict(x_test[[0],:])))

print("real value of y_test[1] is : " + str(y_test[1]) + " and predicted is : " + str(dt.predict(x_test[[1],:])))
x = ["LinearReg", "RandomForestReg", "DecisionTreeReg"]

y = np.array([ r2_score(y_test,y_pred_lr), r2_score(y_test,y_pred_rf), r2_score(y_test,y_pred_dt) ])



plt.bar(x,y)

plt.xlabel("type of regression algorithm")

plt.ylabel("r2 score")

plt.title("comparision of regression algorithms")

plt.show()
# plot of comparision of predicted output from all 3 regression algorithms with the actual value for the indexes 0,10,20,30,40 etc.



red = plt.scatter(np.arange(0,80,10), y_pred_lr[0:80:10], color='red')

blue = plt.scatter(np.arange(0,80,10), y_pred_rf[0:80:10], color='blue')

black = plt.scatter(np.arange(0,80,10), y_pred_dt[0:80:10], color='black')

green = plt.scatter(np.arange(0,80,10), y_test[0:80:10], color='green')



plt.xlabel("index of candidate")

plt.ylabel("chances of admit")

plt.title("comparision of chances of admit from all 3 regression algorithms with actual value of chances")

plt.legend((red,blue,black,green), ("LR","RF","DT","Actual"))

plt.show()