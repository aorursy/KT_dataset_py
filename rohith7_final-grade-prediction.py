# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.utils import shuffle
# Any results you write to the current directory are saved as output.
data1 = pd.read_csv("../input/student-mat.csv",sep=",")
data2 = pd.read_csv("../input/student-por.csv",sep=",")
data = [data1,data2]
data=pd.concat(data)
data=shuffle(data)
data=data.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])
sns.catplot(x="school", hue = "sex" , data=data , kind="count",height=6, aspect=.7)
print("percentage total female : ",(data["sex"] == 'F').value_counts(normalize = True)[1]*100)
print("percentage total male : ",(data["sex"] == 'M').value_counts(normalize = True)[1]*100)
import seaborn as sns
from matplotlib.pyplot import figure
figure(figsize=(15, 15))
hmap = sns.heatmap(data.corr(), square=True, annot=True,linewidths=0.5)
#drop some features that have very less correlation values with grades
data = data.drop(["traveltime","famrel","freetime","goout","health","absences"], axis=1)
sns.swarmplot(x="internet",y="G3",hue='address',data=data)
sns.catplot(x="sex", hue = "romantic", data=data , kind="count",height=6, aspect=.7)
sns.swarmplot(x="Dalc",y="G3",hue="sex",data=data)
sns.swarmplot(x="Walc",y="G3",hue="sex",data=data)
#to suppress the warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#data cleaning(binary and onehot encodings)
data.school[data.school == 'GP'] = 1
data.school[data.school == 'MS'] = 0
data.sex[data.sex == 'M'] = 1
data.sex[data.sex == 'F'] = 0
data.address[data.address == 'U'] = 1
data.address[data.address == 'R'] = 0
data.famsize[data.famsize == 'GT3'] = 1
data.famsize[data.famsize == 'LE3'] = 0
data.Pstatus[data.Pstatus == 'T'] = 1
data.Pstatus[data.Pstatus == 'A'] = 0
#for Medu & Fedu its better in numeric as higher education has higher difference with primary than secondary
#one hot encoding of necessary features
cols_to_transform = [ 'Mjob','Fjob','reason','guardian' ]
data=pd.get_dummies(data,columns=cols_to_transform) 
#failures, traveltime, studytime does not require onehot  
data.schoolsup[data.schoolsup == 'yes'] = 1
data.schoolsup[data.schoolsup == 'no'] = 0
data.famsup[data.famsup == 'yes'] = 1
data.famsup[data.famsup == 'no'] = 0
data.paid[data.paid == 'yes'] = 1
data.paid[data.paid == 'no'] = 0
data.activities[data.activities == 'yes'] = 1
data.activities[data.activities == 'no'] = 0
data.nursery[data.nursery == 'yes'] = 1
data.nursery[data.nursery == 'no'] = 0
data.higher[data.higher == 'yes'] = 1
data.higher[data.higher == 'no'] = 0
data.internet[data.internet == 'yes'] = 1
data.internet[data.internet == 'no'] = 0
data.romantic[data.romantic == 'yes'] = 1
data.romantic[data.romantic == 'no'] = 0
#remaining doesn't require onehot as distances are not same. one hot is used only when they are equidistant
data.head(10)
#Final grades
y =  data[[ 'G3']].mean(axis=1)
data = data.drop(["G3"], axis=1)
data.head()
#spliting data to train and test by 80% and 20% respectievely 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,y, test_size=0.2)
# training the model on training set using linear regression
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = np.round(regr.predict(X_test))
meansqr=[]
Avgdiff=[]
r2=[]
meansqr.append(mean_squared_error(y_test, y_pred))
Avgdiff.append(abs(y_test-y_pred).mean())
r2.append(r2_score(y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
print("Mean difference: %.2f"% abs(y_test-y_pred).mean())
print("r2 score: %.2f"% r2_score(y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,y_test)
# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train);
y_pred =np.round(rf.predict(X_test))
meansqr.append(mean_squared_error(y_test, y_pred))
Avgdiff.append(abs(y_test-y_pred).mean())
r2.append(r2_score(y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
print("Mean difference: %.2f"% abs(y_test-y_pred).mean())
print("r2 score: %.2f"% r2_score(y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,y_test)
#svm regressor
from sklearn.svm import SVR
regressor=SVR(kernel="linear",epsilon=1.0,degree=3)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
meansqr.append(mean_squared_error(y_test, y_pred))
Avgdiff.append(abs(y_test-y_pred).mean())
r2.append(r2_score(y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
print("Mean difference: %.2f"% abs(y_test-y_pred).mean())
print("r2 score: %.2f"% r2_score(y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,y_test)
#knearest neighbhors
from sklearn import neighbors
n_neighbors=5
knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
meansqr.append(mean_squared_error(y_test, y_pred))
Avgdiff.append(abs(y_test-y_pred).mean())
r2.append(r2_score(y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
print("Mean difference: %.2f"% abs(y_test-y_pred).mean())
print("r2 score: %.2f"% r2_score(y_test, y_pred))
#plotting y_pred and y_test
t = np.arange(0,len(y_pred) , 1)
plt.plot(t,y_pred,t,y_test)
objects=('LinearReg','RandForReg','SVMregg','Knn')
plt.bar(np.arange(len(meansqr)),meansqr)
plt.xticks(np.arange(len(meansqr)), objects)
plt.title('Mean Square Error')
plt.show()
plt.bar(np.arange(len(Avgdiff)),Avgdiff)
plt.xticks(np.arange(len(Avgdiff)), objects)
plt.title('Absolute Mean Error')
plt.show()
plt.bar(np.arange(len(r2)),r2)
plt.xticks(np.arange(len(r2)), objects)
plt.title('r2_score')
plt.show()


