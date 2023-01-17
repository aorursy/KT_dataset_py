# General Packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time

# General Mathematics package
import math as math

# Graphing Packages
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use("ggplot")

# Statistics Packages
from scipy.stats import randint
from scipy.stats import skew

# Machine Learning Packages
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn import preprocessing
from skimage.transform import resize
import xgboost as xgb

# Neural Network Packages
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

# H2o packages
import h2o
from h2o.automl import H2OAutoML
student = pd.read_csv("../input/StudentsPerformance.csv")
student.head()
student["overall score"] = np.nan
student["math score letter"] = np.nan
student["reading score letter"] = np.nan
student["writing score letter"] = np.nan
student["overall score letter"] = np.nan
student["overall score"] = round((student["math score"] + 
                                  student["reading score"] + 
                                  student["writing score"])/3 , 2)
student["math score letter"][student["math score"] <= 60] = "F"
student["math score letter"][np.logical_and
                             (student["math score"] > 60 , 
                              student["math score"] <= 69)] = "D"
student["math score letter"][np.logical_and
                             (student["math score"] >= 70 , 
                              student["math score"] <= 73)] = "C-"
student["math score letter"][np.logical_and
                             (student["math score"] >= 74 , 
                              student["math score"] <= 76)] = "C"
student["math score letter"][np.logical_and
                             (student["math score"] >= 77 , 
                              student["math score"] <= 79)] = "C+"
student["math score letter"][np.logical_and
                             (student["math score"] >= 80 , 
                              student["math score"] <= 83)] = "B-"
student["math score letter"][np.logical_and
                             (student["math score"] >= 84 , 
                              student["math score"] <= 86)] = "B"
student["math score letter"][np.logical_and
                             (student["math score"] >= 87 , 
                              student["math score"] <= 89)] = "B+"
student["math score letter"][np.logical_and
                             (student["math score"] >= 90 , 
                              student["math score"] <= 93)] = "A-"
student["math score letter"][np.logical_and
                             (student["math score"] >= 94 , 
                              student["math score"] <= 96)] = "A"
student["math score letter"][student["math score"] >= 97] = "A+"
student["reading score letter"][student["reading score"] <= 60] = "F"
student["reading score letter"][np.logical_and
                             (student["reading score"] > 60 , 
                              student["reading score"] <= 69)] = "D"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 70 , 
                              student["reading score"] <= 73)] = "C-"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 74 , 
                              student["reading score"] <= 76)] = "C"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 77 , 
                              student["reading score"] <= 79)] = "C+"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 80 , 
                              student["reading score"] <= 83)] = "B-"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 84 , 
                              student["reading score"] <= 86)] = "B"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 87 , 
                              student["reading score"] <= 89)] = "B+"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 90 , 
                              student["reading score"] <= 93)] = "A-"
student["reading score letter"][np.logical_and
                             (student["reading score"] >= 94 , 
                              student["reading score"] <= 96)] = "A"
student["reading score letter"][student["reading score"] >= 97] = "A+"
student["writing score letter"][student["writing score"] <= 60] = "F"
student["writing score letter"][np.logical_and
                             (student["writing score"] > 60 , 
                              student["writing score"] <= 69)] = "D"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 70 , 
                              student["writing score"] <= 73)] = "C-"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 74 , 
                              student["writing score"] <= 76)] = "C"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 77 , 
                              student["writing score"] <= 79)] = "C+"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 80 , 
                              student["writing score"] <= 83)] = "B-"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 84 , 
                              student["writing score"] <= 86)] = "B"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 87 , 
                              student["writing score"] <= 89)] = "B+"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 90 , 
                              student["writing score"] <= 93)] = "A-"
student["writing score letter"][np.logical_and
                             (student["writing score"] >= 94 , 
                              student["writing score"] <= 96)] = "A"
student["writing score letter"][student["writing score"] >= 97] = "A+"
student["overall score letter"][student["overall score"] <= 60] = "F"
student["overall score letter"][np.logical_and
                             (student["overall score"] > 60 , 
                              student["overall score"] <= 69.99)] = "D"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 70 , 
                              student["overall score"] <= 73.99)] = "C-"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 74 , 
                              student["overall score"] <= 76.99)] = "C"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 77 , 
                              student["overall score"] <= 79.99)] = "C+"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 80 , 
                              student["overall score"] <= 83.99)] = "B-"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 84 , 
                              student["overall score"] <= 86.99)] = "B"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 87 , 
                              student["overall score"] <= 89.99)] = "B+"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 90 , 
                              student["overall score"] <= 93.99)] = "A-"
student["overall score letter"][np.logical_and
                             (student["overall score"] >= 94 , 
                              student["overall score"] <= 96.99)] = "A"
student["overall score letter"][student["overall score"] >= 97] = "A+"
student.head()
for n in range(0,5):
    student.iloc[:,n] = pd.Categorical(student.iloc[:,n])
# 0 to 5 changes gender to test prepartion course into category
for n in range(-1,-5):
    student.iloc[:,n] = pd.Categorical(student.iloc[:,n])
# -1 to -5 changes math score letter to overall score letter to category
student['overall score'] = pd.to_numeric(student["overall score"])
student.info()
plt.figure(figsize= (20,5))
ax = plt.subplot(131)
pd.crosstab(student["math score letter"] , student.gender).plot(kind = 'bar', ax = ax)
plt.xticks(rotation = 360)
plt.title("Math Score" , size = 20)
plt.legend(loc=2, prop={'size': 15})

ax1 = plt.subplot(132)
pd.crosstab(student["reading score letter"] , student.gender).plot(kind = 'bar', ax = ax1)
plt.xticks(rotation = 360)
plt.title("Reading Score" , size = 20)
plt.legend(loc=2, prop={'size': 15})

ax2 = plt.subplot(133)
pd.crosstab(student["writing score letter"] , student.gender).plot(kind = 'bar', ax = ax2)
plt.title("Writing Score" , size = 20)
plt.xticks(rotation = 360)
plt.legend(loc=2, prop={'size': 15})

pd.crosstab(student["overall score letter"] , student.gender).plot(kind = 'bar' , figsize = (20,5))
plt.xlabel("overall score", size = 30)
plt.xticks(rotation = 360 , size  = 20)
plt.ylabel("count" , size = 20)
plt.legend(loc=2, prop={'size': 15})
plt.show()
pd.crosstab(student["overall score letter"],
            student["parental level of education"]).plot.bar(figsize = (20,10))

plt.xlabel("overall score", size = 30)
plt.xticks(rotation = 360 , size  = 20)
plt.ylabel("count" , size = 30)
plt.title("Parents Degree and Grades" , size = 20)
plt.legend(loc=2, prop={'size': 15})
plt.show()


plt.figure(figsize = (20,8))
sns.boxplot(x="parental level of education", y="overall score", hue="gender" , 
            data=student )
plt.xticks(rotation=360 , size = 17)
plt.yticks( size = 20)
plt.xlabel("Level of Education of Parents", size = 20)
plt.ylabel("overall score" , size = 20)
plt.title("Parents Degree + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()
pd.crosstab(student["overall score letter"],
            student["race/ethnicity"]).plot.bar(figsize = (30,15))

plt.xticks(rotation=360 , size = 30)
plt.yticks( size = 30)
plt.xlabel("overall score", size = 30)
plt.ylabel("count" , size = 30)
plt.title("Ethnicity/Race and Grades" , size = 30)
plt.legend(loc=2, prop={'size': 30})

plt.show()

plt.figure(figsize = (20,8))
sns.boxplot(x="race/ethnicity", y="overall score", hue="gender" , 
            data=student)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Ethnicity/Race", size = 20)
plt.ylabel("overall score" , size = 20)
plt.title("Ethnicity/Race + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()
pd.crosstab(student["overall score letter"],
            student["lunch"]).plot.bar(figsize = (20,5))
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("overall score", size = 20)
plt.ylabel("count" , size = 20)
plt.title("Lunch and Grades" , size = 20)
plt.legend(loc=2, prop={'size': 20})
plt.show()

plt.figure(figsize = (20,8))
sns.boxplot(x="lunch", y="overall score", hue="gender" , 
            data=student)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Lunch", size = 20)
plt.ylabel("overall score" , size = 20)
plt.title("Lunch + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()
pd.crosstab(student["overall score letter"],
            student["test preparation course"]).plot.bar(figsize = (20,5))
plt.xticks(rotation = 360)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("overall score", size = 20)
plt.ylabel("count" , size = 20)
plt.title("Test Prep and Grades" , size = 20)
plt.legend(loc=2, prop={'size': 20})
plt.show()

plt.figure(figsize = (20,8))
sns.boxplot(x="test preparation course", y="overall score", hue="gender" , 
            data=student)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Test Prep", size = 20)
plt.ylabel("Overall Score" , size = 20)
plt.title("Test Preparation + Gender and Grades" , size = 20)
plt.legend(loc=3, prop={'size': 20})
plt.show()
sns.heatmap(student.corr() , square = True , annot = True , 
            linewidths=.8 , cmap="YlGnBu")
plt.title("Correlation Map")
plt.show()
sns.lmplot(x="reading score", y="writing score", hue="gender",
           data=student)
plt.show()
student_1 = student
le = preprocessing.LabelEncoder()
for n in range(0,5):
    student_1.iloc[:,n] = le.fit_transform(student_1.iloc[:,n])
for n in range(0,5):
    student_1.iloc[:,n] = pd.Categorical(student_1.iloc[:,n])
student_1['overall score'] = pd.to_numeric(student_1["overall score"])
student_1["math score letter"] = pd.Categorical(student_1["math score letter"])
student_1["reading score letter"] = pd.Categorical(student_1["reading score letter"])
student_1["writing score letter"] = pd.Categorical(student_1["writing score letter"])
student_1["overall score letter"] = pd.Categorical(student_1["overall score letter"])
student_2 = student_1
student_1 = student_1.replace(["F" , "D" , "C-" , "C" , "C+", "B-" , "B" , "B+", "A-" , "A" , "A+"] , [0,1,2,3,4,5,6,7,8,9,10])

#    F = 0
#    D = 1
#    C- = 2
#    C = 3
#    C+ = 4
#    B- = 5
#    B = 6
#    B+ = 7
#    A- = 8
#    A = 9
#    A+ = 10
student_1.head()
X_score = student_1.drop(["overall score letter" , "overall score"] , axis = 1)
y_score = student_1["overall score letter"]

X_train, X_val, y_train, y_val = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_val)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print("accuracy",accuracy_score(y_val , y_pred)*100,"%")
param_grid = {'n_neighbors' : np.arange(1,50)}

knn_cv = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_cv, param_grid, cv = 5)

knn_cv.fit(X_train , y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred_knn = knn_cv.predict(X_val)

print("accuracy:" , accuracy_score(y_val , y_pred_knn)*100 , "%")
tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred_DT = tree.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_DT))
print("accuracy:",round(accuracy_score(y_val , y_pred_DT)*100,2) , "%")
param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")
param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")
warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_val)

print(classification_report(y_val , y_pred_SVC))
print('accuracy:' , accuracy_score(y_val , y_pred_SVC)*100,"%")
student_int = student_1[0:-1].astype("int64")

X_score_int = student_int.drop(["overall score letter" ,"overall score"] , axis = 1)
y_score_int = student_int["overall score letter"]

X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(X_score_int, y_score_int, test_size = 0.3, random_state=42)
warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_val_int)

print("accuracy:",round(accuracy_score(y_val_int, y_pred_XGB)*100,2) , "%")
student_2.replace(["C-" , "C" , "C+" ,] , ("C") , inplace = True)
student_2.replace(["B-" , "B" , "B+" ,] , ("B") , inplace = True)
student_2.replace(["A-" , "A" , "A+" ,] , ("A") , inplace = True)
student_2.replace(["F" , "D" , "C" , "B" , "A"] , (0,1,2,3,4) , inplace = True)

#    F = 0
#    D = 1
#    C = 2
#    B = 3
#    A = 4
X_score = student_2.drop(["overall score letter" , "overall score"] , axis = 1)
y_score = student_2["overall score letter"]

X_train, X_test, y_train, y_test = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred))
print("accuracy",accuracy_score(y_test , y_pred)*100,"%")
tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred_DT = tree.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_DT))
print("accuracy:",round(accuracy_score(y_test , y_pred_DT)*100,2) , "%")
param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_RF))
print("accuracy:",round(accuracy_score(y_test , y_pred_RF)*100,2),"%")
param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_RF))
print("accuracy:",round(accuracy_score(y_test , y_pred_RF)*100,2),"%")
warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_test)

print(classification_report(y_test , y_pred_SVC))
print('accuracy:' , round(accuracy_score(y_test , y_pred_SVC)*100,2),"%")
student_int = student_2[0:-1].astype("int64")

X_score_int = student_int.drop(["overall score letter" ,"overall score"] , axis = 1)
y_score_int = student_int["overall score letter"]

X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_score_int, y_score_int, test_size = 0.3, random_state=42)
warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_test_int)

print("accuracy:",round(accuracy_score(y_test_int, y_pred_XGB)*100,2) , "%")
X_score = student_1.drop(["overall score letter" , "overall score"] , axis = 1)
y_score = student_1["overall score"]

X_train, X_test, y_train, y_test = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)
reg = linear_model.LinearRegression()
reg.fit(X_train , y_train)
y_pred_reg = reg.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test,y_pred_reg))
print("RMSE: " , RMSE)
MSE = mean_squared_error(y_test,y_pred_reg )
print("MSE: " ,MSE)
print("R^2: " , r2_score(y_test, y_pred_reg))
ridge = Ridge(alpha = 0.4 , normalize = True)

ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("RMSE: " , np.sqrt(mean_squared_error(y_test,ridge_pred)))
print("MSE: " , mean_squared_error(y_test,ridge_pred))
print("R^2: " , r2_score(y_test, ridge_pred))
lasso = Lasso(alpha = 0.4, normalize = True , max_iter = 1000000)

lasso.fit(X_train,y_train)

lasso_pred = ridge.predict(X_test)
print("RMSE: " , np.sqrt(mean_squared_error(y_test,lasso_pred)))
print("MSE: " , mean_squared_error(y_test,lasso_pred))
print("R^2: " , r2_score(y_test, lasso_pred))
X_score = student_1.drop(["gender"] , axis = 1)
y_score = student_1["gender"]

X_train, X_val, y_train, y_val = train_test_split(X_score, y_score, test_size = 0.3, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_val)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print("accuracy",round(accuracy_score(y_val , y_pred)*100,2),"%")
param_grid = {'n_neighbors' : np.arange(1,50)}

knn_cv = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_cv, param_grid, cv = 5)

knn_cv.fit((X_train) , (y_train))

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred_knn = knn_cv.predict(X_val)

print("accuracy:" , round(accuracy_score(y_val , y_pred_knn)*100,2) , "%")
tree = DecisionTreeClassifier()

tree.fit((X_train) , (y_train))
y_pred_DT = tree.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_DT))
print("accuracy:",accuracy_score(y_val , y_pred_DT)*100 , "%")
param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 5 , verbose = 1)
tree_cv.fit((X_train) , (y_train))
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")
param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 5 , verbose = 1)
tree_cv.fit((X_train) ,(y_train))
y_pred_RF = tree_cv.predict(X_val)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred_RF))
print("accuracy:",round(accuracy_score(y_val , y_pred_RF)*100,2),"%")
warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_val)

print(classification_report(y_val , y_pred_SVC))
print('accuracy:' , round(accuracy_score(y_val , y_pred_SVC)*100,2),"%")
student_int = student_1[0:-1].astype("int64")

X_score_int = student_int.drop(["gender"] , axis = 1)
y_score_int = student_int["gender"]

X_train_int, X_val_int, y_train_int, y_val_int = train_test_split(X_score_int, y_score_int, test_size = 0.3, random_state=42)
warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_val_int)

print("accuracy:",accuracy_score(y_val_int, y_pred_XGB)*100 , "%")
student = pd.read_csv("../input/StudentsPerformance.csv")
from sklearn.cluster import KMeans

gender = student_1['gender'].values
X_gen = student_1.drop('gender' , axis = 1).values

model = KMeans(n_clusters = 2)
model.fit(X_gen)
labels = model.predict(X_gen)
xs = X_gen[:,4]
ys = X_gen[:,5]

f, (ax1, ax2) = plt.subplots(figsize=(20, 10) , ncols = 2)

sns.set()

sns.scatterplot(x="math score", y="reading score", hue="gender" , data=student , ax = ax1)
plt.xticks(rotation=360 , size = 20)
plt.yticks( size = 20)
plt.xlabel("Math Score", size = 20)
plt.ylabel("Reading Score" , size = 20)
plt.title("Actual Gender Data" , size = 20)
#plt.legend(loc=2, prop={'size': 20})

sns.scatterplot(x = xs,y = ys , hue = labels , ax=ax2)
plt.xlabel("Math Score" , size = 20)
plt.ylabel("Reading Score" , size = 20)
plt.title("KMeans Predicting Gender" , size = 20)
plt.legend(loc=2, prop={'size': 20})
plt.show()

acc = round(accuracy_score(labels , student_1.gender)*100,2)
print(('accuracy: '+ str(acc)+" %").center(125))