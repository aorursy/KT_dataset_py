import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
sfc = pd.read_csv('../input/train.csv')

lat=sfc['X'].values
long=sfc['Y'].values
category=sfc['Category'].values
#Convert Weekaday from Text to Numeric
week=sfc['DayOfWeek'].values
week_map={'Monday':'1', 'Tuesday':'2', 'Wednesday':'3', 'Thursday':'4', 'Friday':'5', 'Saturday':'6', 'Sunday':'7'}
week1=[]
for i in range(len(week)):
    week1.append(int(week_map[str(week[i])]))
print(week1)
#Create Month, Year and Hour list from DateStamp
month=[]
year=[]
hour=[]
for i in range(len(sfc['Dates'])):
    t = pd.tslib.Timestamp(sfc['Dates'][i])
    month.append(t.month)
    year.append(t.year)
    hour.append(t.hour)
df=pd.DataFrame({'Latitude':lat, 'Longtitude':long, 'Category':category, 'Hour':hour, 'Week':week1, 'Month':month, 'Year':year})
print(df)
import copy
X=copy.deepcopy(df)
X.drop('Category', axis=1, inplace=True)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df['Category'])
Y=le.transform(df['Category']) 
print(Y)
df['Y']=Y
corr = df.corr()
print(corr)
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#Convert to 3 Dimensional point(x, y, z) from Latitude and Longtitude and use as input.
import math
X4=[]
Y4=[]
Z4=[]
for i in range(len(df['Latitude'])):
    X4.append(math.cos(df['Latitude'][i])*math.cos(df['Longtitude'][i]))
    Y4.append(math.cos(df['Latitude'][i])*math.sin(df['Longtitude'][i]))
    Z4.append(math.sin(df['Latitude'][i]))
from sklearn.neighbors import KNeighborsClassifier
X3=pd.DataFrame({'X':X4, 'Y':Y4, 'Z':Z4})
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, Y, test_size=0.2)
neigh3 = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh3.fit(X_train3,y_train3) 
y_pred3 = neigh3.predict(X_test3)
print("Accuracy is ", accuracy_score(y_test3,y_pred3)*100,"% for K-Value:")
#Use only Latitude and Longtitude as input data.
X2=pd.DataFrame({'Latitude':df['Latitude'], 'Longitude':df['Longtitude']})
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y, test_size=0.2)
#Implement KNN(So we take K value to be )
neigh2 = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh2.fit(X_train2,y_train2) 
y_pred2 = neigh2.predict(X_test2)
print("Accuracy is ", accuracy_score(y_test2,y_pred2)*100,"% for K-Value:")
#Use Hour, Week, Month, Year, Latitude, Longtitude as input data.
neigh = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh.fit(X_train,y_train) 
y_pred = neigh.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:")
#Use the columns of dataframe that has positive correlation with category of crime.
X6=pd.DataFrame({'Hour':df['Hour'], 'Month':df['Month'], 'Week':df['Week']})
X_train6, X_test6, y_train6, y_test6 = train_test_split(X6, Y, test_size=0.2)
neigh6 = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh6.fit(X_train6,y_train6) 
y_pred6 = neigh6.predict(X_test6)
print("Accuracy is ", accuracy_score(y_test6,y_pred6)*100,"% for K-Value:")
#Try on a smaller part of dataset maybe the model is overfitting.
df1 = df.sample(frac=0.2).reset_index(drop=True)
X1=copy.deepcopy(df1)
X1.drop('Category', axis=1, inplace=True)
Y1=df1['Category']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2)
#Implement KNN(So we take K value to be )
neigh1 = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh1.fit(X_train1,y_train1) 
y_pred1 = neigh1.predict(X_test1)
print("Accuracy is ", accuracy_score(y_test1,y_pred1)*100,"% for K-Value:")
#Check k for best results of KNN
from sklearn.neighbors import KNeighborsClassifier
for K in range(100):
    K_value = K+1
neigh = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train) 
y_pred = neigh.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",50)

#Implement KNN for the best features after Feature Engineering and best value of K
neigh = KNeighborsClassifier(n_neighbors = 50, weights='uniform', algorithm='auto')
neigh.fit(X_train,y_train) 
y_pred = neigh.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:")
#Implement Grid Serch for best Gamma, C and Selection between rbf and linear kernel
from sklearn import svm, datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
clf.fit(X_train1, y_train1)   
print('Best score for data1:', clf.best_score_) 
print('Best C:',clf.best_estimator_.C) 
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)
#OVA SVM(Grid Search Results: Kernel - linear, C -1 , Gamma - auto)
from sklearn import svm
lin_clf = svm.LinearSVC(C=1)
lin_clf.fit(X_train2, y_train2)
y_pred2=lin_clf.predict(X_test2)
print(accuracy_score(y_test2,y_pred2)*100)
#OVA SVM(Grid Search Results: Kernel - rbf, C -1 , Gamma - auto)
from sklearn import svm
lin_clf=svm.SVC(kernel='rbf')
lin_clf.fit(X_train2, y_train2)
y_pred2=lin_clf.predict(X_test2)
print(accuracy_score(y_test2,y_pred2)*100)
#SVM by Crammer(Grid Search Results: Gamma - Auto, C - 1)
lin_clf = svm.LinearSVC(C=1, multi_class='crammer_singer')
lin_clf.fit(X_train2, y_train2)
y_pred2=lin_clf.predict(X_test2)
print(accuracy_score(y_test2,y_pred2)*100)
#Implementing OVA Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train2, y_train2)
y_pred2 = clf.predict(X_test2)
print(accuracy_score(y_test2,y_pred2)*100)
#Implementing OVA Logistic Regerssion
from sklearn.linear_model import LogisticRegression
X2=pd.DataFrame({'Latitude':df['Latitude'], 'Longitude':df['Longtitude']})
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y, test_size=0.2)
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train2, y_train2)
y_pred = logisticRegr.predict(X_test2)
print(accuracy_score(y_test,y_pred)*100)
data_dict={}
target = df["Category"].unique()
count = 1
for data in target:
    data_dict[data] = count
    count+=1
print(data_dict)
from collections import OrderedDict
data_dict_new = OrderedDict(sorted(data_dict.items()))
print(data_dict_new)
import pandas as pd
test_dataset = pd.read_csv('C:/Users/RAHUL/AppData/Roaming/SPB_16.6/San Francisco Data Set/test.csv')
df_test=pd.DataFrame({'Latitude':test_dataset['X'],'Longtitude':test_dataset['Y']})

predictions = neigh2.predict(df_test)

#One Hot Encoding of knn as per submission format
result_dataframe = pd.DataFrame({
    "Id": test_dataset["Id"]
})
for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_knn.csv", index=False) 
predictions = logisticRegr.predict(df_test)
#One Hot Encoding of logistic regression as per submission format
result_dataframe = pd.DataFrame({
    "Id": test_dataset["Id"]
})
for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_logistic.csv", index=False) 