#Libraries Required

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

data= pd.read_csv("../input/adult.csv")

df= pd.DataFrame(data)
data.head()

data.info()
#Converting Categorical variables into Quantitative variables

print(set(data['occupation']))

data['occupation'] = data['occupation'].map({'?': 0, 'Farming-fishing': 1, 'Tech-support': 2, 

                                                       'Adm-clerical': 3, 'Handlers-cleaners': 4, 'Prof-specialty': 5,

                                                       'Machine-op-inspct': 6, 'Exec-managerial': 7, 

                                                       'Priv-house-serv': 8, 'Craft-repair': 9, 'Sales': 10, 

                                                       'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13, 

                                                       'Protective-serv': 14}).astype(int)







    
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1}).astype(int)
data['race'] = data['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 

                                             'Amer-Indian-Eskimo': 4}).astype(int)
data['marital.status'] = data['marital.status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 

                                                             'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4, 

                                                             'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)



df.occupation.replace(0, np.nan, inplace=True)
print(df.shape)

df=df.dropna()

print(df.shape)
df.head(10)
hmap = df.corr()

plt.subplots(figsize=(12, 9))

sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True);
#SETTING UP DECISSION TREES

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn import metrics





X=df[['education.num','age','hours.per.week']].values

y= df[['income']].values



X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.3, random_state=21, stratify=y)



clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)

predn=clf.predict(X_test)

print('The accuracy of the model is',metrics.accuracy_score(predn,y_test))
#SVM

from sklearn import svm



svc = svm.SVC(kernel='linear')



svc.fit(X_train, y_train)



y_pred=svc.predict(X_test)



print("Test set predictions:\n {}".format(y_pred))

print(svc.score(X_test,y_test))
#Tuning the model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV





param_grid= {'n_neighbors': np.arange(1,80)}

knn = KNeighborsClassifier()

knn_cv=GridSearchCV(knn, param_grid, cv=5)

y = y.reshape(30718,)

knn_cv.fit(X, y)

print(knn_cv.best_params_)

print(knn_cv.best_score_)







#KNN

model=KNeighborsClassifier(n_neighbors=78) 

model.fit(X_train,y_train)

prediction=model.predict(X_test)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))
X1=df[['education.num','age','hours.per.week', 'capital.gain']].values

y1= df[['income']].values



X1_train, X1_test, y1_train, y1_test = train_test_split(X1 ,y1, test_size=0.3, random_state=21, stratify=y)



knn1=KNeighborsClassifier(n_neighbors=78) 

knn1.fit(X1_train,y1_train)

prediction=knn1.predict(X1_test)

print('The accuracy of the KNN1 is',metrics.accuracy_score(prediction,y1_test))
from xgboost import XGBClassifier



X2=df[['education.num','age','hours.per.week', 'capital.gain']].values

y2= df[['income']].values



X2_train, X2_test, y2_train, y2_test = train_test_split(X2 ,y2, test_size=0.3, random_state=21, stratify=y)



# fit model no training data

xgbc = XGBClassifier()

xgbc.fit(X2_train, y2_train)

prediction2=xgbc.predict(X2_test)

print('The accuracy of the xGB is',metrics.accuracy_score(prediction2,y2_test))




