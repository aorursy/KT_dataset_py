

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

#Import wine dataset

df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

#Check data information

df.info()
#Check data first and last 5 rows

df.head()

df.tail()
#Check dataset has any missing data

df.isnull().any()
#number of rows and columns

print(df.shape)
# Data structure

df.describe()
#How data distribute in quality

sns.countplot(x='quality',data=df)

plt.title('Quality Data Distrubution')

plt.xlabel('Quality level')

plt.ylabel('Counts')

plt.show()

print(pd.DataFrame(df['quality'].value_counts()))

#Data relationship

df.corr()

sns.heatmap(df.corr(),annot=False,square=True,vmin=-1,vmax=1,linewidths=.6)
sample_data=df.iloc[750:900]

sns.pairplot(sample_data,plot_kws={'alpha':0.3})
sns.regplot('alcohol','density',data=sample_data,color='darkblue',line_kws={'linewidth':0.9,'color':'black'})
sns.regplot('pH','fixed acidity',data=sample_data,color='darkgreen',line_kws={'linewidth':0.9,'color':'black'})
#Divide quaality into 3 groups

group_bin=[0,4,6,10]

group_labels=['Poor','Acceptable','Good']

df['group']=pd.cut(df.quality,group_bin,3, labels=group_labels)
sns.set(palette="colorblind")

sns.countplot(x='group',data=df)

plt.title('Taste Distrubution')

plt.xlabel('Taste Quality')

plt.ylabel('Counts')

plt.show()

print(pd.DataFrame(df['group'].value_counts()))
#Modelling

#Split data

X=df.iloc[:,:-2]

y=df['group']

print(y.iloc[:10])

lecod=preprocessing.LabelEncoder()  #Encoding Group

y=lecod.fit_transform(y)

print(y[:10])

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state=5)
# Standardize data

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
#SVM

model_SVC=SVC()

model_SVC.fit(X_train_std, y_train)

predictions=model_SVC.predict(X_test_std)

#Classification_report & Confusion_matrix to test the model is good or not

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test,predictions,digits=4))

print(metrics.accuracy_score(y_test,predictions))
#Random forest

rf=RandomForestClassifier(n_estimators=250)

rf.fit(X_train_std,y_train)

Rfpred=rf.predict(X_test_std)

print(classification_report(y_test,Rfpred,digits=4))

print(confusion_matrix(y_test, Rfpred))

print(metrics.accuracy_score(y_test,Rfpred))

print('feature importances',rf.feature_importances_)
#KNN

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train_std,y_train)

knnpred = knn.predict(X_test_std)

print(confusion_matrix(y_test,knnpred))

print(classification_report(y_test,knnpred))

print(metrics.accuracy_score(y_test,knnpred))
##GridSearchcv to find a good estimator

#Increase SVC accuracy to choose proper C and gamma

C_range=[0.01,0.1,1,10,100,1000]

gamma_range=[1,0.1,0.01,0.001,0.0001]

param_grid=dict(gamma=gamma_range, C=C_range)  #set up a dictionary to find suitable gamma & C

model_SVC1=SVC(probability=True)

grid = GridSearchCV(model_SVC1, param_grid=param_grid,cv=4)

grid.fit(X_train_std, y_train)

print('grid.best_params=',grid.best_params_) 

print('grid.best_score =',grid.best_score_)#show the best combination and the best estimator

grid_predictions = grid.predict(X_test_std)#using the best estimator to predict again

print(classification_report(y_test,grid_predictions,digits=4))

print(confusion_matrix(y_test, grid_predictions))

print(metrics.accuracy_score(y_test,grid_predictions))
#Increase Random Forest Accuracy

param_dict={'n_estimators':[10,100,200,250,500],'criterion':['entropy','gini']}

grid=GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_dict,cv=4)

grid.fit(X_train_std, y_train)

print('grid.best_params=',grid.best_params_) 

print('grid.best_score =',grid.best_score_)

grid_predictions = grid.predict(X_test_std)

print(classification_report(y_test,grid_predictions,digits=4))

print(confusion_matrix(y_test, grid_predictions))

print(metrics.accuracy_score(y_test,grid_predictions))
#Increase KNN Accuracy

n=[i+1 for i in range(50)]

param_dict={'n_neighbors':n,'n_jobs':[-1]}

grid=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_dict,cv=4)

grid.fit(X_train_std, y_train)

print('grid.best_params=',grid.best_params_) 

print('grid.best_score =',grid.best_score_)

grid_predictions = grid.predict(X_test_std)

print(classification_report(y_test,grid_predictions,digits=4))

print(confusion_matrix(y_test, grid_predictions))

print(metrics.accuracy_score(y_test, grid_predictions))