import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
home_file=pd.read_csv('../input/breastcancer-dataset/data.csv')
home_file.describe()
home_file['diagnosis'].value_counts()
home_file.shape
home_file.keys()
X=home_file.drop(['diagnosis', 'Unnamed: 32'], axis=1)
y=home_file.diagnosis.map(dict(M=1,B=0))
home_file.info()
y.dtype
print(y.tail())
X.head()
X.head()
sns.pairplot(home_file, hue='diagnosis', vars= ['radius_mean', 'texture_mean','area_mean','perimeter_mean','smoothness_mean'])
sns.scatterplot(x='area_mean', y='compactness_mean', hue='diagnosis', data=home_file)
plt.figure(figsize=(20,10))
sns.heatmap(home_file.corr(), annot=True)
X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=0)
X_train.head()
y_test[:5]
my_model=RandomForestClassifier(n_estimators=100,random_state=0)
my_model.fit(X_train, y_train)
y_prediction=my_model.predict(X_test)
np.mean(y_prediction==y_test)
my_model.score(X_test,y_test)
my_model.score(X_train,y_train)
#to check the auc for the random_forest without tunning
false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_prediction)
roc_auc=auc(false_positive_rate, true_positive_rate)
roc_auc
#to checkmate overfitting, we try to see the best number of estimators or finetune our model
n_estimators=[1,2,4,8,16,32,64,100,200]

train_results=[]
test_results=[]
for estimator in n_estimators:
    my_model=RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    my_model.fit(X_train, y_train)
    
    y_prediction1=my_model.predict(X_train) #we are monitoring the curve for predicting X_train values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_train, y_prediction1)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    train_results.append(roc_auc)
    
    
    y_prediction2=my_model.predict(X_test) # NOW, we are monitoring the curve for predicting X_test values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_prediction2)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    test_results.append(roc_auc)
    
    
    
test_results
line1,=plt.plot(n_estimators, train_results, 'b', label='Train AUC')
line2,=plt.plot(n_estimators, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('n_estimators')
plt.show()
#lets see the range we need to consider in choosing our max_depth
max_depths=np.linspace(1,32,32, endpoint=True)

train_results=[]
test_results=[]
for max_depth in max_depths:
    my_model=RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    my_model.fit(X_train, y_train)
    
    y_prediction1=my_model.predict(X_train) #we are monitoring the curve for predicting X_train values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_train, y_prediction1)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    train_results.append(roc_auc)
    
    
    y_prediction2=my_model.predict(X_test) # NOW, we are monitoring the curve for predicting X_test values for different estimators
    
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test, y_prediction2)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    
    test_results.append(roc_auc)
    
    
    
line1,=plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2,=plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1:HandlerLine2D(numpoints=2)})

plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()
my_model=RandomForestClassifier(n_estimators=100,max_depth=3, n_jobs=-1)
my_model.fit(X_train, y_train)
y_prediction2=my_model.predict(X_test)
my_model.score(X_test,y_test)
my_model.score(X_train,y_train)
y_new=my_model.predict(X)
new_table=pd.DataFrame({'id':home_file['id'],'diagnosis':y_new} )
new_table.head()