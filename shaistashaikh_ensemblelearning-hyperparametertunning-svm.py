# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/income/train.csv')
df
df_Us_Country=df.loc[df['native-country'] == 'United-States']

df_Us_Country
numeric_features = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']

cat_features=[ col for col in list(df.columns) if df[col].dtype =='object']
cat_features
df_Us_Countrydummy=pd.get_dummies(df_Us_Country[cat_features])
df_Us_Countrydummy
df_Us_Countrydummy.shape
final_df_Us_Countrydummy=pd.concat([df_Us_Countrydummy , df_Us_Country[numeric_features],df_Us_Country['income_>50K']], axis = 1)
final_df_Us_Countrydummy.info()
final_df_Us_Countrydummy.isna()
final_df_Us_Countrydummy=final_df_Us_Countrydummy.fillna(0)
final_df_Us_Countrydummy
x=final_df_Us_Countrydummy.drop(['income_>50K'],axis=1)
y=final_df_Us_Countrydummy['income_>50K']
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size = 0.3, random_state = 0) 
print(X_train.shape)
print(X_test.shape)
from sklearn.tree import DecisionTreeClassifier  
Model1 = DecisionTreeClassifier(criterion='gini')  
Model1.fit(X_train, y_train) 
pred1=Model1.predict(X_test)
#ytest1=y_test
pred1
#ytest1
from sklearn.neighbors import KNeighborsClassifier


Model2 = KNeighborsClassifier(n_neighbors= 7)  
Model2.fit(X_train, y_train)
pred2=Model2.predict(X_test)
#ytest2=y_test
pred2

from sklearn.linear_model import LogisticRegression

Model3 = LogisticRegression()
Model3.fit(X_train,y_train)
pred3=Model3.predict(X_test)
pred3
estimators = []
estimators.append(('dt', Model1))
estimators.append(('KNN', Model2))
estimators.append(('lr', Model3))
estimators
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix  
#Manual Ensemble learning 

ensemble_df = pd.DataFrame()
ensemble_df['Pred1'] = pred1
ensemble_df['Pred2'] = pred2
ensemble_df['Pred3'] = pred3
ensemble_df['Sum'] = ensemble_df.sum(axis = 1)
ensemble_df['Final'] = ensemble_df['Sum'] > 2 
ensemble_df['Final'] = ensemble_df['Final'].astype(int)
print(ensemble_df.head())

acc = accuracy_score(y_test,ensemble_df['Final'])
print("Accuracy for Emsemble model {} %".format(acc*100))
print(confusion_matrix(y_test,ensemble_df['Final']))
print('f1 Score -->' ,f1_score(y_test,ensemble_df['Final']))


#Weighted Average Ensemble Learning

ensemble_df = pd.DataFrame()
ensemble_df['Pred1'] = pred1
ensemble_df['Pred2'] = pred2
ensemble_df['Pred3'] = pred3

# DT 10% , KNN 5%  LR 5%  GB 40% RF 40 % 

ensemble_df['Sum'] = 0.3*ensemble_df['Pred1'] + 0.3*ensemble_df['Pred2'] + \
                     0.4*ensemble_df['Pred3'] 
ensemble_df['Final'] = ensemble_df['Sum'] >= 0.4
ensemble_df['Final'] = ensemble_df['Final'].astype(int)

print(ensemble_df.head())

acc = accuracy_score(y_test,ensemble_df['Final'])
print("Accuracy for Emsemble model {} %".format(acc*100))
print(confusion_matrix(y_test,ensemble_df['Final']))
print('f1 Score -->' ,f1_score(y_test,ensemble_df['Final']))
# Voting Classifier 
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('dt', Model1),('KNN',Model2),('lr', Model3)])
model.fit(X_train, y_train)
model.score(X_test,y_test)
fpred=model.predict(X_test)
fpred

acc_1 = accuracy_score(y_test,fpred)
print("Accuracy  = {} %".format(acc_1*100))
print(confusion_matrix(y_test, fpred))
print('f1 Score -->' ,f1_score(y_test,fpred))
#Hyper Parameter Tuning of Decision Tree
from sklearn.model_selection import GridSearchCV
g1 = DecisionTreeClassifier()

param_grid = { 
    'criterion': ['gini', 'entropy'],
    'max_depth': [5,10,15]
}
gs1 = GridSearchCV(estimator=g1, param_grid=param_grid, cv= 5, verbose = 3)
gs1.fit(X_train, y_train)
gs1.best_params_
best_model_1 = gs1.best_estimator_

y1 = best_model_1.predict(X_test)  

acc = accuracy_score(y_test,y1)
print("Accuracy for Grid Search DT  model {} %".format(acc*100))


print(confusion_matrix(y_test, y1))
g2 = KNeighborsClassifier()

param_grid = { 
   
  
    'n_neighbors':[5,6,7,8,9,10],
    'leaf_size':[1,2,3,5],
         
}
g2.get_params().keys()
g1.get_params().keys()
gs2 = GridSearchCV(estimator=g2, param_grid=param_grid, cv= 5, verbose = 3)
gs2.fit(X_train, y_train)
best_model_2 = gs2.best_estimator_

y2 = best_model_2.predict(X_test)  

acc = accuracy_score(y_test,y2)
print("Accuracy for Grid Search KNN  model {} %".format(acc*100))


print(confusion_matrix(y_test, y2))
g3 = LogisticRegression()
g3.get_params().keys()

param_grid = { 
   
  
    'class_weight':[10,15],
    'random_state':[2],
         
}
gs3 = GridSearchCV(estimator=g3, param_grid=param_grid, cv= 5, verbose = 3)
gs3.fit(X_train, y_train)
best_model_3 = gs3.best_estimator_

y3 = best_model_3.predict(X_test)  

acc = accuracy_score(y_test,y3)
print("Accuracy for Grid Search LR  model {} %".format(acc*100))


print(confusion_matrix(y_test, y3))
#Manual Ensemble learning 

ensemble_dfHTN = pd.DataFrame()
ensemble_dfHTN['Pred1'] = y1
ensemble_dfHTN['Pred2'] = y2
ensemble_dfHTN['Pred3'] = y3
ensemble_dfHTN['Sum'] = ensemble_dfHTN.sum(axis = 1)
ensemble_dfHTN['Final'] = ensemble_df['Sum'] > 2 
ensemble_dfHTN['Final'] = ensemble_df['Final'].astype(int)
print(ensemble_dfHTN.head())

acc = accuracy_score(y_test,ensemble_df['Final'])
print("Accuracy for Emsemble model {} %".format(acc*100))
print(confusion_matrix(y_test,ensemble_df['Final']))
print('f1 Score -->' ,f1_score(y_test,ensemble_df['Final']))
#pred3=Model3.predict(X_test)
#pred3

#fpredict=(pred1+pred2+pred3)/3
#fpredict
#fpred=(pred1+pred2+pred3)
#fpred
#model.score(x_test,y_test)
#from sklearn.metrics import accuracy_score #importing accuracy_score function from sklearn.metrics package
#acc_1 = accuracy_score(y_test,fpred)
#print("Accuracy  = {} %".format(acc_1*100))


#from sklearn.metrics import classification_report, confusion_matrix  
#print(confusion_matrix(y_test, fpred))
# feature scaling

from sklearn.preprocessing import StandardScaler

st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

#X_train[:5]
# creating the model and training it
#The kernel parameter can be tuned to take “Linear”,”Poly”,”rbf” etc.
#The gamma value can be tuned by setting the “Gamma” parameter.
#The C value in Python is tuned by the “Cost” parameter in R.

from sklearn.svm import SVC

svcclf = SVC(kernel='rbf').fit(X_train, y_train)
# checking the confision matrix

from sklearn.metrics import confusion_matrix, classification_report

y_pred = svcclf.predict(X_test)
confusion_matrix(y_test, y_pred)
# classification report

print(classification_report(y_test, y_pred))
# training and testing scores

print('Training Set Score: {:.3f}'.format(svcclf.score(X_train, y_train)))
print('Testing Set Score: {:.3f}'.format(svcclf.score(X_test, y_test)))