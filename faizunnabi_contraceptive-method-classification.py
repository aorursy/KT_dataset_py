import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/cmc.data.txt',names=['Wife Age','Wife Education','Husband Education','Children',
                                                'Wife religion','Wife working','Husband Occupation','SOLI',
                                                'Media Exposure','Contraceptive Method'])
data.head()
sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis',cbar=False)
data.info()
data.describe()
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
sns.distplot(data['Wife Age'],bins=50)
sns.jointplot(x='Wife Age',y='Children',data=data)
plt.figure(figsize=(10,6))
sns.distplot(data['Children'],bins=30,kde=False,color="red",hist_kws={'edgecolor':'red'})
sns.countplot(x='Contraceptive Method',data=data,hue="Wife religion")
plt.figure(figsize=(10,6))
sns.countplot(x='Contraceptive Method',data=data,hue="SOLI")
plt.figure(figsize=(10,6))
sns.countplot(x='SOLI',data=data)
plt.figure(figsize=(10,6))
sns.countplot(x='Contraceptive Method',data=data,hue="Wife working")
plt.figure(figsize=(10,6))
sns.countplot(x='Contraceptive Method',data=data,hue="Media Exposure")
sns.pairplot(data)
plt.figure(figsize=(14,6))
sns.countplot(x='Wife Age',data=data,hue="Contraceptive Method")
plt.figure(figsize=(14,6))
sns.boxplot(x='Contraceptive Method',y='Wife Age',data=data)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X=data.drop('Contraceptive Method',axis=1)
y=data['Contraceptive Method']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
rfc=RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred=svc.predict(X_test)
print(classification_report(y_test,pred))
param_grid = {'C': [0.1,1,2,3, 10,20,30,40,50,60,70,80,90, 100,200,300, 1000], 'gamma': [1,0.1,0.01,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))



