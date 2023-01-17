import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/diabetes.csv",sep=',')
data.head(10) 
data.info()
data.shape
data['Outcome'].value_counts()
sns.countplot(x='Outcome',data=data, palette='Dark2')

plt.show()
bmi_median = data['BMI'].median()
data['BMI'].replace(0,bmi_median)
insulin_median = data['Insulin'].median()
data['Insulin'].replace(0,insulin_median)
plt.figure(figsize=(10,10))

plt.title('Pearson Correlation of Variables',y=1, size=15)

sns.heatmap(data.corr(),linewidths=0.1,vmax=0.1,square=True,linecolor='white',annot=True)
sns.boxplot(data.Outcome,data.BMI)
sns.boxplot(data.Outcome,data.Age)
from sklearn.model_selection import train_test_split
X = data.drop('Outcome',axis=1)

y = data['Outcome']

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler(with_mean=False)
X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
from sklearn.svm import SVC
svc_model = SVC(kernel='rbf',random_state=0)
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_auc_score,roc_curve
auc = roc_auc_score(y_test,y_pred)

print("AUC %0.3f" %auc)
fpr, tpr, thresholds = roc_curve(y_test,y_pred)
plt.figure(figsize=(10,10))

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr, label='SVM (AUC = %0.2f)'% auc)

plt.xlabel("1-Specificity",fontsize=12)

plt.ylabel("Sensitivity",fontsize=12)

plt.legend(loc='lower right')

plt.show()
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,verbose=2)

grid.fit(X_train,y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
#Calculate AUC Score after GridSearchCV

auc_grid = roc_auc_score(y_test,grid_predictions)

print('AUC: %.3f' % auc_grid)
#Calculate ROC Curve after Grid Search CV

fpr , tpr , thresholds = roc_curve(y_test,grid_predictions)
plt.figure(figsize=(10,10))

plt.plot([0,1],[0,1],linestyle="--")

plt.title('Receiver Operator Characteristic')

plt.plot(fpr,tpr, label='SVM (AUC = %0.2f)'% auc_grid)

plt.xlabel("1-Specificity",fontsize=12)

plt.ylabel("Sensitivity",fontsize=12)

plt.legend(loc='lower right')

plt.show()