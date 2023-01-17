# import libraries 

import pandas as pd # Import Pandas for data manipulation using dataframes

import numpy as np # Import Numpy for data statistical analysis 

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import seaborn as sns # Statistical data visualization

# %matplotlib inline
# Import Cancer data drom the Sklearn library

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer
cancer.keys()
print(cancer['DESCR'])
print(cancer['target_names'])
print(cancer['target'])
print(cancer['feature_names'])
print(cancer['data'])

cancer['data'].shape
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

df_cancer.head()
df_cancer.tail()
len(df_cancer[df_cancer['target']==0])  # class- 0 means'Malignant' means(cancer), class- 1 means 'Benign' means(No cancer)
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.countplot(df_cancer['target'], label = "Count") 
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

#sns.lmplot('mean area', 'mean smoothness', hue ='target', data = df_cancer, fit_reg=True)
# Let's check the correlation between the variables 

# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter

plt.figure(figsize=(20,10)) 

sns.heatmap(df_cancer.corr(), annot=True) 


# Let's drop the target label coloumns

X = df_cancer.drop(['target'],axis=1)

X
y = df_cancer['target']

y
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
from sklearn.svm import SVC 

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



svc_model = SVC()

svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

cm = confusion_matrix(y_test, y_predict)

cm
accu_score=accuracy_score(y_test,y_predict)

accu_score
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))
from sklearn.preprocessing import minmax_scale
scl=minmax_scale(X_train)

X_train_scaled=pd.DataFrame(scl)

X_train_scaled.columns=X_train.columns

X_train_scaled
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)
scl=minmax_scale(X_test)

X_test_scaled=pd.DataFrame(scl)

X_test_scaled.columns=X_test.columns

X_test_scaled
from sklearn.svm import SVC 

from sklearn.metrics import classification_report, confusion_matrix



svc_model = SVC()

svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)



sns.heatmap(cm,annot=True,fmt="d")

cm
print(classification_report(y_test,y_predict))
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
cm
print(classification_report(y_test,grid_predictions))