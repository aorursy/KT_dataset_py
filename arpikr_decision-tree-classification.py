pwd
import warnings

warnings.filterwarnings('ignore')

import os

import pandas as pd

from pandas import DataFrame

import pylab as pl

import numpy as np

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

%matplotlib inline
banknote=pd.read_csv("../input/banknote.csv") #Importing Data
banknote.head()
print(banknote.shape)

banknote.info()
pd.options.display.float_format = '{:.4f}'.format

data_summary=banknote.describe()

data_summary.T
for k, v in banknote.items():

    q1 = v.quantile(0.25)

    q3 = v.quantile(0.75)

    irq = q3 - q1

    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]

    perc = np.shape(v_col)[0] * 100.0 / np.shape(banknote)[0]

    print("Column %s outliers = %.2f%%" % (k, perc))
plt.figure(figsize=(12,5))

banknote.boxplot(patch_artist=True,vert=False)
my_corr=banknote.corr()

print(my_corr)

plt.figure(figsize=(12,5))

sns.heatmap(my_corr,linewidth=0.5)

plt.show()
print(banknote['Class'].value_counts())

sns.countplot(x='Class',data=banknote)
sns.pairplot(banknote, hue='Class',kind='reg') 

plt.show()
banknote.columns
predictor_var= banknote[['Variance', 'Skewness', 'Curtosis', 'Entropy']] #all columns except the last one

target_var= banknote['Class'] #only the last column
print(predictor_var.shape)

print(target_var.shape)
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(predictor_var,target_var, test_size=0.3, random_state=123)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3,max_features=4)
tree.fit(X_train, Y_train)
tree.feature_importances_

pd.Series(tree.feature_importances_,index=predictor_var.columns).sort_values(ascending=False)
predictions = tree.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':predictions})

df.head(5)
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
accuracy_score(Y_test, predictions) #Calculate number of correctly classified observations.
accuracy_score(Y_test, predictions, normalize=False)
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
confusion_mat = confusion_matrix(Y_test, predictions)

confusion_df = pd.DataFrame(confusion_mat, index=['Class 0','Class 1'],columns=['Class 0','Class 1'])
print(confusion_df)

_=sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()

print("True Negatives: ", tn)

print("False Positives: ", fp)

print("False Negatives: ", fn)

print("True Positives: ", tp)
print(classification_report(Y_test, predictions))
Specificity = tn/(tn+fp)

print("The probability of predicting whether a bank note is authentic or fake",Specificity)
Sensitivity = tp/(tp+fn)

print("The probability of predicting whether a bank note is authentic or fake correctly is",Sensitivity)
from sklearn.tree import export_graphviz

import graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=predictor_var.columns, out_file=None)
graphviz.Source(dot_data)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
param_grid = [{"max_depth":[3, 4, 5, None], "max_features":[4,5,6,7]}]
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=123),param_grid = param_grid,cv=10)
gs.fit(X_train, Y_train)
gs.cv_results_['params']
gs.best_params_
gs.best_estimator_
tree = DecisionTreeClassifier(max_depth=None,max_features=4)
tree.fit(X_train,Y_train)
predictions = tree.predict(X_test)
df=pd.DataFrame({'Actual':Y_test, 'Predicted':predictions})

df.head(5)
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
accuracy_score(Y_test, predictions)#Calculate number of correctly classified observations.
accuracy_score(Y_test, predictions, normalize=False) 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
confusion_mat = confusion_matrix(Y_test, predictions)

confusion_df = pd.DataFrame(confusion_mat, index=['Class 0','Class 1'],columns=['Class 0','Class 1'])
print(confusion_df)

_=sns.heatmap(confusion_df, cmap='coolwarm', annot=True)
tn, fp, fn, tp = confusion_matrix(Y_test, predictions).ravel()

print("True Negatives: ", tn)

print("False Positives: ", fp)

print("False Negatives: ", fn)

print("True Positives: ", tp)
print(classification_report(Y_test, predictions))
Specificity = tn/(tn+fp) 

print("The probability of predicting whether a bank note is authentic or fake is:",Specificity)
Sensitivity = tp/(tp+fn)

print("The probability of predicting whether a bank note is authentic or fake is:",Sensitivity)
from sklearn.tree import export_graphviz

import graphviz
dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=predictor_var.columns, out_file=None)
graphviz.Source(dot_data)
DT_Classifier=[['Max_Depth',3,'None'],['Max_Feature',4,4],['Accuracy Score',0.93,0.985],['f1 Score for Class0',0.94,0.99],['f1 Score for Class1',0.93,0.98],['Specificity',0.95,0.99],['Sensitivity',0.92,0.98],['Misclassified',26,6]]

Result_Summary2= pd.DataFrame(DT_Classifier, columns = ['Parameters','Without Grid Search','With Grid Search'])

Result_Summary2