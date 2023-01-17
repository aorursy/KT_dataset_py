import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC 

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential

from keras.layers import Dropout

from keras.layers import Dense

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Reading dataset 

df=pd.read_csv('../input/heart.csv')
# Checking first few entries of the dataset

df.head()
# looking into the summary of the dataset such as mean, standard deviation minimum and maximum values of the attributes

df.describe()
df.isna().sum()
# checking the number of observation i.e., number of rows and columns/features

df.shape
df.dtypes
# Checking the number of disease and healthy observations 

sns.countplot(df['target'], label = "Count") 
plt.figure(figsize=(20,10)) 

sns.heatmap(df.corr(), annot=True)
X=df.drop(['target'],axis=1)

X.corrwith(df['target']).plot.bar(

        figsize = (20, 10), title = "Correlation with Target", fontsize = 20,

        rot = 90, grid = True)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10,stratify=y, random_state=5)
min_train = X_train.min()

range_train = (X_train - min_train).max()

X_train_scaled = (X_train - min_train)/range_train
min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
svc_model = SVC(gamma='auto')

svc_model.fit(X_train_scaled, y_train)
y_predict = svc_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,y_predict))
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear']} 



grid = GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=4,cv=5,scoring='neg_log_loss')

grid.fit(X_train_scaled,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True)
print(classification_report(y_test,grid_predictions))
from sklearn import metrics

y_pred=grid.predict(X_test_scaled) # predict the test data

# Compute False postive rate, and True positive rate

fpr, tpr, thresholds = metrics.roc_curve(y_test, grid.predict_proba(X_test_scaled)[:,1])

# Calculate Area under the curve to display on the plot

auc = metrics.roc_auc_score(y_test,grid.predict(X_test_scaled))

# Now, plot the computed values

plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (SVC, auc))

# Custom settings for the plot 

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('1-Specificity(False Positive Rate)')

plt.ylabel('Sensitivity(True Positive Rate)')

plt.title('Receiver Operating Characteristic')

plt.legend(loc="lower right")

plt.show()