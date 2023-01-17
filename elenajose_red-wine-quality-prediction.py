#alias importing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
red_wine = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
red_wine.head()
red_wine.tail()
red_wine.info()
red_wine.quality.head()
red_wine.quality.value_counts()
red_wine.corr()['quality'].sort_values(ascending=False).drop('quality')
red_wine.corr()['quality'].drop('quality').plot(kind='bar',color='magenta', title='Graph 1.1 - Pearson Correlation');
sns.heatmap(red_wine.corr())

plt.title('Graph 1.2 - Heatmap of Correlation');
red_wine.isnull().sum()
#Binary classification of the target variable into 'good' and 'bad'

bins=(2,6.9,8)

group_names=['bad','good']

red_wine['quality']=pd.cut(red_wine['quality'],bins=bins,labels=group_names)
red_wine.quality.head()
from sklearn.preprocessing import LabelEncoder
label_qual = LabelEncoder()
#Assigning labels - Bad becomes 0 and good becomes 1 

red_wine['quality'] = label_qual.fit_transform(red_wine['quality'])
red_wine.head()
red_wine['quality'].value_counts()
#Plotting

plt.style.use('fivethirtyeight')

red_wine['quality'].value_counts().plot(kind='bar', title='Graph 1.3 - Count of good and bad quality wine');
red_wine.columns
feature_columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol']
#response variables and target variable

X=red_wine[feature_columns]

y=red_wine.quality
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=13)
from sklearn.linear_model import LogisticRegression
redwinelogit=LogisticRegression()
redwinelogit.fit(X_train,y_train)
redwinelogit.score(X_train,y_train)
redwinelogit.score(X_test,y_test)
redwinelogit.predict(X_test)
print(redwinelogit.coef_)

print(redwinelogit.intercept_)
feature_columns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
confusion_matrix(y_test,redwinelogit.predict(X_test))
print(classification_report(y_test,redwinelogit.predict(X_test)))
precision, recall, _ = precision_recall_curve(y_test,redwinelogit.predict(X_test))

plt.plot(recall,precision)

plt.xlabel('Recall')

plt.ylabel("Precision")

plt.title("Graph 1.4 - Precision Recall Curve");
# plot_roc_curve throwing an import error. Referred this method from a Medium article. Link given in sources section at the end.

from sklearn.metrics import roc_auc_score, roc_curve
logit_roc_auc = roc_auc_score(y_test, redwinelogit.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, redwinelogit.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r-')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Graph 1.5 - Receiver Operating Characteristic')

plt.legend(loc="lower right");