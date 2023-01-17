import pandas as pd
diabetes = pd.read_csv('../input/pima-indians-diabetes-database//diabetes.csv')
diabetes.info()
import seaborn as sns
%matplotlib inline

sns.countplot(x='Outcome', data=diabetes, palette='hls')
diabetes.groupby('Outcome').mean()
import numpy as np
from sklearn import linear_model, datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
diabetes.isnull().sum()
sns.boxplot(x='Outcome', y='Glucose', data=diabetes, palette='hls')
sns.heatmap(diabetes.corr())
X_train, X_test, y_train, y_test = train_test_split(diabetes.drop('Outcome', 1), diabetes['Outcome'], test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred_quant = LogReg.predict_proba(X_test)[:, 1] #Only keep the first column, which is the 'pos' values
y_pred_bin = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_bin)
confusion_matrix
from sklearn.metrics import classification_report

total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])
print('Specificity : ', specificity)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_quant)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
metrics.auc(fpr, tpr)