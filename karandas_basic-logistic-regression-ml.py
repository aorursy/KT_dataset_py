import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/us-heart-patients/US_Heart_Patients.csv')
df=df.sample(frac=1, random_state=3)
df
df['TenYearCHD'].value_counts()
df['TenYearCHD'].value_counts().plot.bar()
plt.show()
df['TenYearCHD'].value_counts(normalize=True).plot.bar()
plt.show()
df.isnull().sum()
df['glucose'].describe()
df = df.fillna(method='ffill')
y=df['TenYearCHD']
X=df.drop('TenYearCHD', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 3)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression(fit_intercept= True, solver= 'liblinear')

lr.fit(X_train, y_train)
y_train_pred= lr.predict(X_train)
y_train_prob= lr.predict_proba(X_train)[:,1]

print('Confusion Matrix: ', '\n', confusion_matrix(y_train, y_train_pred))
print('Overall Accuracy -Train: ', accuracy_score(y_train, y_train_pred))
print('AUC-Train:',roc_auc_score(y_train, y_train_prob))
lr.fit(X_train, y_train)
y_test_pred= lr.predict(X_test)
y_test_prob= lr.predict_proba(X_test)[:,1]

print('Confusion Matrix: -Test', '\n', confusion_matrix(y_test, y_test_pred))
print('Overall Accuracy -Test: ', accuracy_score(y_test, y_test_pred))
print('AUC-Test:',roc_auc_score(y_test, y_test_prob))
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, 'r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Analysis')
plt.show()