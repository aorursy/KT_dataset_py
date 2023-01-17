import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(r'/kaggle/input/framingham-heart-study-dataset/framingham.csv')
df.dropna(inplace=True)
df.head(5)


df0 = df[df['TenYearCHD']==0]
df1 = df[df['TenYearCHD']==1]
print('Count of TenYearCHD is 0: ',df0.shape[0])
print('Count of TenYearCHD is 1: ',df1.shape[0])

df0 = df0.sample(n=557)
df = pd.concat([df0,df1])
print('Final df shape: ',df.shape[0])

corrmat = df.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 
X = df[['age','prevalentHyp','sysBP','glucose']]
y = df[['TenYearCHD']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
sklearn.metrics.accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
conf_matrix = pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
from sklearn.metrics import roc_curve, auc
y_pred_prob_yes = model.predict_proba(X_test)

fpr,tpr,threshold = roc_curve(y_test, y_pred_prob_yes[:,1])
roc_auc = auc(fpr,tpr)
print('AUC: ', roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='r',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
