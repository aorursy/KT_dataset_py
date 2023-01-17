import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

%matplotlib inline
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.head()
df.shape
all(df.isnull().any())
df['Class'].value_counts()
print((492/(284807+492))*100)
plt.figure(dpi=100)

sns.set_style('darkgrid')

sns.countplot('Class',data=df)

plt.xlabel('Target Class')

plt.ylabel('Count')

plt.xticks([0,1],['Not Fraud','Fraud'])

plt.show()
mask = np.triu(np.ones_like(df.corr(),dtype=bool))

plt.figure(dpi=100,figsize=(10,8))

sns.heatmap(df.corr(),yticklabels=True,mask=mask,cmap='viridis',annot=False, lw=1)

plt.show()
x=df.iloc[:,:-1]

y=df.iloc[:,-1]
print((x.shape,y.shape))
from imblearn.combine import SMOTETomek

smk=SMOTETomek(ratio=1,random_state=0)

x_new,y_new=smk.fit_sample(x,y)

print(x_new.shape,y_new.shape)


from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x_new,y_new,test_size=0.80,random_state=0,stratify=y_new)


print(x_train.shape,x_test.shape)
from sklearn.linear_model import LogisticRegression

lrm=LogisticRegression(C=0.1,penalty='l1',n_jobs=-1)

lrm.fit(x_train,y_train)
y_pred=lrm.predict(x_test)
print("Train Set Accuracy is ==> ",metrics.accuracy_score(y_train,lrm.predict(x_train)))

print("Test Set Accuracy is ==> ",metrics.accuracy_score(y_test,y_pred))
print("Classification Report on Hold Out Dataset==>\n\n",metrics.classification_report(y_test,y_pred))
probs = lrm.predict_proba(x_test)

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc = metrics.auc(fpr, tpr)

plt.figure(dpi=100)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()