# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../input/diabetes-dataset/diabetes2.csv")
df.head()
df.shape
df.dtypes
df.describe()
df['Outcome'].value_counts()
sns.countplot(x='Outcome',data=df,palette='Paired')
plt.show()
plt.savefig('count_plot')
count_nondisease=len(df[df['Outcome']==0])
count_disease=len(df[df['Outcome']==1])
pct_of_nondisease=count_nondisease/(count_nondisease+count_disease)
print("Percentage of non disease is ",pct_of_nondisease*100)
pct_of_disease=count_disease/(count_nondisease+count_disease)
print("Percentage of disease is ",pct_of_disease*100 )
df.isnull().sum()
correlations=df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(data=correlations,square=True,annot = True,cmap="bwr")

plt.yticks(rotation=0)
plt.xticks(rotation=90)
df.corr()
df1=df.drop(['SkinThickness','Insulin','DiabetesPedigreeFunction'],axis=1)
correlations=df1.corr()
plt.figure(figsize=(10,8))
sns.heatmap(data=correlations,square=True,annot = True,cmap="bwr")

plt.yticks(rotation=0)
plt.xticks(rotation=90)
sns.boxplot(x=df1['Pregnancies'])
sns.boxplot(x=df1['Glucose'])
sns.boxplot(x=df1['BloodPressure'])
sns.boxplot(x=df1['BMI'])
sns.boxplot(x=df1['Age'])
filter=df1['Pregnancies'].values<13

df2=df1[filter]

df2.shape
filter=df2['Glucose'].values>20

df2=df2[filter]

df2.shape
filter=df2['BMI'].values>15
df2=df2[filter]

filter=df2['BMI'].values<65
df2=df2[filter]


df2.shape
filter= df2['Age'].values<65
df2=df2[filter]

df2.shape
sns.boxplot(x=df2['Pregnancies'])
sns.boxplot(x=df2['Glucose'])
sns.boxplot(x=df2['BMI'])
sns.boxplot(x=df2['BloodPressure'])
sns.boxplot(x=df2['Age'])
X=df2.drop('Outcome',axis=1)
y=df2['Outcome']

print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_log_pred=logreg.predict(X_test)
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
print('Accuracy Score')
print(metrics.accuracy_score(y_test,y_log_pred))
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_log_pred))
print(classification_report(y_test,y_log_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_log_pred)
