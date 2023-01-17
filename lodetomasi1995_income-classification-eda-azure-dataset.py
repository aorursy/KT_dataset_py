import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(os.listdir("../input"))



df = pd.read_csv('../input/income_evaluation.csv')

df.head(10)



# Any results you write to the current directory are saved as output.
df[' education'].value_counts().head(10).plot.bar()
df[' sex'].value_counts().head(10).plot.bar()
sns.countplot(x=" income", data=df, palette="bwr")

plt.show()
pd.crosstab(df[' sex'],df[' income']).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])

plt.title('Income  for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.ylabel('Frequency')

plt.show()
pd.crosstab(df[' education'],df[' income']).plot(kind="bar",figsize=(20,6),color=['#FFC300','#581845' ])

plt.title('Income  for education')

plt.xticks(rotation=0)

plt.ylabel('Frequency')

plt.show()
df_treat = pd.get_dummies(df)

df_treat.dtypes.value_counts()

df_treat = df_treat.rename(columns=({' income_ <=50K':'minusEqual50',' income_ >50K':'Plus50'}))

df_treat.head(10)
temp= [df_treat['minusEqual50'],df_treat['Plus50']]

y= pd.DataFrame(temp)

y= y.transpose()

x = df_treat.drop(columns=['minusEqual50','Plus50'])



from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



model_LR= LogisticRegression()
model_LR.fit(x_train,y_train['minusEqual50'])
y_prob = model_LR.predict_proba(x_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_LR.score(x_test, y_pred)





from sklearn.model_selection import cross_val_score

from sklearn import metrics



confusion_matrix=metrics.confusion_matrix(y_test['minusEqual50'],y_pred)

confusion_matrix
auc_roc=metrics.roc_auc_score(y_test['minusEqual50'],y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test['minusEqual50'], y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

confusion_matrix=metrics.confusion_matrix(y_test['minusEqual50'],y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(confusion_matrix,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier()

mlp.fit(x_train,y_train['minusEqual50'])
y_prob = mlp.predict_proba(x_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

mlp.score(x_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test['minusEqual50'],y_pred)

confusion_matrix

auc_roc=metrics.classification_report(y_test['minusEqual50'],y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test['minusEqual50'],y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test['minusEqual50'], y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.metrics import confusion_matrix

confusion_matrix=metrics.confusion_matrix(y_test['minusEqual50'],y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(confusion_matrix,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()