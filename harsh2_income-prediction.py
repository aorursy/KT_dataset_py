import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df= pd.read_csv("../input/adult.csv",na_values='#NAME?')
df.head()
print(df.head())
df.info()
df.income.unique()
df['income'].value_counts()

df['income'] = [0 if x == '<=50K' else 1 for x in df['income'] ]

print(df['income'].value_counts().sort_values(ascending=False).head())
print(df['native-country'].value_counts().sort_values(ascending=False).head(10))

df['native-country']=['United-States' if x == 'United-States' else 'others' for x in df['native-country']]
print(df['native-country'].value_counts().sort_values(ascending=False).head(10))
df.info()
#Assign X as a datafrsme of features and y as a series of the outcome variable
X= df.iloc[:,:-1]
y= df.iloc[:,14]
print(X)
print(y)
#df.info()
df.info()

cat_data= X[['workclass','education','marital-status','occupation','relationship','race','gender','native-country']]
cat_data.head()
X= X.drop(cat_data,1)
X.head()
D_data=pd.get_dummies(cat_data,drop_first=True)
D_data.head()
#Join Dummies data and Original Data
X= pd.concat([X,D_data], axis=1)
X.head()
X.columns

X.describe().transpose()
X.isnull().sum().sort_values(ascending=False)
X.corr()


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25, random_state=0);
print(X.head())
print(y.head())
X_train.info()
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
X_train
X_test


df.columns

df.groupby('age').mean()
df.groupby('workclass').mean()
df.groupby('education').mean()
df.groupby('marital-status').mean()
df.groupby('native-country').mean()

import scipy.stats
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred= classifier.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix
confn_mtrx= confusion_matrix(y_test,y_pred)
print(confn_mtrx)
779   +438
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
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
