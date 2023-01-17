import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import LocalOutlierFactor
df = pd.read_csv('../input/seattleWeather_1948-2017.csv')
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)


imputer = imputer.fit(df.iloc[:,1:])
df.iloc[:,1:]=imputer.transform(df.iloc[:,1:])
df.isnull().sum()
from scipy import stats
df = df[(np.abs(stats.zscore(df.iloc[:,1:])) < 3).all(axis=1)]
sns.countplot(x='RAIN',data=df, palette='hls')
plt.show()
from datetime import datetime
df['DATE'] = pd.to_datetime(df['DATE'])
df['month'] = df['DATE'].map(lambda x: x.strftime("%b"))
sns.factorplot(x="month", y="RAIN", data=df,
                   size=6, kind="bar", palette="muted")
plt.show()
y_act = df.RAIN

df_train = df
df_train.drop(["DATE", "RAIN"], axis = 1 , inplace = True)
df_train = pd.get_dummies(df_train, columns=['month'])
df_train.dtypes
y_act.dtypes             
df_train["PRCP"] = df_train["PRCP"].astype(float)
df_train["TMAX"] = df_train["TMAX"].astype(np.int64)
df_train["TMIN"] = df_train["TMIN"].astype(np.int64)
y_act = y_act.astype(int)
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(df_train, y_act )
print(rfe.support_)
print(rfe.ranking_)
import statsmodels.api as sm
logit_model=sm.Logit(y_act,df_train)
result=logit_model.fit()
print(result.summary())
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_act, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(max_iter=1000)
logreg.fit(df_train, y_act)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
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
