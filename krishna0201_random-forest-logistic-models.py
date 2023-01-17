#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/bankingmarket/bank.csv')   #reading dataset
df.head()
df.describe()
df.shape
#seperating numerical and categorical



numerical_features= ['age','salary','balance','day','duration','campaign','pdays','previous','response','poutcome'];

categorical_features= ['job','marital','education','targeted','default','housing'];
df['poutcome'].value_counts()


helo=df.replace({'poutcome': {"unknown": 0,'failure':1,'other':2,'success':3}},inplace=True)
pd.concat([df,helo],axis=1)
df.shape
df.info()
#pdays describe made 



df['pdays'].describe()
#Plot a horizontal bar graph with the median values of balance for each education level value

df.groupby(['education'])['balance'].median()
df.groupby(['education'])['balance'].median().plot.barh(color="lightgreen")

plt.show()
plt.figure(figsize=(30,20))

data2 = df[['pdays']]

plt.boxplot(data2.values, labels=['outliers count']);
#handling the pdays column with a value of -1  and handling outliers.



upper_lim = df['pdays'].quantile(.95)

lower_lim = df['pdays'].quantile(.05)



df= df[(df['pdays'] < upper_lim) & (df['pdays'] > lower_lim)]
plt.figure(figsize=(30,20))

data2 = df[['pdays']]

plt.boxplot(data2.values, labels=['outliers count']);
df.corr()
df['response'].unique()
#Quality correlation matrix

k = 19 #number of variables for heatmap

cols = df.corr().nlargest(k, 'balance')['balance'].index

heat = df[cols].corr()

plt.figure(figsize=(20,10))

sns.heatmap(heat, annot=True, cmap = 'viridis')

plt.show()
from sklearn.preprocessing import OneHotEncoder
df.replace({'response': {"yes": 1,'no':0}},inplace=True)
#performing bi-variate analysis to identify the features

from numpy import median



for x in categorical_features[1:]:

    plt.figure(figsize=(8,6))

    sns.barplot(df[x],df["response"])

    plt.title("Response vs "+x,fontsize=15)

    plt.xlabel(x,fontsize=10)

    plt.ylabel("Response",fontsize=10)

    plt.show()
sns.countplot(df['response'])

plt.show()
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df.job = le.fit_transform(df.job)

df.marital = le.fit_transform(df.marital)

df.education = le.fit_transform(df.education)

df.default = le.fit_transform(df.default)

df.housing = le.fit_transform(df.housing)

df.loan = le.fit_transform(df.loan)

df.contact = le.fit_transform(df.contact)

df.poutcome = le.fit_transform(df.poutcome)

df.month=le.fit_transform(df.month)

df.targeted= le.fit_transform(df.targeted)

df.head()
plt.figure(figsize=(15,12))

sns.heatmap(df.corr(),annot=True,cmap='RdBu_r')

plt.title("Correlation")

plt.show()
df.corr()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

np.random.seed(42)
import warnings

warnings.filterwarnings("ignore")
x = df.drop("response", axis=1)
y= df[['response']]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
log_Reg = LogisticRegression()
log_Reg.fit(X_train,y_train)
cv_score= cross_val_score(log_Reg,X_train,y_train, cv=5)

np.mean(cv_score)
y_pred = log_Reg.predict(X_test)
y_pred
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred)
print(classification_report(y_test, y_pred))
from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

scaler = MinMaxScaler()

rfe = RFE(log_Reg, 5)

rfe.fit(X_train,y_train)
rfe.support_
X_train.columns[rfe.support_]
print(rfe.ranking_)
print(X_train.columns[rfe.support_])
tab = X_train.columns[rfe.support_]
log_Reg.fit(X_train[tab],y_train)
y_pred2 = log_Reg.predict(X_test[tab])
f1_score(y_pred2,y_test)
confusion_matrix(y_pred2,y_test)
X_train.head()

X_train_inter = sm.add_constant(X_train[tab])

X_train_inter.head()
logreg = sm.OLS(y_train, X_train_inter).fit()
logreg.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=5,max_leaf_nodes=50)
rfc.fit(X_train,y_train)
cv1_score= cross_val_score(rfc,X_train,y_train, cv=5)

np.mean(cv1_score)
y_pred1 = rfc.predict(X_test)
print(classification_report(y_test, y_pred1))
f1_score(y_test,y_pred1)
confusion_matrix(y_test,y_pred1)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred1)
from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

rfe1 = RFE(rfc, 5)

rfe1.fit(X_train,y_train)
rfe1.support_
X_train.columns[rfe1.support_]
tab = X_train.columns[rfe1.support_]
rfc.fit(X_train[tab],y_train)
y_pred3 = rfc.predict(X_test[tab])
f1_score(y_pred3,y_test)
confusion_matrix(y_pred3,y_test)