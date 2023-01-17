# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt, seaborn as sns

%matplotlib inline
# Reading from CSV

bm0= pd.read_csv("/kaggle/input/bank-marketing.csv")

print("Dataset with rows {} and columns {}".format(bm0.shape[0],bm0.shape[1]))

bm0.head()
bm0.info()
bm0.describe()
bm0.pdays.describe()
bm1=bm0.copy()
bm1.drop(bm1[bm1['pdays'] < 0].index, inplace = True) 
bm1.pdays.describe()
bm1.groupby(['education'])['balance'].median().plot.barh()
bm1.pdays.plot.box()

plt.show()
bm1.response.value_counts(normalize=True)
bm1.replace({'response': {"yes": 1,'no':0}},inplace=True)
bm1.response.value_counts()
# here we are seperating object and numerical data types 

obj_col = []

num_col = []

for col in bm1.columns:

    if bm1[col].dtype=='O':

        obj_col.append(col)

    else:

        num_col.append(col)
print("Object data type features ",obj_col)

print("Numerical data type features ",num_col)
from numpy import median

for col in obj_col[1:]:

    plt.figure(figsize=(8,6))

    sns.violinplot(bm1[col],bm1["response"])

    plt.title("Response vs "+col,fontsize=15)

    plt.xlabel(col,fontsize=10)

    plt.ylabel("Response",fontsize=10)

    plt.show()

#sns.despine()

# violin plots give best of both worlds 

# it gives boxplot and distribution of data like whether the data is skewed or not.

# if normally distributed then it's the best you can get.

# you can also use barplots in this case.
plt.figure(figsize=(8,6))

sns.heatmap(bm1.corr(),annot=True,cmap='RdBu_r')

plt.title("Correlation Of Each Numerical Features")

plt.show()
for col in num_col[:-1]:

    plt.figure(figsize=(10,8))

    sns.jointplot(x = bm1[col],y = bm1["response"],kind='reg')

    plt.xlabel(col,fontsize = 15)

    plt.ylabel("Response",fontsize = 15)

    plt.grid()

    plt.show()
from sklearn.preprocessing import LabelEncoder
bm2 = bm1[obj_col].apply(LabelEncoder().fit_transform)
bm2.head()
bm3 = bm2.join(bm1[num_col])
bm3.head()
bm3.corr()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

np.random.seed(42)
import warnings

warnings.filterwarnings("ignore")
X = bm3.drop("response", axis=1)

X.head()
y= bm3[['response']]

y.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)
lr = LogisticRegression()
lr.fit(X_train,y_train)
cv_score= cross_val_score(lr,X_train,y_train, cv=5)

np.mean(cv_score)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))
confusion_matrix(y_pred,y_test)
f1_score(y_pred,y_test)
from sklearn.feature_selection import RFE

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

rfe = RFE(lr, 5)

rfe.fit(X_train,y_train)
rfe.support_
X_train.columns[rfe.support_]
cols = X_train.columns[rfe.support_]
lr.fit(X_train[cols],y_train)
y_pred2 = lr.predict(X_test[cols])
f1_score(y_pred2,y_test)
confusion_matrix(y_pred2,y_test)
import statsmodels.api as sm
X_train.head()
X_train_sm = sm.add_constant(X_train[cols])

X_train_sm.head()
lr1 = sm.OLS(y_train, X_train_sm).fit()
lr1.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=5, random_state=42,max_leaf_nodes=50)
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
cols = X_train.columns[rfe1.support_]
rfc.fit(X_train[cols],y_train)
y_pred3 = rfc.predict(X_test[cols])
f1_score(y_pred3,y_test)
confusion_matrix(y_pred3,y_test)