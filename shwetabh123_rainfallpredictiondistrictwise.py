# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
%matplotlib inline
df =pd.read_csv('../input/rainfall-in-india/district wise rainfall normal.csv')
df.head()
df.info()
df.describe()
df.columns
df['STATE_UT_NAME'].value_counts()
df_up =df[df.STATE_UT_NAME == 'UTTAR PRADESH']
df_up.head(4)
df_up.plot()
sort_up = df_up.sort_values('ANNUAL')
sort_up.head()
sort_up.head(10).plot(x = 'DISTRICT',y= 'ANNUAL' )
df.head()
df_so_ana = df.sort_values('ANNUAL')
df_so_ana.head()
df_so_ana.head().plot(kind = 'bar')
df_so_ana.tail().plot(kind = 'bar')
df_so_ana.head().plot(x = 'DISTRICT', y='ANNUAL',kind = 'bar')
df_so_ana.tail().plot(x = 'DISTRICT', y='ANNUAL',kind = 'bar')
df_so_ana.head().plot(x = 'STATE_UT_NAME', y='ANNUAL',kind = 'bar')
sort_val_sat = df_so_ana.groupby('STATE_UT_NAME').ANNUAL.mean().sort_values()
sort_val_sat.plot(kind= 'bar')
sort_val_sat.head(10).plot(kind= 'bar')
sort_val_sat.tail(10).plot(kind= 'bar')
sns.FacetGrid(sort_val_sat)
df_so_ana.head()
df_so_ana.columns = ['STATE_UT_NAME','DISTRICT','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL','Jan_Feb','Mar_May','Jun_Sep','Oct_Dec']
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().head(10).plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().tail(10).plot(kind = 'bar')
df_so_ana.columns
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().head(10).plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().tail(10).plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().tail(10).plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().head(10).plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().head(10).plot(kind = 'bar')
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().tail(10).plot(kind = 'bar')
data =pd.read_csv('../input/rainfall-in-india/rainfall in india 1901-2015.csv')
data.head()
data['SUBDIVISION'].value_counts()
data.plot()
data.describe()
data.info()
df1 = data[['SUBDIVISION','YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].copy()
df1.head()
df1['YEAR'].value_counts(sort =False).head()
df1.describe()
corr = df1.corr()
sns.heatmap(corr)
sns.clustermap(corr)
df1[:100].plot(x ='YEAR', y= 'JUN')
#df1[:100].plot(x='YEAR', y='ANNUAL')
df1[:100].plot(x='YEAR', y='JUL')
df1[:100].JUN.plot()
df1[:100].ANNUAL.plot()
df1[:100].JUL.plot()
df1[:100].JUN.plot()
#df[:100].ANNUAL.plot()
df1[:100].JUL.plot()
df1[:100].JUN.plot()
#df[:100].ANNUAL.plot()
df1[:100].JUL.plot()
df1[:100].AUG.plot()
df[:100].JUN.plot()
#df[:100].ANNUAL.plot()
df[:100].JUL.plot()
df[:100].AUG.plot()
corr = df1.corr()
print(corr)
df2 = df1.sort_values(['YEAR','ANNUAL'])
df2.head()
df2[:100].JUN.plot()

df2[:100].JUL.plot()
df2[:100].AUG.plot()
df2

df3 = df2.sort_values('YEAR')
annual_array = []
for element in range(0,4116,116):
    annual_array.append(df2.loc[element,"ANNUAL"])
    print(element,df2.loc[element,"YEAR"])
df2.head(10)
df2['SUBDIVISION'].value_counts()
#Changing department to unique numbers
df2.SUBDIVISION.unique()
dictionary = {}
for c, value in enumerate(df2.SUBDIVISION.unique(), 1):
    #print(c, value)
    dictionary[value] = c
print(dictionary)
df2["SUBDIVISION"] = df2.SUBDIVISION.map(dictionary)
df2.head()
df2.columns
sns.heatmap(df2.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df2.dropna( inplace = True) 
df2.isnull().sum().sum()
df2.replace([np.inf, -np.inf], np.nan, inplace = True)
df2.isnull().sum().sum()
df3.head()
df3 = df2
df3.head()
sns.heatmap(df3.isnull(),yticklabels=False,cbar=False,cmap='viridis')
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.pipeline import Pipeline
df2.head(0)
features = []
for element in df2.head(0):
    features.append(element)
features.remove('ANNUAL')
features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
print(features)
features2 = ['AUG']
# Separating out the features
x  = df2.loc[:, features].values
# Separating out the target
y = df2.loc[:,features2].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.decomposition import PCA 
pca = PCA(n_components=4)
pca.fit(x)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
pca.score(x,y)
pca.score_samples(x)
from sklearn import svm
from sklearn.datasets import make_classification
clf = svm.SVC()
X_train, y_train = make_classification()
X_test, y_test = make_classification()
clf.fit(X_train, y_train)
print(clf.intercept_)
clf.predict(X_test)
clf.score(X_test, y_test, sample_weight=None)
y_score = clf.decision_function(X_test)
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)
print(average_precision)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)
features = []
for element in df3.head(0):
    features.append(element)
features.remove('ANNUAL')
features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
features.remove('JUL')
print(features)
features2 = ['JUL']
# Separating out the features
p  = df2.loc[:, features].values
# Separating out the target
q = df2.loc[:,features2].values
from sklearn.model_selection import train_test_split
P_train, P_test, q_train, q_test = train_test_split(p, q, test_size=0.33, random_state=42)
from sklearn import svm
from sklearn.datasets import make_classification
clf = svm.SVC()
P_train, q_train = make_classification()
P_test, q_test = make_classification()
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)
q_score = clf.decision_function(P_test)
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(q_test, q_score)
print(average_precision)
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)

df4 = df3
df4.head()
features = []
for element in df3.head(0):
    features.append(element)
features.remove('ANNUAL')
features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
features.remove('JUL')
features.remove('JUN')
print(features)
features2 = ['JUL']

# Separating out the features
r  = df2.loc[:, features].values
# Separating out the target
s = df2.loc[:,features2].values
from sklearn.model_selection import train_test_split
R_train, R_test, s_train, s_test = train_test_split(r, s, test_size=0.33, random_state=42)
from sklearn import svm
from sklearn.datasets import make_classification
clf = svm.SVC()
R_train, s_train = make_classification()
R_test, s_test = make_classification()
clf.fit(R_train, s_train)
clf.predict(R_test)
clf.score(R_test, s_test)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(R_train, s_train)
clf.predict(R_test)
clf.score(R_test, s_test)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(R_train, s_train)
clf.predict(R_test)
clf.score(R_test, s_test)
ax=plt.figure(figsize=(30,20))
ax=sns.countplot(x="SUBDIVISION",palette="inferno",data=data)
ax.set_xlabel("Classes")
ax.set_ylabel("Count")
ax.set_title("Subdivision count")
