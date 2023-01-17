# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier, plot_importance

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/xAPI-Edu-Data.csv')

df=pd.read_csv('../input/xAPI-Edu-Data.csv')
pd.crosstab(df['Class'],df['Topic'])
sns.countplot(x='Topic',hue='Class',data=df1,palette="muted")
df1=pd.read_csv('../input/xAPI-Edu-Data.csv')

df.head(4)
df.columns
sns.countplot(x='gender',data=df,hue='NationalITy')
sns.countplot(x='gender',data=df,palette="muted")
sns.countplot(x="Topic", data=df, palette="muted");
df.head(4)
sns.pairplot(data=df)
df.head(3)
columns=df.dtypes[df.dtypes=='object'].index

columns
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

for col in df.columns:

    df[col]=encoder.fit_transform(df[col])

    



df.head(3) 
df.head(3)
##Co-relation

corr=df.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')

sns.regplot(x='Topic',y='Class',data=df)
Y=df['Class']

df=df.drop(['Class'],axis=1)

X=df
X.head(3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, 

random_state = 30)

#PCA

#Principle component analysis

from sklearn.decomposition import PCA

pca = PCA()

pa=pca.fit_transform(X)

pa
covariance=pca.get_covariance()

explained_variance=pca.explained_variance_

explained_variance
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    

    plt.bar(range(16), explained_variance, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df1)

Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df1, color=".15")

plt.show()
ax = sns.boxplot(x="Class", y="Discussion", data=df1)

ax = sns.swarmplot(x="Class", y="Discussion", data=df1, color=".25")

plt.show()
Anounce_bp = sns.boxplot(x="Class", y="AnnouncementsView", data=df1)

Anounce_bp = sns.swarmplot(x="Class", y="AnnouncementsView", data=df1, color=".25")

plt.show() 
X_train.head(3)
from sklearn.preprocessing import scale



X_train=scale(X_train)

X_test=scale(X_test)

from sklearn.linear_model import Perceptron



ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

ppn.fit(X_train, Y_train)

y_pred = ppn.predict(X_test)

print('Misclassified samples: %d' % (Y_test != y_pred).sum())
from sklearn.metrics import accuracy_score

print('Accuracy: %.2f' % accuracy_score(Y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred))
from sklearn.svm import SVC



svm = SVC(kernel='linear', C=2.0, random_state=0)

svm.fit(X_train, Y_train)



y_pred_SVM = svm.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(Y_test, y_pred_SVM))

print('Misclassified samples: %d' % (Y_test != y_pred_SVM).sum())
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

#clf = MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=1)

clf = MLPClassifier(solver='lbfgs',alpha=.1,random_state=1)

clf.fit(X_train, Y_train)

scores=cross_val_score(clf,X_test,Y_test,cv=10)
clf.score(X_test,Y_test)
RF = RandomForestClassifier(n_jobs = -1)

RF.fit(X_train, Y_train)

Y_pred = RF.predict(X_test)

RF.score(X_test,Y_test)

print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)

xgb.fit(X_train, Y_train)

xgb.predict(X_test)

print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
xgb_pred=xgb.predict(X_test)
print (classification_report(Y_test,xgb_pred))
plot_importance(xgb)