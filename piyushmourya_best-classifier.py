import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
!wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv
import pandas as pd

df = pd.read_csv("../input/loan-train-dataset/loan_train.csv")

df.head()
df.shape
df['due_date'] = pd.to_datetime(df['due_date'])

df['effective_date'] = pd.to_datetime(df['effective_date'])

df.head()
df['loan_status'].value_counts()
# notice: installing seaborn might takes a few minutes

!conda install -c anaconda seaborn -y
import seaborn as sns



bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'Principal', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
bins = np.linspace(df.age.min(), df.age.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'age', bins=bins, ec="k")



g.axes[-1].legend()

plt.show()
df['dayofweek'] = df['effective_date'].dt.dayofweek

bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)

g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)

g.map(plt.hist, 'dayofweek', bins=bins, ec="k")

g.axes[-1].legend()

plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

df.head()
df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

df.head()
df["loan_status"].replace(to_replace= ['PAIDOFF','COLLECTION'],value=[1,0],inplace=True)
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head()
Feature = df[['Principal','terms','age','Gender','weekend']]

Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)

Feature.drop(['Master or Above'], axis = 1,inplace=True)

Feature.head()

X = Feature

X[0:5]
y = df['loan_status'].values

y[0:5]
X= preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, random_state=1)

print("Train set :",X_train.shape, y_train.shape)

print("Test set :",X_test.shape,y_test.shape)
y_train.reshape(-1,1)

y_test.reshape(-1,1)
k = 4

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)

neigh

yhat = neigh.predict(X_test)

print(yhat[0:5])

print(y_test[0:5])
yhat.reshape(-1,1)
from sklearn import metrics

print("Train set accuracy: ",metrics.accuracy_score(y_train,neigh.predict(X_train)))

print("Test set accuracy: ",metrics.accuracy_score(y_test,yhat))
Ks= 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfusionMx = [];

for n in range(1,Ks):

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat = neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)

    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc

    

plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1*std_acc, mean_acc +1*std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
print('The best accuracy was with', mean_acc.max(),"with k =", mean_acc.argmax()+1)
from sklearn.tree import DecisionTreeClassifier

dec = DecisionTreeClassifier().fit(X_train,y_train)
yhat1=dec.predict(X_test)

yhat1[0:5]
from sklearn.metrics import f1_score

f1_score(y_test,yhat1,average='weighted')
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test,yhat2)
from sklearn import svm

clf = svm.SVC(kernel='rbf',class_weight='balanced')

clf.fit(X_train,y_train)
yhat2 =clf.predict(X_test)

yhat2[0:10]
from sklearn.metrics import f1_score

f1_score(y_test,yhat2,average='weighted')
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test,yhat2)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)

yhat3 = lr.predict(X_test)

yhat3[0:5]
from sklearn.metrics import f1_score

f1_score(y_test,yhat3,average='weighted')
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test,yhat3)
metrics.accuracy_score(y_test,yhat3)
from sklearn.metrics import log_loss

log_loss(y_test,yhat3)
from sklearn.metrics import jaccard_similarity_score

from sklearn.metrics import f1_score

from sklearn.metrics import log_loss
!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv
test_df = pd.read_csv("../input/loan-test-dataset/loan_test.csv")

test_df.head()
test_df = pd.read_csv("../input/loan-test-dataset/loan_test.csv")

test_df.head()
import pandas as pd

loan_test = pd.read_csv("../input/loan_test.csv")
test_df['effective_date']=pd.to_datetime(test_df['effective_date'])

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

Feature_test = test_df[['Principal','terms','age','Gender','weekend']]

Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)

Feature_test.drop(['Master or Above'], axis = 1,inplace=True)

Feature_test.head()
X_testset=Feature_test

y_testset=pd.get_dummies(test_df['loan_status'])['PAIDOFF'].values

y_testset
y_pred_knn=neigh.predict(X_testset)

y_pred_dt=dec.predict(X_testset)

y_pred_svm=clf.predict(X_testset)

y_pred_lr=lr.predict(X_testset)

y_pred_lr_proba=lr.predict_proba(X_testset)
y_pred_lr

print(f1_score(y_testset,y_pred_knn))

print(f1_score(y_testset,y_pred_dt))

print(f1_score(y_testset,y_pred_svm))

print(f1_score(y_testset,y_pred_lr))
print(f1_score(y_testset,y_pred_lr))
print(jaccard_similarity_score(y_testset,y_pred_knn))

print(jaccard_similarity_score(y_testset,y_pred_dt))

print(jaccard_similarity_score(y_testset,y_pred_svm))

print(jaccard_similarity_score(y_testset,y_pred_lr))
LR_log_loss=log_loss(y_testset,y_pred_lr_proba)

LR_log_loss