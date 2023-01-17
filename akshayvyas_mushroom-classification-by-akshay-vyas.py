# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/mushrooms.csv')

data.head(5)
data.describe().transpose()
data.shape
data.columns
data.isnull().sum()
data['class'].unique()
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 

data.head()
data['stalk-color-above-ring'].unique()
print(data.groupby('class').size())
data.describe().transpose()
X = data.iloc[:,1:23]  # all rows, all the features and no labels

y = data.iloc[:, 0]  # all rows, label only

X.head()

y.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X
from sklearn.decomposition import PCA

pca = PCA()

pca.fit_transform(X)
covariance = pca.get_covariance()

covariance
explained_variance = pca.explained_variance_

explained_variance
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    plt.bar(range(22), explained_variance, alpha=0.5, align='center', label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    plt.bar(range(22), explained_variance.cumsum(), alpha=0.5, align='center', label='Cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
N=data.values

pca = PCA(n_components=2)

x = pca.fit_transform(N)

plt.figure(figsize = (6,6))

plt.scatter(x[:,0],x[:,1], c='Y')

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=5)

X_clustered = kmeans.fit_predict(N)



LABEL_COLORED_MAP = {

    0: 'b',

    1: 'r'

}

label_color = [LABEL_COLORED_MAP[l] for l in X_clustered]

plt.figure(figsize = (6,6))

plt.scatter(x[:,0],x[:,1], c=label_color)

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
y_prob = model_LR.predict_proba(X_test)[:,1]

y_pred = np.where(y_prob > 0.5, 1, 0)

model_LR.score(X_test, y_pred)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

confusion_matrix
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

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
LR_ridge= LogisticRegression(penalty='l2')

LR_ridge.fit(X_train,y_train)
y_prob = LR_ridge.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

LR_ridge.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='black',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'right')

plt.plot([0, 1], [0, 1],linestyle='-')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)
y_prob = nb_model.predict_proba(X_test)[:,1] #Gives positive class prediction probabilites

y_pred = np.where(y_prob > 0.5, 1, 0) #Thresholds probabilities for predictions

nb_model.score(X_test, y_pred)
print("# of Mislabeled points from %d points: %d" 

      % (X_test.shape[0], (y_test!= y_pred).sum() )

     )
scores = cross_val_score(nb_model, X, y, cv=10, 

                         scoring='accuracy')

print(scores)
scores.mean()
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

confusion_matrix
auc_roc = metrics.classification_report(y_test, y_pred)

auc_roc
auc_roc = metrics.roc_auc_score(y_test, y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

plt.title('Receiver Operating Characteristics')

plt.plot(false_positive_rate, true_positive_rate, color='green',

        label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc='right')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')