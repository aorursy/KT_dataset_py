import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
%matplotlib inline
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv'
df = pd.read_csv(url)
df.head()
df.info ()
df['custcat'].value_counts().plot.bar ()
plt.figure (figsize = (20,4))
plt.subplot (1,5,1)
sns.distplot (df ['income'], hist = False)

plt.subplot (1,5,2)
sns.distplot (df ['tenure'], hist = False)

plt.subplot (1,5,3)
sns.distplot (df ['age'], hist = False)

plt.subplot (1,5,4)
sns.distplot (df ['employ'], hist = False)

plt.subplot (1,5,5)
sns.distplot (df ['address'], hist = False)

plt.tight_layout ()
plt.figure (figsize = (24,4))

plt.subplot (1,6,1)
sns.countplot (df ['ed'])

plt.subplot (1,6,2)
sns.countplot (df ['region'])

plt.subplot (1,6,3)
sns.countplot (df ['retire'])

plt.subplot (1,6,4)
sns.countplot (df ['reside'])

plt.subplot (1,6,5)
sns.countplot (df ['marital'])

plt.subplot (1,6,6)
sns.countplot (df ['gender'])

plt.tight_layout ()
df.tail ()
X1 = df[['tenure','age', 'address', 'income', 'employ', 'ed', 'retire']].values
X1[0:5]

y = df['custcat'].values
y[0:5]
pipe = Pipeline ([('Poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=True, order='F')), ('Scaler',StandardScaler())])
X = pipe.fit_transform (X1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
#Train Model and Predict  
Score = {}
for k in range (1,50):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
   
    Score.update ({k :  metrics.accuracy_score(y_test, neigh.predict (X_test))})
Score
pd.DataFrame (Score, index = [0]).transpose ().plot ()
neigh = KNeighborsClassifier(n_neighbors = 47).fit(X_train,y_train)
yhat = neigh.predict(X_test)
yhat[0:5]
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print (confusion_matrix (yhat, y_test))
print (classification_report (yhat, y_test))
Ks = 40
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 