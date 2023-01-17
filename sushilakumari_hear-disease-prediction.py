# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams

from matplotlib.cm import rainbow

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import pickle
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
import os

os.environ['KAGGLE_USERNAME'] = "sushilakumari" # username from the json file

os.environ['KAGGLE_KEY'] = "7917b5aaaaee4968aed8788fb4acb514" # key from the json file

!kaggle datasets download -d ronitf/heart-disease-uci
from zipfile import ZipFile

# Create a ZipFile Object and load sample.zip in it

with ZipFile('heart-disease-uci.zip', 'r') as zipObj:

   # Extract all the contents of zip file in current directory

   zipObj.extractall()
df = pd.read_csv('heart.csv')
df.info()
df.shape
df.head()
df.isna().sum()
df.apply(lambda x:len(x.unique()))


df.describe()
x_train = df.drop(['target'], axis=1)

y_train = df['target']
clf = LogisticRegression()

#cross_val_score(clf, x_train, y_train, cv=5)
clf.fit(x_train, y_train)
clf.predict(x_train)
with open('model.pkl','wb') as f:

    pickle.dump(clf,f)


# load

with open('model.pkl', 'rb') as f:

    clf2 = pickle.load(f)
x_train[:1]
clf2.predict(x_train[:1])


def get_preds(*args):

    vals = [[*args]]

    print(vals)

    return clf2.predict(vals)[0]
get_preds(63, 1, 3, 145, 233, 1, 0, 150, 0,2.3,0,0,1)
from sklearn.externals import joblib

# Save to file in the current working directory

joblib_file = "joblib_model.pkl"

joblib.dump(clf, joblib_file)



# Load from file

joblib_model = joblib.load(joblib_file)
joblib_model.predict(x_train[:1])
import seaborn as sns

#get correlations of each features in dataset

corrmat = df.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.hist()
sns.set_style('whitegrid')

sns.countplot(x='target',data=df,palette='RdBu_r')
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
dataset.head()


y = dataset['target']

X = dataset.drop(['target'], axis = 1)
from sklearn.model_selection import cross_val_score

knn_scores = []

for k in range(1,21):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    score=cross_val_score(knn_classifier,X,y,cv=10)

    knn_scores.append(score.mean())
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')

for i in range(1,21):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 21)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')


knn_classifier = KNeighborsClassifier(n_neighbors = 12)

score=cross_val_score(knn_classifier,X,y,cv=10)

score.mean()
from sklearn.ensemble import RandomForestClassifier



randomforest_classifier= RandomForestClassifier(n_estimators=10)



score=cross_val_score(randomforest_classifier,X,y,cv=10)

score.mean()