# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import statistics
from statistics import mode
data = pd.read_csv("../input/adult-income-dataset/adult.csv")
data.head()
data.shape
data.info()
data.isnull().sum()
new_data = data.copy()
new_data.columns
new_data['age'].value_counts()
data.describe()
from numpy import nan

new_data = new_data.replace("?", nan)
new_data.isnull().sum()
new_data['workclass'].value_counts()
sns.set(style='whitegrid')

plt.figure(figsize=(10,10))

sns.countplot(x='workclass',data=data)
new_data['occupation'].value_counts()
plt.figure(figsize=(15,15))

sns.countplot(x='occupation', data=new_data)
new_data['native-country'].value_counts()

plt.figure(figsize=(15,15))

sns.countplot(x='native-country', data=new_data)
missing_data = new_data.isnull().sum()

total_data = np.product(new_data.shape)

percent_missing_data = (missing_data/total_data)*100

percent_missing_data
new_data['workclass'].fillna(new_data['workclass'].mode()[0], inplace=True)

new_data['occupation'].fillna(new_data['occupation'].mode()[0], inplace=True)

new_data['native-country'].fillna(new_data['native-country'].mode()[0], inplace=True)
new_data['age'].hist(figsize=(8,8))
new_data['fnlwgt'].hist(figsize=(8,8))
new_data.skew()
for i in new_data.columns:

    new_data[i].hist(figsize=(10,10))
plt.figure(figsize=(12,8))



total = float(len(new_data["income"]) )



ax = sns.countplot(x="workclass", data=new_data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(8,8))

total = float(len(new_data) )



ax = sns.countplot(x="gender", data=new_data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
plt.figure(figsize=(7,7))

total = float(len(new_data) )



ax = sns.countplot(x="income", data=new_data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.show()
pd.crosstab(new_data['workclass'], new_data['income'])
z=pd.crosstab(new_data['age'], new_data['income'])
from scipy.stats import chi2_contingency

from scipy.stats import chi2





stat, p, dof, expected = chi2_contingency(z)

print('dof=%d' % dof)

print('p_value', p)

print(expected)



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
sns.pairplot(new_data)
cor=new_data.corr()

cor
sns.heatmap(cor, annot=True)
from sklearn.model_selection import train_test_split
new_data['income']=[1 if each=='>50k' else 0 for each in new_data.income]

y = new_data['income']
y.head()
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

new_data = new_data.apply(LabelEncoder().fit_transform)

new_data.head()
cols = ['native-country', 'education', 'income','hours-per-week']

y = new_data["income"]

x = new_data.drop(cols, axis=1)
x.columns
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
scale=StandardScaler()

x_train = scale.fit_transform(x_train)

x_test=scale.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

score_list=[]

score_list2=[]

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    score_list.append(knn.score(x_test,y_test))

plt.plot(range(1,20),score_list)

plt.show()
# K-Nearest Neighbors

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

#y_pred = knn.predict(X_test)

score_knn = knn.score(x_test,y_test)

print('The accuracy of the KNN Model is',score_knn)