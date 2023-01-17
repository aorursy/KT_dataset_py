import pandas as pd

import numpy as np

from io import StringIO

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
csv_data = '''A,B,C,D

   1.0,2.0,3.0,4.0

   5.0,6.0,,8.0

   10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df.isnull().sum()
df.values
df.dropna()

#deleting rows with missing values
df.dropna(axis=1)
# only drop rows where all columns are NaN

df.dropna(how='all')

   # drop rows that have not at least 4 non-NaN values

df.dropna(thresh=4)

   # only drop rows where NaN appear in specific columns (here: 'C')

df.dropna(subset=['C'])
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

#Other options for the strategy parameter are median or most_frequent

imr.fit(df)

imurated_data = imr.transform(df.values)

imurated_data
df = pd.DataFrame([

    ['green', 'M', 10.1, 'class1'],

    ['red', 'L', 13.5, 'class2'],

    ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

df
# size_mapping = {

#     'XL':3,

#     'M':2,

#     'L':1

# }

# df['size'] = df['size'].map(size_mapping)

size_mapping = {

    'XL': 3,

    'L': 2,

    'M': 1}

df['size'] = df['size'].map(size_mapping)

df
#Encoding Class Labels

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}

class_mapping

df['classlabel'] = df['classlabel'].map(class_mapping)
#for inversing

inv_class_mapping = {v:k for k,v in class_mapping.items()}

df['classlabel'] = df['classlabel'].map(inv_class_mapping)

#we can use label encoder class from scikit learn to achieve the same

class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)
class_le.inverse_transform(y)
X = df[['color','size','price']].values

X[:,0] = class_le.fit_transform(X[:,0])

X
ohe = OneHotEncoder(categorical_features=[0], sparse = False)

ohe.fit_transform(X)
#ohe.fit_transform(df).toarray()

#df['classlabel'] = df['classlabel'].map(y)

#df['color'] = class_le.fit_transform(df['color'])
pd.get_dummies(df[['color','size','price']])
df_wine = pd.read_csv("../input/winedata/wine.csv")
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']

df_wine.columns
np.unique(df_wine['Class label'])
X,y = df_wine.iloc[:,1:], df_wine.iloc[:,0]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)

X_test_norm = mms.transform(X_test)
stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)

X_train_std
#lets try the L1 regularization with LogistiRegression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty = 'l1', C=0.1)

lr.fit(X_train_std, y_train)

print(lr.score(X_train_std,y_train))

print(lr.score(X_test_std,y_test))
lr.intercept_

#Since we the  t the LogisticRegression object on a multiclass dataset, it uses the One-vs-Rest (OvR) approach by default where the  rst intercept belongs to the model that  ts class 1 versus class 2 and 3; the second value is the intercept of the model that  ts class 2 versus class 1 and 3; and the third value is the intercept of the model that  ts class 3 versus class 1 and 2, respectively:
lr.coef_

#The weight array that we accessed via the lr.coef_ attribute contains three rows of weight coef cients, one weight vector for each class. Each row consists of 13 weights where each weight is multiplied by the respective feature in the 13-dimensional Wine dataset to calculate the net input
import matplotlib.pyplot as plt

fig = plt.figure()

ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights,param = [], []

for c in np.arange(-4,6,dtype=float):

    #print(c)

#     po = 10**c

#     print(po)

    lr = LogisticRegression(penalty='l1', C=10**c,random_state=0)

    lr.fit(X_train_std,y_train)

    weights.append(lr.coef_[1])

    param.append(10**c)

weights = np.array(weights)

for column,color in zip(range(weights.shape[1]),colors):

    plt.plot(param,weights[:,column], label=df_wine.columns[column+1],color=color)



plt.axhline(0, color='black', linestyle='--', linewidth=3)

plt.xlim([10**(-5), 10**5])

plt.ylabel('weight coefficient')

plt.xlabel('C')

plt.xscale('log')

plt.legend(loc='upper left')

ax.legend(loc='upper center',bbox_to_anchor=(1.38, 1.03),ncol=1, fancybox=True)

plt.show()

    
from sklearn.base import clone

from itertools import combinations

import numpy as np

#from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
class SBS():

    

    def __init__(self, estimator, k_features,

        scoring=accuracy_score,

        test_size=0.25, random_state=1):

        self.scoring = scoring

        self.estimator = clone(estimator)

        self.k_features = k_features

        self.test_size = test_size

        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        dim = X_train.shape[1]

        self.indices_ = tuple(range(dim))

        self.subsets_ = [self.indices_]

        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        self.scores_ = [score]

        while dim > self.k_features:

            scores = []

            subsets = []

            

            for p in combinations(self.indices_, r=dim-1):

                

                score = self._calc_score(X_train, y_train, X_test, y_test, p)

                scores.append(score)

                subsets.append(p)

            best = np.argmax(scores)

            self.indices_ = subsets[best]

            self.subsets_.append(self.indices_)

            dim -= 1

            self.scores_.append(scores[best])

            self.k_score_ = self.scores_[-1]

            return self

    def transform(self, X):

        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):

        

        self.estimator.fit(X_train[:, indices], y_train)

        y_pred = self.estimator.predict(X_test[:, indices])

        score = self.scoring(y_test, y_pred)

        return score
#Now, let's see our SBS implementation in action using the KNN classi er from scikit-learn:

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors = 2)

sbs = SBS(knn, k_features=1)

sbs.fit(X_train_std,y_train)
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_,marker='o')

plt.ylim([0.7, 1.1])

plt.grid()

plt.show()
#k5 = list(sbs.subsets_[8])

print(sbs.subsets_)
from sklearn.ensemble import RandomForestClassifier

feat_label = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators = 1000, random_state = 0, n_jobs=-1)

forest.fit(X_train,y_train) #in the random forest we dont need to standardize the data

importances = forest.feature_importances_

indeces = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):

    print(feat_label[indeces[f]], importances[indeces[f]])
plt.title('Feature Importance')

plt.bar(range(X_train.shape[1]),importances[indeces],color='blue',align='center')

plt.xticks(range(X_train.shape[1]), feat_label[indeces], rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.tight_layout()

plt.show()