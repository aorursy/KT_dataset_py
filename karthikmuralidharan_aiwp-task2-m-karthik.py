import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline





import os

print(os.listdir("../input"))



data = pd.read_csv("../input/adultcensus/income_data.csv")
print(len(data))

data.head(10)
data.isnull().sum()
data.dtypes
sns.countplot(data['income'])

plt.show()
# Sex distribution

sns.countplot(data['sex'])

plt.show()
# Age distribution

ages = data['age'].hist(bins=max(data['age'])-min(data['age']))

mean_val = np.mean(data['age'])

plt.axvline(mean_val, linestyle='dashed', linewidth=2, color='yellow', label='mean age')

plt.xlabel('age')

plt.ylabel('count')

plt.legend()

plt.show()
data['hours.per.week'].hist()

plt.xlabel('hours per week')

plt.ylabel('count')

plt.show()
plt.figure(figsize=(24, 6))

ro = sns.countplot(data['occupation'], hue=data['sex'])

ro.set_xticklabels(ro.get_xticklabels(), rotation=30, ha="right")

plt.show()
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
plt.figure(figsize=(20, 6))

sns.countplot(data['marital.status'], hue=data['income'])

plt.show()
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0}) 
data['race'] = data['race'].map({'White': 1, 'Asian-Pac-Islander': 1, 'Black':0, 'Amer-Indian-Eskimo':0, 'Other':0}) 

data['relationship'] = data['relationship'].map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})

data['marital.status'] = data['marital.status'].map({'Widowed':0, 'Divorced':0, 'Separated':0, 'Never-married':0, 'Married-civ-spouse':1, 'Married-AF-spouse':1, 'Married-spouse-absent':0})
g = sns.heatmap(data[['relationship', 'marital.status']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

plt.show()
data.drop(['marital.status'], axis=1,inplace=True)
# data.drop(['workclass', 'education', 'occupation', 'native.country'], axis=1,inplace=True)



data.drop(['education'], axis=1,inplace=True)



labels = ['workclass', 'occupation', 'native.country']



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for l in labels:

    data[l]=le.fit_transform(data[l])
data.head(10)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier



from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold, learning_curve, train_test_split, KFold

# from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.svm import SVC

seed = 42



from sklearn.preprocessing import StandardScaler



X = StandardScaler().fit_transform(data.loc[:, data.columns != 'income'])

Y = data['income']



# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)



kf = KFold(n_splits=10, shuffle=True, random_state=seed)
a = len(data.loc[data.income==0])/len(data)

print(a)
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(24, 14))

classifiers = [

    LogisticRegression(solver='newton-cg'),

    KNeighborsClassifier(n_neighbors=17), # Some trial and error I don't show went into this hyperpa

    LinearDiscriminantAnalysis(),

    GaussianNB()

]





for i, c in enumerate(classifiers):

    

    x_axs = i%2

    y_axs = int(i/2)

    # print(c)

    print(type(c).__name__)

    pred = cross_val_predict(c, X, Y, cv=kf)

    print("Accuracy score:", round(accuracy_score(Y, pred), 4), '\n')

    sns.heatmap(confusion_matrix(Y, pred), annot=True, fmt='g', ax=axs[y_axs][x_axs])

    axs[y_axs][x_axs].set_xlabel('Predicted')

    axs[y_axs][x_axs].set_ylabel('Real')

    axs[y_axs][x_axs].set_title(type(c).__name__)



plt.show()

    
import warnings

warnings.filterwarnings(action='ignore')



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(24, 21))

classifiers = [

    DecisionTreeClassifier(),

    BaggingClassifier(),

    RandomForestClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    AdaBoostClassifier()

]





for i, c in enumerate(classifiers):

    

    x_axs = i%2

    y_axs = int(i/2)

    

    # print(c)

    print(type(c).__name__)

    pred = cross_val_predict(c, X, Y, cv=kf)

    print("Accuracy score:", round(accuracy_score(Y, pred), 4), '\n')

    sns.heatmap(confusion_matrix(Y, pred), annot=True, fmt='g', ax=axs[y_axs][x_axs])

    axs[y_axs][x_axs].set_xlabel('Predicted')

    axs[y_axs][x_axs].set_ylabel('Real')

    axs[y_axs][x_axs].set_title(type(c).__name__)



plt.show()

    