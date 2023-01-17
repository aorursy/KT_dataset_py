%matplotlib inline



import numpy as np 

import pandas as pd  

import sklearn as skl

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
train_file = '../input/train.csv'

df_train = pd.read_csv(train_file)

df_train.head(n=10)
df_train.Cabin.unique()
test_file = '../input/test.csv'

df_test = pd.read_csv(test_file)

df_test.head(n=10)
fig, ax = plt.subplots(figsize=(12,7))



df_saved = df_train.loc[df_train['Survived'] == 1]

df_saved['Fare'].hist(ax=ax, bins=20, alpha=0.4, label='safe')



 

df_unsaved = df_train.loc[df_train['Survived'] == 0]

df_unsaved['Fare'].hist(ax=ax, bins=20, alpha=0.4, label='dead')



ax.legend()



df_train['Fare'] = df_train['Fare'].apply(lambda x: min(x, 150.))
fig, ax = plt.subplots(figsize=(12,7))



col = 'SibSp'



df_saved = df_train.loc[df_train['Survived'] == 1]

df_saved[col].hist(ax=ax, bins=20, alpha=0.4, label='safe')



 

df_unsaved = df_train.loc[df_train['Survived'] == 0]

df_unsaved[col].hist(ax=ax, bins=20, alpha=0.4, label='dead')



ax.legend()

ax.set_ylim([0,20])



df_train[col] = df_train[col].apply(lambda x: min(x, 5.))
fig, ax = plt.subplots(figsize=(12,7))



col = 'Parch'



df_saved = df_train.loc[df_train['Survived'] == 1]

df_saved[col].hist(ax=ax, bins=20, alpha=0.4, label='safe')



 

df_unsaved = df_train.loc[df_train['Survived'] == 0]

df_unsaved[col].hist(ax=ax, bins=20, alpha=0.4, label='dead')



ax.legend()

ax.set_ylim([0,20])



df_train[col] = df_train[col].apply(lambda x: min(x, 3))
for col in df_train:

    print('{0:12s}is\t{1}\t--- percent of null: {2:.2f}% --- number of unique vals:\t{3}/{4}'

          .format(col, 

                  df_train[col].dtype, 

                  df_train[col].isnull().sum()/df_train.shape[0]*100, 

                  len(df_train[col].unique()), df_train[col].notnull().sum()

                 ))
df_train.info()
df_train.describe()
df_train.describe(include=['O'])
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

df_train.head(n=10)

df_train.drop('Name', axis=1, inplace=True)
#for x in df_train.Title.unique(): print(x)

df_train.Title.value_counts()
df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr',

         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'RareTitle')



df_train['Title'] = df_train['Title'].replace('Mlle', 'Miss')

df_train['Title'] = df_train['Title'].replace('Ms', 'Miss')

df_train['Title'] = df_train['Title'].replace('Mme', 'Mrs')
df_train.head(n=10)
nominals = ['Sex', 'Embarked', 'Title']

for col in nominals:

    one_hot = pd.get_dummies(df_train[col])

    df_train = df_train.join(one_hot)

    df_train.drop(col, axis=1, inplace=True) # DROP: sex, embarked, title



# Binary variables: just use 1 column!!    

df_train.drop('female', axis=1, inplace=True) # DROP: female
df_train.head()
drop = ['PassengerId', 'Ticket', 'Cabin']

for col in drop:

    df_train.drop(col, axis=1, inplace=True)

df_train.head()
imputation = ['Age']

for col in imputation:

    df_train[col].fillna(value=df_train[col].median(), inplace=True)

#df_train['FarexClass'] = df_train['Fare']* df_train['Pclass']
df_train.head()
X = df_train.loc[:, df_train.columns != 'Survived'].as_matrix()

y = df_train.as_matrix(columns=['Survived'])
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from xgboost import XGBClassifier
names = ["Decision Tree", 

         "Random Forest", 

         "AdaBoost",

         "XgBoost"

        ]



classifiers = [

    DecisionTreeClassifier(max_depth=3, criterion='entropy'),

    RandomForestClassifier(max_depth=3, n_estimators=100, max_features=1),

    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100),

    XGBClassifier(max_depth=3, n_estimators=100)

]

import sys 



accuracies_dict = dict()



X = df_train.loc[:, df_train.columns != 'Survived'].as_matrix()

y = df_train.as_matrix(columns=['Survived'])

y = np.ravel(y)



n_folds=2

skf = StratifiedKFold(n_splits=2)



for name, classifier in zip(names, classifiers):

            

    kf_accuracy_test = []

    kf_accuracy_trai = []

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train.ravel())

        kf_accuracy_trai.append(classifier.score(X_train, y_train))

        kf_accuracy_test.append(classifier.score(X_test, y_test))

        

    print('--- {0} ---'.format(name))

    n_samples = df_train.shape[0]

    acc_cv = np.mean(kf_accuracy_test)

    Var_acc_cv = acc_cv*(1-acc_cv)/n_samples

    Std_acc_cv = np.sqrt(Var_acc_cv)

    print('Accuracy = {0:.2f} –– 95% CI = [{0:.2f}, {2:.2f}] '. \

          format(acc_cv*100, 100*(acc_cv-2*Std_acc_cv), 100*(acc_cv+2*Std_acc_cv)))

    accuracies_dict[name] = (acc_cv, Std_acc_cv)

k = 0

f, ax = plt.subplots(figsize=(15,5));

for x,y in accuracies_dict.values():

    ax.errorbar(x=k, y=x, yerr=y, fmt='--o')

    k += 1



a = ['0']

a.extend(accuracies_dict.keys());

ax.set_xticklabels(a);

ax.grid()
def plot_embedding(X, title=None):

    x_min, x_max = np.min(X, 0), np.max(X, 0)

    X = (X - x_min) / (x_max - x_min)



    f, ax = plt.subplots(figsize=(12,8))

    for i in range(X.shape[0]):

        if y[i]==0:

            plt.plot(X[i, 0], X[i, 1], 'r.')

        if y[i]==1:

            plt.plot(X[i, 0], X[i, 1], 'b.') 

from sklearn.manifold import TSNE



TSNE_PLOT = False

if TSNE_PLOT:

    X_embedded = TSNE(n_components=2).fit_transform(X)

    plot_embedding(X_embedded)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca.fit(X)    

print(pca.explained_variance_ratio_)
X = pca.fit_transform(X)
plot_embedding(X)


for name, classifier in zip(names, classifiers):

            

    kf_accuracy_test = np.empty(n_kf, dtype=np.float32)

    kf_accuracy_trai = np.empty(n_kf, dtype=np.float32)

    i = 0

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        classifier.fit(X_train, y_train.ravel())

        kf_accuracy_test[i] = classifier.score(X_test, y_test)

        kf_accuracy_trai[i] = classifier.score(X_train, y_train)

        i += 1

    print('--- {0} ---'.format(name))

    print('Test Accuracy: {0:.2f} ± {1:.2f}'.format(kf_accuracy_test.mean()*100, kf_accuracy_test.std()*100))

    print('Train Accuracy: {0:.2f} ± {1:.2f}'.format(kf_accuracy_trai.mean()*100, kf_accuracy_trai.std()*100))

    accuracies_dict[name] = (kf_accuracy_test.mean(), kf_accuracy_test.std())

    print()
