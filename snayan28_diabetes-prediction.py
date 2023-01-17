# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/diabetes.csv')

data.head()
data.shape
data.describe()
g = sns.PairGrid(data, vars=['Glucose', 'Insulin', 'BMI'], hue="Outcome", size=2.4)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.add_legend()

plt.show()
g = sns.PairGrid(data, vars=['Age', 'SkinThickness', 'BloodPressure'], hue="Outcome", size=2.4)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.add_legend()

plt.show()
g = sns.PairGrid(data, vars=['Pregnancies', 'DiabetesPedigreeFunction'], hue="Outcome", size=3.5)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.add_legend()

plt.show()
columns = ['Glucose', 'Age', 'BloodPressure', 'Insulin','BMI','SkinThickness' ,'Pregnancies',  'DiabetesPedigreeFunction']

n_cols = 2

n_rows = 4

idx = 0



for i in range(n_rows):

    fg,ax = plt.subplots(nrows=1,ncols=n_cols,sharey=True,figsize=(8, 2.4))

    for j in range(n_cols):

        sns.violinplot(x = data.Outcome, y=data[columns[idx]], ax=ax[j]) 

        idx += 1

        if idx >= 8:

            break
# remove the Outlier of skin thickness



max_skinthickness = data.SkinThickness.max()

data = data[data.SkinThickness!=max_skinthickness]
# create a helper function

def replace_zero(df, field, target):

    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()

    data.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]

    data.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]



    # run the function

for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:   

    replace_zero(data, col, 'Outcome')    
data.describe()
# split data



X = data.iloc[:,:-1]

y = data.iloc[:, -1]



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)

print(X_train.shape)

print(X_test.shape)

print(y_train.size)

print(y_test.size)
# load algorithms

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors  import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.metrics import f1_score



# helper functions

def train_clf(clf, X_train, y_train):

    

    return clf.fit(X_train, y_train)

    

def pred_clf(clf, features, target):

    

    y_pred = clf.predict(features)

    return f1_score(target.values, y_pred, pos_label = 1)



def train_predict(clf, X_train, y_train, X_test, y_test):

    

    train_clf(clf, X_train, y_train)

    

    print("F1 score for training set is: {:.4f}".format(pred_clf(clf, X_train, y_train)))

    print("F1 score for testing set is: {:.4f}\n".format(pred_clf(clf, X_test, y_test)))
# load algorithms

nb = GaussianNB()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=0)

rfc = RandomForestClassifier(random_state=0)

abc = AdaBoostClassifier(random_state=0)

gbc = GradientBoostingClassifier(random_state=0)



algorithms = [nb,knn, dtc, rfc, abc, gbc]



for clf in algorithms:

    """

    print("\n{}: \n".format(clf.__class__.__name__))

    

    # create training data from first 100, then 200, then 300

    #for n in [179, 358, 537]:

        #train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)

    """        

    print("{}:".format(clf))

    train_predict(clf, X_train, y_train, X_test, y_test)
# split training set into training and testing set

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size = 0.3, random_state=100)

for n in range(3,10):    

    knn = KNeighborsClassifier(n_neighbors=n)

    print("Number of neighbors is: {}".format(n))

    train_predict(knn, X_train_cv, y_train_cv, X_test_cv, y_test_cv)
from sklearn.metrics import accuracy_score



knn = KNeighborsClassifier(n_neighbors=8)

clf_ = knn.fit(X_train, y_train)

y_pred = clf_.predict(X_test)

print('Accuracy is {}'.format(accuracy_score(y_test,y_pred )))
knn
from sklearn.metrics import roc_curve

from sklearn.model_selection import KFold

params = {'n_estimators':1200, 'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}

gbc = GradientBoostingClassifier(**params)



n_estimators = 10

clf = gbc



# split training set into training and testing set

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size = 0.3, random_state=100)



X_train_cv = X_train_cv.reset_index(drop=True, inplace=False)

y_train_cv = y_train_cv.reset_index(drop=True, inplace=False)





clf.fit(X_train_cv,y_train_cv)

#score = f1_score(y_train, clf.predict(X_train), pos_label = 1)

acc = clf.score(X_test_cv, y_test_cv)

    

n_estimators = params['n_estimators']

x = np.arange(n_estimators) + 1



""" The following part code was stole from sklearn """

def heldout_score(clf, X_test_cv, y_test_cv):

    """compute deviance scores on ``X_test`` and ``y_test``. """

    score = np.zeros((n_estimators,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_decision_function(X_test_cv)):

        score[i] = clf.loss_(y_test_cv, y_pred)

    return score





def cv_estimate(n_splits=10):

    cv = KFold(n_splits=n_splits)

    cv_clf = clf

    val_scores = np.zeros((n_estimators,), dtype=np.float64)

    for train, test in cv.split(X_train_cv):

        cv_clf.fit(X_train_cv.iloc[train], y_train_cv[train])

        val_scores += heldout_score(cv_clf, X_train_cv.iloc[test], y_train_cv[test])

    val_scores /= n_splits

    return val_scores





# Estimate best n_estimator using cross-validation

cv_score = cv_estimate(3)



# Compute best n_estimator for test data

test_score = heldout_score(clf, X_test_cv, y_test_cv)



# negative cumulative sum of oob improvements

cumsum = -np.cumsum(clf.oob_improvement_)



# min loss according to OOB

oob_best_iter = x[np.argmin(cumsum)]



# min loss according to test (normalize such that first loss is 0)

test_score -= test_score[0]

test_best_iter = x[np.argmin(test_score)]



# min loss according to cv (normalize such that first loss is 0)

cv_score -= cv_score[0]

cv_best_iter = x[np.argmin(cv_score)]

    

# color brew for the three curves

oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))

test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))

cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))



# plot curves and vertical lines for best iterations

plt.plot(x, cumsum, label='OOB loss', color=oob_color)

plt.plot(x, test_score, label='Test loss', color=test_color)

plt.plot(x, cv_score, label='CV loss', color=cv_color)

plt.axvline(x=oob_best_iter, color=oob_color)

plt.axvline(x=test_best_iter, color=test_color)

plt.axvline(x=cv_best_iter, color=cv_color)



# add three vertical lines to xticks

xticks = plt.xticks()

xticks_pos = np.array(xticks[0].tolist() +

                        [oob_best_iter, cv_best_iter, test_best_iter])

xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +

                        ['OOB', 'CV', 'Test'])

ind = np.argsort(xticks_pos)

xticks_pos = xticks_pos[ind]

xticks_label = xticks_label[ind]

plt.xticks(xticks_pos, xticks_label)



plt.legend(loc='upper right')

plt.ylabel('normalized loss')

plt.xlabel('number of iterations')



plt.show()
from sklearn.metrics import accuracy_score



params = {'max_depth':9, 'subsample':0.5, 'learning_rate':0.01, 'min_samples_leaf':1, 'random_state':0}

gbc = GradientBoostingClassifier(n_estimators=290, **params)

clf_ = gbc.fit(X_train, y_train)

y_pred = clf_.predict(X_test)

print('Accuracy is {}'.format(accuracy_score(y_test,y_pred )))

train_predict(gbc, X_train, y_train, X_test, y_test)