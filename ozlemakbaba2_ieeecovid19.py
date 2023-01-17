import numpy as np 

import pandas as pd 



data = pd.read_csv("../input/ieee8023covidchestxraydataset/metadata.csv")
data.head()
y = (data['finding'] == 'COVID-19').astype(int)

fields = list(data.columns[:-1])

correlations = data[fields].corrwith(y)

correlations.sort_values(inplace=True)

correlations


import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_context('talk')

sns.set_palette('dark')

sns.set_style('white')

sns.pairplot(data, hue='finding')
data.drop(['filename', 'view', 'view', 'location', 'doi', 'license', 'clinical notes','other notes', 'Unnamed: 16', 'date', 'modality', ' url', 'survival'], axis=1, inplace=True)
data.head()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



le = LabelEncoder() #offset to int

data['offset'] = le.fit_transform(data.offset)

data['age'] = le.fit_transform(data.age)#offset to age
data.dtypes.value_counts()
data.drop(['sex'], axis=1, inplace=True)
from sklearn.preprocessing import OneHotEncoder

from numpy import array

enc = OneHotEncoder(handle_unknown='ignore')

X = [['F', 0], ['M', 1]]

enc.fit(X)

enc.categories_

[array(['F', 'M'], dtype=object), array([0, 1, 3], dtype=int)]

data.head()
from sklearn.preprocessing import MinMaxScaler



fields = correlations.map(abs).sort_values().iloc[-2:].index

print(fields)

X = data[fields]

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=['%s_scaled' % fld for fld in fields])

print(X.columns)
data.dtypes.tail()
from sklearn.svm import LinearSVC



LSVC = LinearSVC()

LSVC.fit(X, y)



X_color = X.sample(100, random_state=50)



y_color = y.loc[X_color.index]

y_color = y_color.map(lambda r: 'red' if r == 1 else 'yellow')

ax = plt.axes()

ax.scatter(

    X_color.iloc[:, 0], X_color.iloc[:, 1],

    color=y_color, alpha=1)

# -----------

x_axis, y_axis = np.arange(0, 1.005, .005), np.arange(0, 1.005, .005)

xx, yy = np.meshgrid(x_axis, y_axis)

xx_ravel = xx.ravel()

yy_ravel = yy.ravel()

X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T

y_grid_predictions = LSVC.predict(X_grid)

y_grid_predictions = y_grid_predictions.reshape(xx.shape)

ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3)

# -----------

ax.set(

    xlabel=fields[0],

    ylabel=fields[1],

    xlim=[0, 1],

    ylim=[0, 1],

    title='decision boundary for LinearSVC');
def plot_decision_boundary(estimator, X, y):

    estimator.fit(X, y)

    X_color = X.sample(100)

    y_color = y.loc[X_color.index]

    y_color = y_color.map(lambda r: 'red' if r == 1 else 'yellow')

    x_axis, y_axis = np.arange(0, 1, .005), np.arange(0, 1, .005)

    xx, yy = np.meshgrid(x_axis, y_axis)

    xx_ravel = xx.ravel()

    yy_ravel = yy.ravel()

    X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T

    y_grid_predictions = estimator.predict(X_grid)

    y_grid_predictions = y_grid_predictions.reshape(xx.shape)



    fig, ax = plt.subplots(figsize=(10, 10))

    ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3)

    ax.scatter(X_color.iloc[:, 0], X_color.iloc[:, 1], color=y_color, alpha=1)

    ax.set(

        xlabel=fields[0],

        ylabel=fields[1],

        title=str(estimator))

    

    
from sklearn.svm import SVC



gammas = [.5, 1, 2, 10]

for gamma in gammas:

    SVC_Gaussian = SVC(kernel='rbf', gamma=gamma)

    plot_decision_boundary(SVC_Gaussian, X, y)

    

    
Cs = [.1, 1, 10]



for C in Cs:

    SVC_Gaussian = SVC(kernel='rbf', gamma=2, C=C)

    plot_decision_boundary(SVC_Gaussian, X, y)

    

    

    

    




Cs = [10,20,100,200]

for C in Cs:

    SVC_Poly = SVC(kernel='poly', degree=3, C=C)

    plot_decision_boundary( SVC_Poly, X, y)

    

    

    
from sklearn.kernel_approximation import Nystroem

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier



y = data.finding == 'COVID-19'

X = data[data.columns[:-1]]



kwargs = {'kernel': 'rbf'}

svc = SVC(**kwargs)

nystroem = Nystroem(**kwargs)

sgd = SGDClassifier()

%%timeit

svc.fit(X, y)
%%timeit

X_transformed = nystroem.fit_transform(X)

sgd.fit(X_transformed, y)
X2 = pd.concat([X]*5)

y2 = pd.concat([y]*5)



print(X2.shape)

print(y2.shape)
%timeit svc.fit(X2, y2)
%%timeit

X2_transformed = nystroem.fit_transform(X2)

sgd.fit(X2_transformed, y2)
from sklearn import svm

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



y = data.finding == 'COVID-19'

X = data[data.columns[:-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)





parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



scores = ['precision', 'recall']

for score in scores:

    print("# Tuning hyper-parameters for %s" % score)

    print()

    clf = GridSearchCV(SVC(), parameters, scoring='%s_macro' % score)

    clf.fit(X_train, y_train)

    

    print("Best parameters in the training set:")

    print(clf.best_params_)

    

#uzun sürüyor işlem
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
y_true, y_pred = y_test, clf.predict(X_test)

print(classification_report(y_true, y_pred))



cm = confusion_matrix(y_true, y_pred)

print(cm)



accuracy = accuracy_score(y_true, y_pred)

print(accuracy)

from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(random_state=42)

dt = dt.fit(X_train, y_train)

dt.tree_.node_count, dt.tree_.max_depth
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def measure_error(y_true, y_pred, label):

    return pd.Series({'accuracy':accuracy_score(y_true, y_pred),

                      'precision': precision_score(y_true, y_pred),

                      'recall': recall_score(y_true, y_pred),

                      'f1': f1_score(y_true, y_pred)},

                      name=label)
y_train_pred = dt.predict(X_train)

y_test_pred = dt.predict(X_test)



train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),

                              measure_error(y_test, y_test_pred, 'test')],

                              axis=1)



train_test_full_error

from sklearn import metrics 

print("Accuracy:",accuracy_score(y_true, y_pred))
from sklearn.model_selection import GridSearchCV



param_grid = {'max_depth':range(1, dt.tree_.max_depth+1, 2),

              'max_features': range(1, len(dt.feature_importances_)+1)}



GR = GridSearchCV(DecisionTreeClassifier(random_state=42),

                  param_grid=param_grid,

                  scoring='accuracy',

                  n_jobs=-1)



GR = GR.fit(X_train, y_train)
GR.best_estimator_.tree_.node_count, GR.best_estimator_.tree_.max_depth
y_train_pred_gr = GR.predict(X_train)

y_test_pred_gr = GR.predict(X_test)



train_test_gr_error = pd.concat([measure_error(y_train, y_train_pred_gr, 'train'),

                                 measure_error(y_test, y_test_pred_gr, 'test')],

                                axis=1)
train_test_gr_error
from sklearn import metrics 

print("Accuracy:",metrics.accuracy_score(y_train, y_train_pred_gr))
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_context('notebook')

sns.set_style('white')

sns.set_palette('dark')



%matplotlib inline
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X, y)

tree.plot_tree(clf) 
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score

X = data[data.columns[:-1]]

y = data.finding



GNB = GaussianNB()

cv_N = 4

scores = cross_val_score(GNB, X, y, n_jobs=cv_N, cv=cv_N)

print(scores)

np.mean(scores)
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

X = data[data.columns[:-1]]

y = data.finding

nb = {'gaussian': GaussianNB(),

      'bernoulli': BernoulliNB(),

      'multinomial': MultinomialNB()}

scores = {}

for key, model in nb.items():

    s = cross_val_score(model, X, y, cv=cv_N, n_jobs=cv_N, scoring='accuracy')

    scores[key] = np.mean(s)

scores
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

X = data[['age', 'offset']]

y = data.finding



nb = {'gaussian': GaussianNB(),

      'bernoulli': BernoulliNB(),

      'multinomial': MultinomialNB()}

scores = {}

for key, model in nb.items():

    s = cross_val_score(model, X, y, cv=cv_N, n_jobs=cv_N, scoring='accuracy')

    scores[key] = np.mean(s)

scores

X = data[data.columns[:-1]]

y = data.finding



n_copies = [0, 1, 3, 5, 10, 50, 100]





def create_copies_age(X, n):

    X_new = X.copy()

    for i in range(n):

        X_new['age_copy%s' % i] = X['age']

    return X_new





def get_cross_val_score(n):

    X_new = create_copies_age(X, n)

    scores = cross_val_score(GaussianNB(), X_new, y, cv=cv_N, n_jobs=cv_N)

    return np.mean(scores)





avg_scores = pd.Series(

    [get_cross_val_score(n) for n in n_copies],

    index=n_copies)



ax = avg_scores.plot()

ax.set(

    xlabel='number of extra copies of "age"',

    ylabel='average accuracy score',

    title='Decline in Naive Bayes performance');



from sklearn.model_selection import train_test_split

X = data[data.columns[:-1]]

Y = data.finding

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)



from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

GNB = GNB.fit(X_train, Y_train) 

Y_predict = GNB.predict(X_test)





from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,Y_predict)

print(cm)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_predict)

print(accuracy)



import seaborn as sns; sns.set()

uniform_data = np.random.rand(3,3)

ax = sns.heatmap(uniform_data, vmin=0, vmax=1)

ax = sns.heatmap(cm, annot=True, fmt="d")
X_discrete = pd.DataFrame.rank(X,pct=True)

X_discrete.applymap(lambda x: round(x,2))

X_train, X_test, Y_train, Y_test = train_test_split(X_discrete, Y, test_size=0.33,random_state=0)



MNB = MultinomialNB()



MNB = MNB.fit(X_train, Y_train) 

Y_predict = MNB.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_predict)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,Y_predict)

print(cm)

import matplotlib.pyplot as plt





fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

langs = ['SVM', 'DESICION TREE', 'NAVIE BAYES']

result = [0.8846153846153846,0.9611650485436893,0.7681159420289855]

ax.bar(langs,result)

plt.show()
