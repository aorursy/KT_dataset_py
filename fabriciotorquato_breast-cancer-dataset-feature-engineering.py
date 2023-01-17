from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_classif

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier



import warnings

warnings.filterwarnings(action='ignore')
def plot_accs(values):

    plt.plot(range(len(values)), values, '-ko')

    sns.despine(offset=15)

    

def train_with_range(train_function, X_train, X_test, y_train, y_test):

    accs = []

    for value in range(2,25):

        accs.append(train_function(value, X_train, X_test, y_train, y_test))

    return accs
breast_cancer = load_breast_cancer()

data = pd.DataFrame(breast_cancer.data,columns=breast_cancer.feature_names)

data['target'] = pd.Series(breast_cancer.target)

data.head()
data.info()
data.describe()
col = data.columns       # .columns gives columns names in data 

print(col)
ax = sns.countplot(data.target,label="Count")

B, M = data.target.value_counts()

print('Number of Benign: ',B)

print('Number of Malignant : ',M)
featureMeans = list(data.columns[:-1])

featureMeans
X = data.loc[:,featureMeans]

y = data.loc[:, 'target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.95, random_state = 42)
def train_logistic(X_train, X_test, y_train, y_test):

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print('Acurácia:',acc )

    return acc
def use_PCA(pca_value,X_train,X_test):

    pca = PCA(pca_value)

    X_train_pca = pca.fit_transform(X_train)

    X_test_pca = pca.transform(X_test)    

    return X_train_pca, X_test_pca
def train_logistic_with_pca(pca_value, X_train, X_test, y_train, y_test):

    X_train_pca, X_test_pca = use_PCA(pca_value,X_train,X_test)

    return train_logistic(X_train_pca, X_test_pca, y_train, y_test)
accs = train_with_range(train_logistic_with_pca, X_train, X_test, y_train, y_test)
plot_accs(accs)
def use_KBest(kbest_value, X_train, X_test, y_train, y_test):

    fvalue_selector = SelectKBest(f_classif, k=kbest_value)

    X_train_kbest = fvalue_selector.fit_transform(X_train, y_train)

    X_test_kbest = fvalue_selector.transform(X_test)    

    return X_train_kbest, X_test_kbest
def train_logistic_with_kbest(kbest_value, X_train, X_test, y_train, y_test):

    X_train_kbest, X_test_kbest = use_KBest(kbest_value, X_train, X_test, y_train, y_test)

    return train_logistic(X_train_kbest, X_test_kbest, y_train, y_test)
accs = train_with_range(train_logistic_with_kbest, X_train, X_test, y_train, y_test)
plot_accs(accs)
def train_ensemble_models(X, y):

    clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)

    clf2 = RandomForestClassifier(n_estimators=10, random_state=1)

    clf3 = GaussianNB()

    eclf_x2 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='hard')

    eclf_x3 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb',clf3)], voting='hard')



    for clf, label in zip([clf1, clf2, clf3, eclf_x2, eclf_x3], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble X2', 'Ensemble X3']):

        execute_pipeline(clf, X, y, label)

    
def execute_pipeline(clf, X, y, title):



    pipe = Pipeline([

        ('reduce_dim', 'passthrough'),

        ('classify', clf)

    ])



    N_FEATURES_OPTIONS = [2, 4, 10, 20]

    

    param_grid = [

        {

            'reduce_dim': [PCA()],

            'reduce_dim__n_components': N_FEATURES_OPTIONS

        },

        {

            'reduce_dim': [SelectKBest()],

            'reduce_dim__k': N_FEATURES_OPTIONS

        },

    ]

    reducer_labels = ['PCA', 'KBest']



    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid,return_train_score=True)

    grid.fit(X, y)



    mean_train_scores = np.array(grid.cv_results_['mean_train_score'])

    mean_scores = np.array(grid.cv_results_['mean_test_score'])

    mean_scores = mean_scores.reshape(2, len(N_FEATURES_OPTIONS))

    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + .5)



    plt.figure()

    COLORS = 'bgrcmyk'

    for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):

        plt.bar(bar_offsets + i, mean_train_scores[i], label='{} train'.format(label),alpha=.7)

        plt.bar(bar_offsets + i, reducer_scores, label='{} test'.format(label), color=COLORS[i])



    plt.title(title)

    plt.xlabel('Number of features')

    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)

    plt.ylabel('Classification accuracy')

    plt.ylim((0.87, 1))

    plt.legend(loc='upper left')



    plt.show()
train_ensemble_models(X, y)