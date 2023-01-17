import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



from sklearn import preprocessing 

file = '../input/mushrooms.csv'

m_data = pd.read_csv(file)
m_data.head()
m_data = m_data.apply(preprocessing.LabelEncoder().fit_transform)
m_data.head()
m_data.describe()
df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'class']
df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'gill-size']
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2





y=m_data['class']

x=m_data.drop(['class'], axis=1)



#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=11)

fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



etmodel = ExtraTreesClassifier()

etmodel.fit(x,y)

feat_importances = pd.Series(etmodel.feature_importances_, index=x.columns).sort_values(kind="quicksort", ascending=False).reset_index()

print(feat_importances)
# feature extraction

model = LogisticRegression()

rfe = RFE(model, 10)

fit = rfe.fit(x, y)

print("Num Features: {}".format(fit.n_features_))

print("Selected Features: {}".format(fit.support_))

print("Feature Ranking: {}".format(fit.ranking_))
x.columns
df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'gill-attachment']
df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'veil-color']
df_all_corr = m_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_all_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_all_corr[df_all_corr['Feature 1'] == 'ring-number']
features = ['gill-color', 'gill-attachment', 'ring-type', 'ring-number', 'gill-size', 'bruises', 'stalk-root',

            'gill-spacing', 'habitat', 'spore-print-color', 'stalk-surface-above-ring', 'class']



data_prefinal = m_data[features]
import sklearn as sk

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score





from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.constraints import maxnorm



x_all = data_prefinal.drop(['class'], axis=1)

y_all = data_prefinal['class']



x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.33, random_state=23)
x_all.shape
def print_score(classifier,x_train,y_train,x_val,y_val,train=True):

    if train == True:

        print("Training results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(x_train))))

        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(x_train))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(x_train))))

        res = cross_val_score(classifier, x_train, y_train, cv=10, n_jobs=-1, scoring='balanced_accuracy')

        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))

        print('Standard Deviation:\t{0:.4f}'.format(res.std()))

    elif train == False:

        print("Test results:\n")

        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_val,classifier.predict(x_val))))

        print('Classification Report:\n{}\n'.format(classification_report(y_val,classifier.predict(x_val))))

        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_val,classifier.predict(x_val))))
svcmodel = svm.SVC(kernel='linear', gamma='scale').fit(x_train, y_train)

svprediction = svcmodel.predict(x_val)



def svc_param_selection(X, y, nfolds):

    Cs = [0.001]

    gammas = [0.1]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(svm.SVC(kernel = 'linear'), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    print(grid_search.best_params_) 

    return grid_search.best_estimator_



sv_best = svc_param_selection(x_train, y_train, 10)

sv_prediction = sv_best.predict(x_val)

print_score(sv_best, x_train, y_train, x_val, y_val, train=True)
print_score(sv_best, x_train, y_train, x_val, y_val, train=False)
rf = RandomForestClassifier()

rfmodel = rf.fit(x_train, y_train)

prediction = rfmodel.predict(x_val)
print_score(rfmodel, x_train, y_train, x_val, y_val, train=True)
print_score(rfmodel, x_train, y_train, x_val, y_val, train=False)
lrmodel = LogisticRegression()

lrmodel = lrmodel.fit(x_train, y_train)
print_score(lrmodel, x_train, y_train, x_val, y_val, train=True)
def knn_param_selection(X, y, nfolds):

    n_neighbors = [1, 2, 3, 4, 5, 6, 7]

    weights = ['distance', 'uniform']

    metric = ['euclidean', 'manhattan']

    param_grid = {'n_neighbors': n_neighbors, 'weights' : weights, 'metric': metric}

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    print(grid_search.best_params_) 

    return grid_search.best_estimator_



knn_best = knn_param_selection(x_train, y_train, 10)
print_score(knn_best, x_train, y_train, x_val, y_val, train=True)
print_score(knn_best, x_train, y_train, x_val, y_val, train=False)
nnmodel = Sequential()

nnmodel.add(Dense(15, input_dim = 11, activation='relu'))

nnmodel.add(Dropout(0.2))

nnmodel.add(Dense(15, activation='relu'))

nnmodel.add(Dropout(0.2))

nnmodel.add(Dense(15, activation='sigmoid'))

nnmodel.add(Dropout(0.2))

nnmodel.add(Dense(15, activation='relu'))

nnmodel.add(Dropout(0.2))

nnmodel.add(Dense(1, activation='sigmoid'))



nnmodel.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
nnmodel.fit(x_train, y_train, batch_size=8100, epochs=2000)
y_pred=nnmodel.predict(x_val)

y_pred=(y_pred>0.5)
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))