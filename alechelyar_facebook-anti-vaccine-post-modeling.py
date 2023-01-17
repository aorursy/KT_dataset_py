#Basic

import os

import numpy as np

import pandas as pd

from functools import reduce

import matplotlib.pyplot as plt

%matplotlib inline



#Features

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split



#Plotting

from matplotlib.patches import Patch

from sklearn.metrics import roc_curve, auc



#ML - XGBoost

import xgboost as xgb



#ML - keras

from keras.layers import Dense

from keras.layers import Dropout

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization



#ML - sklearn

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier
data = pd.read_csv('../input/features_scaled.csv', index_col=0)

X, y = data.drop('anti_vax', axis=1), data.anti_vax

data.head()
data.shape
rfc_feature_estimator = RandomForestClassifier(n_estimators=120, 

                                               random_state=21)

rfc_feature_estimator.fit(X, y)

importances = pd.DataFrame({'Feature':X.columns, 

                            'Importance':rfc_feature_estimator.feature_importances_}

                          ).sort_values('Importance', ascending=False).set_index('Feature')

importances['Kept'] = importances.Importance >= 0.004

importances.loc['has_words', ['Kept']] = True

importances.loc['has_text', ['Kept']] = True

kept_columns = importances[importances.Kept].index.values

X = X[kept_columns]

print(data.shape, ' => ', X.shape)
print("Dropped Features:\n", importances[~importances.Kept].index.values)
importances[importances.Kept].head(30)
#Removed some parameters that were underperforming after running so many times

models = {

    'XGBoostClassifier': {

        'model_base': xgb.XGBClassifier,

        'param_grid': {

            'n_estimators': [400], #200, 500

            'max_depth': [3],

            'learning_rate': [.05], #0.1

            'random_state': [40]

        }

    },

    'LinearSVC': {

        'model_base': LinearSVC,

        'param_grid': {

            'penalty': ['l2'], #l1

            'fit_intercept': [True], #False

            'random_state': [20],

            'dual': [False]

        }

    },

    'AdaBoostClassifier': {

        'model_base': AdaBoostClassifier,

        'param_grid': {

            'n_estimators': [120],

            'learning_rate': [0.5],

            'random_state': [30]

        }

    },

# KNC underperforming and taking 10x as long as others to train

#    'KNeighborsClassifier': {  

#        'model_base': KNeighborsClassifier,

#        'param_grid': {

#            'n_neighbors': [8]

#        }

#    },

    'RandomForestClassifier': {

        'model_base': RandomForestClassifier,

        'param_grid': {

            'n_estimators': [130], #100

            'criterion': ['gini', 'entropy'],

            'random_state': [940]

        }

    },

    'LogisticRegression': {

        'model_base': LogisticRegression,

        'param_grid': {

            'penalty': ['l2'],

            'fit_intercept': [True], #False

            'solver': ['liblinear'],

            'random_state': [382]

        }

    },

    'GradientBoostingClassifier': {

        'model_base': GradientBoostingClassifier,

        'param_grid': {

            'loss': ['deviance'],

            'learning_rate': [0.4], #.3, .5

            'tol': [.005],

            'validation_fraction': [.2],

            'n_iter_no_change': [5],

            'random_state': [30]

        }

    }

}
def score_best_model(model_base, param_grid, cv):

    gs = GridSearchCV(model_base(), param_grid=param_grid, cv=cv, 

                      verbose=0, n_jobs=-1, return_train_score=True)

    gs.fit(X, y)

    results = {'score_' + str(i) : score for i, score 

               in enumerate(list(gs.cv_results_['mean_test_score']))}

    results['best_score'] = gs.best_score_

    results['params'] = gs.best_params_

    return results

def pipeline(models, cv=5):

    results = []

    for name, m in models.items():

        print(name)

        result = score_best_model(*m.values(), cv)

        result['name'] = name

        results.append(result)

    return results

results = pipeline(models)

results = pd.DataFrame(results).set_index('name').sort_values(

    'best_score', ascending=False)

results
print(results.iloc[0].name, results.iloc[0].params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,

                                                    random_state=30)

best_model = results.iloc[0]

_bm = models[best_model.name]['model_base'](**best_model.params)

y_score = _bm.fit(X_train, y_train).predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])

roc_auc = auc(fpr, tpr)



fig, ax = plt.subplots(figsize=(15, 8))

ax.plot(fpr, tpr, color='r',lw=2, label='ROC Curve')

ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve');
test_results = pd.DataFrame(X_test.values, columns=X_test.columns)

for c in test_results.columns:

    test_results[c] = pd.to_numeric(test_results[c])

test_results['predicted'] = _bm.predict(test_results.fillna(test_results.mean())).astype('bool')

test_results['anti_vax'] = y_test.reset_index().anti_vax.astype('bool')

test_results['correct'] = (test_results.anti_vax == test_results.predicted).astype('bool')

print('Accuracy on posts with text: {0:.2f}'.format(test_results[test_results.has_text].correct.mean()))

print('Accuracy on posts without text: {0:.2f}'.format(test_results[~test_results.has_text].correct.mean()))

print('Accuracy on posts with words: {0:.2f}'.format(test_results[test_results.has_words].correct.mean()))

print('Accuracy on posts without words: {0:.2f}'.format(test_results[~test_results.has_words].correct.mean()))
test_result_gb = test_results.groupby('correct').mean().transpose()

test_result_gb['Difference'] = (test_result_gb[True] - test_result_gb[False]).abs()

test_result_gb = test_result_gb.sort_values('Difference', ascending=False).rename(columns={False: 'Incorrect', True: 'Correct'})

test_result_gb.head()
class StackedModel:

    def __init__(self, first_layer_models, second_layer_model):

        self.first_models = first_layer_models

        self.second_model = second_layer_model

    

    def _train_first_model(self, model, X_train, y_train, X_test, n_folds=5):

        train_scores = np.zeros(X_train.shape[0])

        test_set = np.zeros(X_test.shape[0])

        test_scores = np.empty((n_folds, X_test.shape[0]))

        

        cv = KFold(n_splits=n_folds, random_state=70)



        for i, (train_fold, test_fold) in enumerate(cv.split(X_train, y_train)):

            model.fit(X_train[train_fold], y_train[train_fold])

            train_scores[test_fold] = model.predict(X_train[test_fold])

            test_scores[i, :] = model.predict(X_test)



        test_set[:] = test_scores.mean(axis=0)

        return train_scores.reshape(-1, 1), test_set.reshape(-1, 1)

    

    def _train_first_layer(self, X_train, y_train, X_test):

        print('Training First Layer')

        self.train_results = [self._train_first_model(m, X_train, y_train, X_test) 

                              for m in self.first_models]

        

    def _prepare_second_dataset(self):

        print('Preparing Second Dataset')

        second_x_train, second_x_test = zip(*self.train_results)

        self.second_X_train = np.concatenate(second_x_train, axis=1)

        self.second_X_test = np.concatenate(second_x_test, axis=1)

        

    def _train_second_layer(self, y_train, y_test, early_stopping_rounds=10):

        print('Training Second Layer')

        self.second_model.fit(self.second_X_train, y_train, 

                              early_stopping_rounds=early_stopping_rounds, 

                              eval_set=[(self.second_X_test, y_test)])



    def pipeline(self, X_train, y_train, X_test, y_test):

        #Fit models in first layer and compute new datasets for second layer

        self._train_first_layer(X_train, y_train, X_test)

        #Split results and concatenate into columns

        self._prepare_second_dataset()

        #Train second layer

        self._train_second_layer(y_train, y_test)

        return self.second_model.predict(self.second_X_test)
NUM_FIRST_LAYER_MODELS = 4

first_layer = []

for name, params in results.params.head(NUM_FIRST_LAYER_MODELS).items():

    if 'n_jobs' in params:

        del params['n_jobs']

    first_layer.append(models[name]['model_base'](**params))

second_layer = xgb.XGBClassifier(n_jobs=-1, n_estimators= 1000, 

                                 objective='binary:logistic', random_state=2)

sm = StackedModel(first_layer, second_layer)

sm_predictions = sm.pipeline(X_train.values, y_train.values, 

                             X_test.values, y_test.values)

print("Stacked Model Accuracy:", accuracy_score(y_test, sm_predictions))
seed = np.random.seed(4)

def build_model(*layers):

    model = Sequential()

    for layer in layers:

        model.add(layer)

    model.compile(loss='binary_crossentropy', optimizer='adam', 

                  metrics=['accuracy', 'binary_crossentropy'])

    return model

keras_models = {

    'baseline_12': build_model(

        Dense(12, input_dim=X.shape[1], activation='relu', use_bias=True), 

        Dense(1, activation='sigmoid')

    ),

    'baseline_dropout_12': build_model(

        Dropout(rate=0.5),

        Dense(12, input_dim=X.shape[1], activation='relu', use_bias=True),

        Dense(1, activation='sigmoid')

    ),

    'baseline_dropout_50': build_model(

        Dropout(rate=0.5),

        Dense(12, input_dim=X.shape[1], activation='relu', use_bias=True),

        Dense(1, activation='sigmoid')

    ),

    'baseline_norm_50': build_model(

        BatchNormalization(),

        Dense(50, input_dim=X.shape[1], activation='relu', use_bias=True), 

        Dense(1, activation='sigmoid')

    ),

    'baseline_norm_30_12': build_model(

        BatchNormalization(),

        Dense(30, input_dim=X.shape[1], activation='relu', use_bias=True), 

        Dense(12, activation='relu', use_bias=True), 

        Dense(1, activation='sigmoid')

    ),

    'baseline_droupout_30_12': build_model(

        Dropout(rate=0.5),

        Dense(30, input_dim=X.shape[1], activation='relu', use_bias=True), 

        Dense(12, activation='relu', use_bias=True), 

        Dense(1, activation='sigmoid')

    )

}
def train_model(model):

    return model.fit(X.values, y.values, validation_split=0.20, epochs=30, 

                     callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, 

                                             patience=2, verbose=0, mode='auto')])

nn_results = [(name, train_model(m)) for name, m in keras_models.items()]

nn_results = [(name, pd.DataFrame(hist.history)) for name, hist in nn_results]
def plot_epochs(results, col, **kwargs):

    def plot_epoch_helper(hist_df, col, ax):

        ax.plot(hist_df[col], **kwargs)

        ax.set_title(col + ' per epoch')

        ax.set_ylabel(col)

        ax.set_xlabel('epoch')

        for sp in ax.spines:

            ax.spines[sp].set_visible(False)

        ax.yaxis.grid(True, alpha=0.3)

        ax.legend(labels=[n[0] for n in nn_results])

        ax.set_ylim(0, 1)

    fig, ax = plt.subplots(figsize=(21, 10))

    for name, hist in results:

        plot_epoch_helper(hist, col, ax)

plot_epochs(nn_results, 'loss')
plot_epochs(nn_results, 'acc')
plot_epochs(nn_results, 'val_loss')
plot_epochs(nn_results, 'val_acc')
best_nn = []

for name, history in nn_results:

    best_nn.append((name, history.val_acc.max(), history.val_acc.idxmax()))

pd.DataFrame(best_nn, columns=['name', 'best_val_acc', 'best_epoch']).sort_values('best_val_acc', ascending=False)