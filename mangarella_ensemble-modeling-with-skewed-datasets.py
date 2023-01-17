import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.calibration import calibration_curve
def load_fraud_data(filename):

    df_data = pd.read_csv(filename)

    return df_data



df_data = load_fraud_data('../input/creditcard.csv')
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



    

def classification_setup(df_data):

    '''Returns X_data, y_data, ls_features'''

    X_data, y_data = df_data, df_data['Class']

    ls_features = list(X_data.keys())

    class_index = ls_features.index('Class')

    ls_features.pop(class_index)

    return X_data, y_data, ls_features



X_data, y_data, ls_features = classification_setup(df_data)
def adaboost_no_undersample(X_data, y_data):

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

    dt_clf = DecisionTreeClassifier(max_depth = 1)

    ada_real = AdaBoostClassifier(base_estimator = dt_clf, 

                                  learning_rate = 0.1, 

                                  n_estimators = 100)

    ada_real.fit(X_train[ls_features], y_train)

    y_pred = ada_real.predict(X_test[ls_features])

    test_conf = confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(test_conf)



adaboost_no_undersample(X_data, y_data)
def under_sample_kfold(X_data, y_data, n_folds = 10):

    '''Returns list of 10 (default) folds of

    X_train, X_test, y_train, y_test data'''

    

    pos_events = X_data[X_data['Class'] == 1]

    neg_events = X_data[X_data['Class'] == 0]

    

    #Randomize and pick same n number of events

    number_pos_events = len(pos_events)  

    undersampled_folds = []



    for fold in range(0, n_folds):

        pos_events = pos_events.reindex(np.random.permutation(pos_events.index))

        neg_events = neg_events.reindex(np.random.permutation(neg_events.index))

        undersampled_events = pd.concat([neg_events.head(number_pos_events), pos_events])

        X_data_u, y_data_u = undersampled_events, undersampled_events['Class']

        X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_data_u, y_data_u, test_size=0.3)

        undersampled_folds.append([X_train_u, X_test_u, y_train_u, y_test_u])

    return undersampled_folds





folds = under_sample_kfold(X_data, y_data)
def adaboost_undersample(folds, full_test_set = False):

    

    for fold in folds:

        X_train, X_test, y_train, y_test = fold[0], fold[1], fold[2], fold[3]

    

        dt_clf = DecisionTreeClassifier(max_depth = 1)

        ada_real = AdaBoostClassifier(base_estimator = dt_clf, 

                                      learning_rate = 0.1, 

                                      n_estimators = 100)

        ada_real.fit(X_train[ls_features], y_train)



        if full_test_set == False:

            y_pred = ada_real.predict(X_test[ls_features])

            test_conf = confusion_matrix(y_test, y_pred)

            print (test_conf)

    

        else:

            X_train_and_test = pd.concat([X_train, X_data])

            X_test_full = (X_train_and_test.reset_index()

                                           .drop_duplicates(subset= 'index', keep= False)

                                           .set_index('index'))

            y_test_full = X_test_full['Class']

        

            #Eval

            y_pred = ada_real.predict(X_test_full[ls_features])

            test_conf = confusion_matrix(y_test_full, y_pred)

            print (test_conf)
adaboost_undersample(folds)
adaboost_undersample(folds, full_test_set = True)
def cv_setup(X_data, y_data):

    '''Returns X_data, y_data, X_cv, y_cv, ls_features

    X_cv, y_cv are randomized 10% of data with same class

    proportions'''

    

    

    pos_events = X_data[X_data['Class'] == 1]

    neg_events = X_data[X_data['Class'] == 0]

    number_pos_events, number_neg_events, number_events = (len(pos_events), 

                                                           len(neg_events),

                                                           len(X_data))

    pos_events = pos_events.reindex(np.random.permutation(pos_events.index))

    neg_events = neg_events.reindex(np.random.permutation(neg_events.index))

    X_cv = pd.concat([neg_events.tail(number_neg_events//10), 

                      pos_events.tail(number_pos_events//10)])

    y_cv = X_cv['Class']



    #Get rid of duplicates between CV and Test/Train bank of data

    X_data = pd.concat([X_cv, X_data])

    X_data = (X_data.reset_index()

                    .drop_duplicates(subset= 'index', keep= False)

                    .set_index('index'))

    y_data = X_data['Class']

    

    return X_data, y_data, X_cv, y_cv, ls_features
#Remake our folds without including 10% of the data for a CV set

X_data, y_data, X_cv, y_cv, ls_features = cv_setup(X_data, y_data)

folds = under_sample_kfold(X_data, y_data)
def adaboost_undersample_ensemble(folds):

    

    X_train_all_folds = pd.DataFrame()

    ada_ensemble = []

    

    for fold in folds:

        X_train, X_test, y_train, y_test = fold[0], fold[1], fold[2], fold[3]

    

        dt_clf = DecisionTreeClassifier(max_depth = 1)

        ada_real = AdaBoostClassifier(base_estimator = dt_clf, 

                                      learning_rate = 0.1, 

                                      n_estimators = 100)

        ada_real.fit(X_train[ls_features], y_train)

        #Concatenate all train sets and store each fold models

        X_train_all_folds = pd.concat([X_train_all_folds, X_train])

        ada_ensemble.append(ada_real)

    

    #Call scorer

    ensemble_score(X_cv, y_cv, ada_ensemble)



def ensemble_score(X_test, y_test, models, cutoff = 0.5, e_cutoff = 5):

    '''Prints confusion matrix for an ensemble of models based on 

    probability cutoff (cutoff) of each model and voting cutoff 

    (e_cutoff) for an ensemble vote. Default is >50% probability

    and > 5 votes out of 10'''

    

    for fold in range(0, len(models)):

        X_test.loc[:, ('Prob fold ' + str(fold))] = models[fold].predict_proba(X_test[ls_features])[:,1]

        X_test.loc[:, ('ADAPred fold ' + str(fold))] = (X_test[('Prob fold ' + str(fold))] >= cutoff).astype(int)

    

    X_test.loc[:, 'Ensemble Score'] = X_test[['ADAPred fold ' + str(i) for i in range(0, len(models))]].sum(axis = 1)

    X_test.loc[:, 'Ensemble Pred'] = (X_test['Ensemble Score'] > e_cutoff).astype(int)



    ensemble_conf = confusion_matrix(y_test, X_test[('Ensemble Pred')])

    print (ensemble_conf)



adaboost_undersample_ensemble(folds)
X_data, y_data, ls_features = classification_setup(df_data)

X_data, y_data, X_cv, y_cv, ls_features = cv_setup(X_data, y_data)

folds = under_sample_kfold(X_data, y_data)

adaboost_undersample_ensemble(folds)