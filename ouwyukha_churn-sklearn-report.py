import numpy as np

import pandas as pd

import os

import seaborn as sns

from matplotlib import pyplot as plt

import time

import warnings



warnings.filterwarnings('ignore')



%matplotlib inline



pd.set_option("display.max_columns", 500) 

pd.set_option("display.max_rows", 200)

pd.set_option('display.max_colwidth', -1)

sns.set_style("darkgrid")

plt.style.use('seaborn-darkgrid')



csv_paths = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        csv_paths.append(os.path.join(dirname, filename))
csv_paths
raw_dataframes = []

for path in csv_paths:

    df = pd.read_csv(path, index_col=0)

    raw_dataframes.append(df)
from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import RidgeClassifierCV

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,  accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
def preprocess_data(dataframe, dropped_features = [None]):

    dropped_features.append('churn')

    feature_label = [col for col in dataframe.columns if col not in dropped_features]

    X = dataframe[feature_label]

    y = dataframe[['churn']]

    

    return X,y
def create_model(model_dict):

    return model_dict['model'](**model_dict['params'])



def train_model(model, x, y):

    return model.fit(x,y)



def predict_result(model, x):

    return model.predict(x)
def metric_score(y_true, y_pred, title):

    accuracy = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_pred)

    

    print("Metrics for " + title)

    print("Accuracy  :", accuracy)

    print("F1        :", f1)

    print("Precision :", precision)

    print("Recall    :", recall)

    print("A.U.C     :", auc)
def plot_confusion_matrix(y_true, y_pred, title):

    plt.figure(figsize = (20,7))

    plt.subplots_adjust(right=1)

    plt.subplot(1, 2, 1)



    data = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))

    print(df_cm)

    df_cm.index.name = 'Actual'

    df_cm.columns.name = 'Predicted'

    plt.title("Confusion Matrix : " + title)

    sns.set(font_scale=1.4)

    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

    

    plt.subplot(1, 2, 2, facecolor='aliceblue')

    

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    roc_auc = auc(fpr, tpr)



    plt.title('ROC Curve : ' + title)

    plt.plot(fpr, tpr, 'r', label='AUC = %0.3f' % roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'b--')

    plt.xlim([-0.01, 1.01])

    plt.ylim([-0.01, 1.01])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    

    plt.show()
def print_duration(start):

    duration = time.time() - start

    if duration < 60:

        print("{0:.2f}sec".format(duration))

    elif duration < 3600:

        print("{0}min {1:.2f}sec".format(duration//60, duration%60))

    else:

        print("{0}hour {1}min {2:.2f}sec".format(duration//3600, (duration%3600)//60, (duration%3600)%60))
def run_classification(X_train, X_valid, y_train, y_valid, model_name, churn_type, f5=""):

    clf = create_model(models[model_name])

    print("Training Duration : ", end="")

    start = time.time()

    clf = train_model(clf, X_train, y_train)

    print_duration(start)

    

    print("Predicting Training Data Duration : ", end="")

    start = time.time()

    train_prediction = predict_result(clf, X_train)

    print_duration(start)

    

    print("Predicting Testing Data Duration : ", end="")

    start = time.time()

    prediction = predict_result(clf, X_valid)

    print_duration(start)

    print("\n")

    evaluate(y_train, train_prediction, 'Training {} Churn {} using {} Model'.format(churn_type, f5, model_name))

    evaluate(y_valid, prediction, 'Validation {} Churn {} using {} Model'.format(churn_type, f5, model_name))
def evaluate(y_valid, y_predicted, title):

    metric_score(y_valid, y_predicted, title)

    plot_confusion_matrix(y_valid, y_predicted, title)
models = {

    'BernoulliNB_hard': {

        'model' : BernoulliNB,

        'params' : {'alpha': 0.23112322386736384,

                         'binarize': 0.48805747021059187,

                         'class_prior': None,

                         'fit_prior': True}

    },

    'BernoulliNB_soft': {

        'model' : BernoulliNB,

        'params' : {'alpha': 0.48760930442730993,

                         'binarize': 0.5157549656481472,

                         'class_prior': None,

                         'fit_prior': True}

    },  

    'SVC_Polynomial_hard': {

        'model' : SVC,

        'params' : {'C': 0.00045,

                     'coef0': 0.1,

                     'degree': 3,

                     'kernel': 'poly',

                     'max_iter': -1,

                     'random_state': 0,

                     'shrinking': False}

    },

    'SVC_Polynomial_soft': {

        'model' : SVC,

        'params' : {'C': 0.05,

                     'coef0': 0.1,

                     'degree': 3,

                     'kernel': 'poly',

                     'max_iter': -1,

                     'random_state': 0,

                     'shrinking': False}

    },

    'RandomForestClassifier_hard': {

        'model' : RandomForestClassifier,

        'params' : {

            'n_estimators': 124,

            'criterion': 'entropy',

            'min_samples_leaf': 2,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'RandomForestClassifier_soft': {

        'model' : RandomForestClassifier,

        'params' : {

            'n_estimators': 153,

            'criterion': 'entropy',

            'min_samples_leaf': 2,

            'max_features': 'log2',

            'random_state': 0

        }

    },    

    'ExtraTreesClassifier_hard': {

        'model' : ExtraTreesClassifier,

        'params' : {

            'n_estimators': 179,

            'criterion': 'entropy',

            'min_samples_leaf': 2,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier_soft': {

        'model' : ExtraTreesClassifier,

        'params' : {

            'n_estimators': 145,

            'criterion': 'entropy',

            'min_samples_leaf': 1,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'AdaBoostClassifier_hard': {

        'model' : AdaBoostClassifier,

        'params' : {

            'n_estimators': 191,

            'learning_rate': 0.9824531051022739,

            'random_state': 0

        }

    },

    'AdaBoostClassifier_soft': {

        'model' : AdaBoostClassifier,

        'params' : {

            'n_estimators': 200,

            'learning_rate': 0.9774069535137836,

            'random_state': 0

        }

    },

    'GradientBoostingClassifier_hard': {

        'model' : GradientBoostingClassifier,

        'params' : {

            'n_estimators': 198,

            'learning_rate': 0.6841850952084808,

            'subsample': 0.9,

            'min_samples_leaf': 3,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'GradientBoostingClassifier_soft': {

        'model' : GradientBoostingClassifier,

        'params' : {

            'n_estimators': 166,

            'learning_rate': 0.7479432334085875,

            'subsample': 0.9,

            'min_samples_leaf': 1,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'RidgeClassifier_hard': {

        'model' : RidgeClassifierCV,

        'params' : {

            'alphas' : [0.01, 0.1, 1.0, 10.0, 100.0]

        }

    },

    'RidgeClassifier_soft': {

        'model' : RidgeClassifierCV,

        'params' : {

            'alphas' : [0.01, 0.1, 1.0, 10.0, 100.0]

        }

    },

    'LogisticRegression_hard': {

        'model' : LogisticRegressionCV,

        'params' : {

            'random_state': 0,

            'Cs' : 10

        }

    },

    'LogisticRegression_soft': {

        'model' : LogisticRegressionCV,

        'params' : {

            'random_state': 0,

            'Cs' : 10

        }

    },

    'SGDClassifier_hard': {

        'model' : SGDClassifier,

        'params' : {

            'loss' : 'huber',

            'penalty' : 'elasticnet',

            'eta0' : 0.5

        }

    },

    'SGDClassifier_soft': {

        'model' : SGDClassifier,

        'params' : {

            'loss' : 'huber',

            'penalty' : 'elasticnet',

            'eta0' : 0.5

        }

    },

    'Perceptron_hard': {

        'model' : Perceptron,

        'params' : {

            'random_state' : 0,

        }

    },

    'Perceptron_soft': {

        'model' : Perceptron,

        'params' : {

            'random_state' : 0

        }

    },

    'PassiveAggressiveClassifier_hard': {

        'model' : PassiveAggressiveClassifier,

        'params' : {

            'random_state' : 0

        }

    },

    'PassiveAggressiveClassifier_soft': {

        'model' : PassiveAggressiveClassifier,

        'params' : {

            'random_state' : 0

        }

    },

    'MLPClassifier_hard' : {

        'model' : MLPClassifier,

        'params' : {

            'activation': 'relu', 

            'alpha': 0.0001, 

            'hidden_layer_sizes': (50, 50), 

            'learning_rate': 'adaptive', 

            'solver': 'adam'

        }

    },

    'MLPClassifier_soft' : {

        'model' : MLPClassifier,

        'params' : {

            'activation': 'tanh', 

            'alpha': 0.0001, 

            'hidden_layer_sizes': (50, 100), 

            'learning_rate': 'constant', 

            'solver': 'adam'

        }

    },

    'KNeighborsClassifier_hard' : {

        'model' : KNeighborsClassifier,

        'params' : {

            'metric': 'manhattan', 

            'n_neighbors': 10, 

            'weights': 'distance'

        }

    },

    'KNeighborsClassifier_soft' : {

        'model' : KNeighborsClassifier,

        'params' : {

            'metric': 'manhattan', 

            'n_neighbors': 10, 

            'weights': 'distance'

        }

    }

}

std_scaler = StandardScaler()

mmax_scaler = MinMaxScaler()
hard_churn_df = raw_dataframes[0]

churn_type = 'Hard'

hard_churn_df.head()
X,y = preprocess_data(hard_churn_df, ['imei_name'])

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = .5)



#removing imei_name (irrelevant data for train) and feature_5 (too high correlation)

X_wo5, y_wo5 = preprocess_data(hard_churn_df, ['imei_name', 'feature_5'])

X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid = train_test_split(X_wo5,y_wo5, test_size = .5)
#MinMax Scaled

X_train_m = mmax_scaler.fit_transform(X_train)

X_valid_m = mmax_scaler.transform(X_valid)



X_wo5_train_m = mmax_scaler.fit_transform(X_wo5_train)

X_wo5_valid_m = mmax_scaler.transform(X_wo5_valid)



#Standard Scaled

X_train_s = std_scaler.fit_transform(X_train)

X_valid_s = std_scaler.transform(X_valid)



X_wo5_train_s = std_scaler.fit_transform(X_wo5_train)

X_wo5_valid_s = std_scaler.transform(X_wo5_valid)
#All Features

run_classification(X_train_m, X_valid_m, y_train, y_valid, 'BernoulliNB_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_m, X_wo5_valid_m, y_wo5_train, y_wo5_valid, 'BernoulliNB_hard', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'SVC_Polynomial_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'SVC_Polynomial_hard', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'RandomForestClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'RandomForestClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'ExtraTreesClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'ExtraTreesClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'AdaBoostClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'AdaBoostClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'GradientBoostingClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'GradientBoostingClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'RidgeClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'RidgeClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'LogisticRegression_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'LogisticRegression_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_m, X_valid_m, y_train, y_valid, 'SGDClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_m, X_wo5_valid_m, y_wo5_train, y_wo5_valid, 'SGDClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'Perceptron_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'Perceptron_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_m, X_valid_m, y_train, y_valid, 'PassiveAggressiveClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_m, X_wo5_valid_m, y_wo5_train, y_wo5_valid, 'PassiveAggressiveClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'MLPClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'MLPClassifier_hard', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'KNeighborsClassifier_hard', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'KNeighborsClassifier_hard', churn_type, "sans F5")
soft_churn_df = raw_dataframes[1]

churn_type = 'Soft'

soft_churn_df.head()
X,y = preprocess_data(soft_churn_df, ['imei_name'])

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size = .5)



#removing imei_name (irrelevant data for train) and feature_5 (too high correlation)

X_wo5, y_wo5 = preprocess_data(soft_churn_df, ['imei_name', 'feature_5'])

X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid = train_test_split(X_wo5,y_wo5, test_size = .5)
#MinMax Scaled

X_train_m = mmax_scaler.fit_transform(X_train)

X_valid_m = mmax_scaler.transform(X_valid)



X_wo5_train_m =mmax_scaler.fit_transform(X_wo5_train)

X_wo5_valid_m = mmax_scaler.transform(X_wo5_valid)



#Standard Scaled

X_train_s = std_scaler.fit_transform(X_train)

X_valid_s = std_scaler.transform(X_valid)



X_wo5_train_s = std_scaler.fit_transform(X_wo5_train)

X_wo5_valid_s = std_scaler.transform(X_wo5_valid)
#All Features

run_classification(X_train_m, X_valid_m, y_train, y_valid, 'BernoulliNB_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_m, X_wo5_valid_m, y_wo5_train, y_wo5_valid, 'BernoulliNB_soft', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'SVC_Polynomial_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'SVC_Polynomial_soft', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'RandomForestClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'RandomForestClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'ExtraTreesClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'ExtraTreesClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'AdaBoostClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'AdaBoostClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train, X_valid, y_train, y_valid, 'GradientBoostingClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train, X_wo5_valid, y_wo5_train, y_wo5_valid, 'GradientBoostingClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'RidgeClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'RidgeClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'LogisticRegression_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'LogisticRegression_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_m, X_valid_m, y_train, y_valid, 'SGDClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_m, X_wo5_valid_m, y_wo5_train, y_wo5_valid, 'SGDClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'Perceptron_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'Perceptron_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_m, X_valid_m, y_train, y_valid, 'PassiveAggressiveClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_m, X_wo5_valid_m, y_wo5_train, y_wo5_valid, 'PassiveAggressiveClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'MLPClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'MLPClassifier_soft', churn_type, "sans F5")
#All Features

run_classification(X_train_s, X_valid_s, y_train, y_valid, 'KNeighborsClassifier_soft', churn_type, "all F")
#Without Feature 5

run_classification(X_wo5_train_s, X_wo5_valid_s, y_wo5_train, y_wo5_valid, 'KNeighborsClassifier_soft', churn_type, "sans F5")