#Suppress Deprecation and Incorrect Usage Warnings 

import pandas as pd

import numpy as np

import warnings

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support

from scipy import interp

import pickle

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/bcwd.csv")

df.head()
df['diagnosis'].value_counts()
df.isna().sum()
df.loc[:,'radius_mean':'fractal_dimension_mean'].boxplot(figsize=(20,5))

plt.show()
df.loc[:,'radius_se':'fractal_dimension_se'].boxplot(figsize=(20,5))

plt.show()
df.loc[:,'radius_worst':'fractal_dimension_worst'].boxplot(figsize=(20,5))

plt.show()
mapper = {'M': 1, 'B': 0}

df['diagnosis'] = df['diagnosis'].replace(mapper)

df['diagnosis'].value_counts()
# also store a list with the names of all predictors

names_all = [c for c in df if c not in ['diagnosis']]



# define column groups with same data preparation

names_outliers = ['area_mean', 'area_se', 'area_worst']

names_no_outliers = list(set(names_all) - set(names_outliers))
class AddColumnNames(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return pd.DataFrame(data=X, columns=self.columns)
class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)



        try:

            return X[self.columns]

        except KeyError:

            cols_error = list(set(self.columns) - set(X.columns))

            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
preprocess_pipeline = make_pipeline(

    AddColumnNames(columns=names_all),

    FeatureUnion(transformer_list=[

        ("outlier_columns", make_pipeline(

            ColumnSelector(columns=names_outliers),

            FunctionTransformer(func=np.log, validate=False),

            RobustScaler()

        )),

        ("no_outlier_columns", make_pipeline(

            ColumnSelector(columns=names_no_outliers),

            StandardScaler()

        ))

    ])

)
y = df['diagnosis']

X = df.drop('diagnosis', axis=1).values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
# create the pipeline

pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('svm', svm.SVC(probability=True))])



# prepare a prameter grid

# note that __ can be used to specify the name of a parameter for a specific element in a pipeline

# note also that this is not an exhaustive list of the parameters of SVM and their possible values



param_grid = {

    'svm__C': [0.1, 1, 10, 100],  

    'svm__gamma': [1, 0.1, 0.01, 0.001], 

    'svm__kernel': ['rbf', 'linear', 'poly']}



warnings.filterwarnings('ignore')

search = GridSearchCV(pipe, param_grid, cv=10, refit=True)

search.fit(X_train, y_train)

print("Best CV score = %0.3f:" % search.best_score_)

print("Best parameters: ", search.best_params_)



# store the best params and best model for later use

SVM_best_params = search.best_params_

SVM_best_model = search.best_estimator_
# create the pipeline

pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('rf', RandomForestClassifier())])



# prepare a prameter grid

# note that __ can be used to specify the name of a parameter for a specific element in a pipeline

# note also that this is not an exhaustive list of the parameters of Random Forest and their possible values

param_grid = {

    'rf__n_estimators' : [10,20,30],

    'rf__max_depth': [2, 4, 6, 8]

}



warnings.filterwarnings('ignore')

search = GridSearchCV(pipe, param_grid, cv=10, refit=True)

search.fit(X_train, y_train)

print("Best CV score = %0.3f:" % search.best_score_)

print("Best parameters: ", search.best_params_)



# store the best params and best model for later use

RF_best_params = search.best_params_

RF_best_model = search.best_estimator_
mean_fpr = np.linspace(start=0, stop=1, num=100)
# model - a trained binary probabilistic classification model;

#         it is assumed that there are two classes: 0 and 1

#         and the classifier learns to predict probabilities for the examples to belong to class 1



def evaluate_model(X_test, y_test, model):

    # compute probabilistic predictiond for the evaluation set

    _probabilities = model.predict_proba(X_test)[:, 1]

    

    # compute exact predictiond for the evaluation set

    _predicted_values = model.predict(X_test)

        

    # compute accuracy

    _accuracy = accuracy_score(y_test, _predicted_values)

        

    # compute precision, recall and f1 score for class 1

    _precision, _recall, _f1_score, _ = precision_recall_fscore_support(y_test, _predicted_values, labels=[1])

    

    # compute fpr and tpr values for various thresholds 

    # by comparing the true target values to the predicted probabilities for class 1

    _fpr, _tpr, _ = roc_curve(y_test, _probabilities)

        

    # compute true positive rates for the values in the array mean_fpr

    _tpr_transformed = np.array([interp(mean_fpr, _fpr, _tpr)])

    

    # compute the area under the curve

    _auc = auc(_fpr, _tpr)

            

    return _accuracy, _precision[0], _recall[0], _f1_score[0], _tpr_transformed, _auc
SVM_accuracy, SVM_precision, SVM_recall, SVM_f1_score, SVM_tpr, SVM_auc = evaluate_model(X_test, y_test, SVM_best_model)

RF_accuracy, RF_precision, RF_recall, RF_f1_score, RF_tpr, RF_auc = evaluate_model(X_test, y_test, RF_best_model)
SVM_metrics = np.array([SVM_accuracy, SVM_precision, SVM_recall, SVM_f1_score])

RF_metrics = np.array([RF_accuracy, RF_precision, RF_recall, RF_f1_score])

index = ['accuracy', 'precision', 'recall', 'F1-score']

df_metrics = pd.DataFrame({'SVM': SVM_metrics, 'Random Forest': RF_metrics}, index=index)

df_metrics.plot.bar(rot=0)

plt.legend(loc="lower right")

plt.show()
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)

plt.plot(mean_fpr, SVM_tpr[0,:], lw=2, color='blue', label='SVM (AUC = %0.2f)' % (SVM_auc), alpha=0.8)

plt.plot(mean_fpr, RF_tpr[0,:], lw=2, color='orange', label='Random Forest (AUC = %0.2f)' % (RF_auc), alpha=0.8)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC curves for multiple classifiers')

plt.legend(loc="lower right")

plt.show()
# function to remove the string 'svm__' from the names of the parameters in SVM_best_params

def transform(dict):

    return {key.replace('svm__','') :  value for key, value in dict.items()}



pipe = Pipeline(steps=[('preprocess', preprocess_pipeline), ('rf', svm.SVC(**transform(SVM_best_params)))])



final_model =pipe.fit(X, y)
filename = 'final_model.sav'

pickle.dump(final_model, open(filename, 'wb'))