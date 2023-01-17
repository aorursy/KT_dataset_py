import numpy as np

import pandas as pd

import os

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

import torch

import random

import copy

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier



# Seeding#####################################################################

def random_seed(seed_value, use_cuda):

    """

    Parameters

    ----------

    seed_value : int

        The desired seed number

    use_cuda : Boolean

        True if GPU seeding is desired, False, otherwise

    Returns

    -------

    None.

    """

    np.random.seed(seed_value) # numpy seed

    torch.manual_seed(seed_value) # torch cpu vars

    random.seed(seed_value) # Python seed

    if use_cuda:

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # torch gpu vars

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False



random_seed(14, True)

##############################################################################
def create_folds(data, target, k):

    """

    Basen on a function from the book: Approaching (Almost) Any Machine Learning Problem by Abhishek Thakur

    Parameters

    ----------

    data : pandas DataFrame

        the dataframe containing the data

    target : str

        The column holding the ground truth

    k : int

        number of folds

    Returns

    -------

    data : pandas DataFrame

        a dataframe containing the data with a new column for fold number

    """

    # Create a new column called kfold and fill it woth -1

    data["kfold"] = -1



    # Randomize rows of the data

    data = data.sample(frac=1).reset_index(drop=True)



    # fetch targets

    y = data[target].values



    # initiate the kfold class from model_selection module

    kf = StratifiedKFold(n_splits=k)



    # fill the new kfold column

    for f, (t_, v_) in enumerate(kf.split(X=data, y=y)):

        data.loc[v_, "kfold"] = f



    return data
def load_data(root_dir):

    """

    Parameters

    ----------

    root_dir : str

        path to the directory in which Data and Codes are placed

    Returns

    -------

    folded_train_spreadsheet : DataFrame

    test_spreadsheet : DataFrame

    test_label_spreadsheet : DataFrame

    """

    train_data_path = os.path.join(root_dir, "train.csv")

    test_data_path = os.path.join(root_dir, "test.csv")

    test_label_path = os.path.join(root_dir, "test_GT.csv")

    train_spreadsheet = pd.read_csv(train_data_path)

    test_spreadsheet = pd.read_csv(test_data_path)

    test_label_spreadsheet = pd.read_csv(test_label_path)

    print("train size:", train_spreadsheet.shape[0])

    print("test size:", test_spreadsheet.shape[0])

    print("class_0 size(within the train set):", sum(train_spreadsheet["Mutacion"]==0))

    print("class_1 size(within the train set):", sum(train_spreadsheet["Mutacion"]==1))

    folded_train_spreadsheet = create_folds(train_spreadsheet, "Mutacion", 5)

    return folded_train_spreadsheet, test_spreadsheet, test_label_spreadsheet
root_dir = "/kaggle/input/radiomics-for-lgg-dataset/"

folded_train_spreadsheet, test_spreadsheet, test_label_spreadsheet = load_data(root_dir)
def X_y_of_trainset(df, target):

    """

    Parameters

    ----------

    df : pandas DataFrame

        the train dataframe containing radiomics+patientID+kfold+target(labels)

    target : str

        target(labels) column name

    Returns

    -------

    x_train : Numpy Array

        radiomics features of the train set

    y_train : Numpy Array

        ground truth for the train set

    """

    # drop the label column from dataframe and convert it to a 

    # numpy array by using .values

    # target is the label column in the dataframe

    x_train = df.drop([target, "patientID", "kfold"], axis=1).values

    y_train = df[target].values



    return x_train, y_train



def X_of_test(df):

    """

    Parameters

    ----------

    df : pandas DataFrame

        the test dataframe containing radiomics+patientID

    Returns

    -------

    x_test : Numpy Array

        radiomics features of the test set

    ids : Numpy Array

        PatientIDs

    """

    # drop the label column from dataframe and convert it to a 

    # numpy array by using .values

    # target is the label column in the dataframe

    x_test = df.drop("patientID", axis=1).values

    ids = df["patientID"].values



    return x_test, ids





def y_of_test(df):

    """

    Parameters

    ----------

    df : pandas DataFrame

        the test dataframe containing radiomics+patientID

    Returns

    -------

    y_test : Numpy Array

        ground truth for the test set

    ids : Numpy Array

        PatientIDs

    """

    # drop the label column from dataframe and convert it to a 

    # numpy array by using .values

    # target is the label column in the dataframe

    y_test = df.drop("patientID", axis=1).values

    ids = df["patientID"].values



    return y_test, ids
    # creating train/test sets

    x_train, y_train = X_y_of_trainset(folded_train_spreadsheet, "Mutacion")

    x_test, ids = X_of_test(test_spreadsheet)

    y_test, _ = y_of_test(test_label_spreadsheet)
cv = 5 # number of folds
def ExtremelyRandomizedTrees(x_train, y_train, cv):

    # Extremely Randomized Trees

    et_clf = ExtraTreesClassifier()

    et_clf_param_grid = {

        'n_estimators':[10, 15, 20, 25, 50, 80, 90, 100, 150, 200, 250, 300],

        'max_features':['auto', 'sqrt', 'log2']}

    et_clfs = GridSearchCV(estimator=et_clf, param_grid=et_clf_param_grid, cv=cv)

    et_clfs.fit(x_train, y_train)

    et_clf = ExtraTreesClassifier(n_estimators=et_clfs.best_params_['n_estimators'],

                                  max_features=et_clfs.best_params_['max_features'])

    et_clf.fit(x_train, y_train)

    return et_clf, et_clf_param_grid
def evaluate(clf, x_test, y_test):

    """

    Parameters

    ----------

    clf : classifier

    x_test : Numpy Array

        radiomics features of the test set

    y_test : Numpy Array

        ground truth for the test set

    Returns

    -------

    acc : float

        Accuracy

    """

    preds = clf.predict(x_test)

    acc = accuracy_score(y_test, preds)

    auc = roc_auc_score(y_test, preds)

    return acc, auc
# Extremely Randomized Trees

et_clf, et_clf_param_grid = ExtremelyRandomizedTrees(x_train, y_train, cv)

et_clf.fit(x_train, y_train)

acc, auc = evaluate(et_clf, x_test, y_test)

print("Classifier: Extremely Randomized Trees ", "Test Accuracy:", acc, "Test AUC:", auc)
def X_y_of_fold(df, target, fold):

    """

    Parameters

    ----------

    df : Pandas DataFrame

        The DataFrame Containing the Radiomics Features and 

    target : str

        target(labels) column name

    fold : ind

        fold index

    Returns

    -------

    x_train : Numpy Array

        radiomics features for training set of the specified fold

    y_train : Numpy Array

        ground truth for the training set of the specified fold

    x_valid : Numpy Array

        radiomics features for validation set of the specified fold

    y_valid : Numpy Array

        ground truth for the validation set of the specified fold

    """

    # training data is where kfold is not equal to provided fold

    # also note that we reset the index

    df_train = df[df.kfold!=fold].reset_index(drop=True)



    # validation data is where kfold is equal to provided fold

    df_valid = df[df.kfold==fold].reset_index(drop=True)



    # drop the label column from dataframe and convert it to a 

    # numpy array by using .values

    # target is the label column in the dataframe

    x_train = df_train.drop([target, "patientID", "kfold"], axis=1).values

    y_train = df_train[target].values



    # similarly fo validation we have

    x_valid = df_valid.drop([target, "patientID", "kfold"], axis=1).values

    y_valid = df_valid[target].values



    return x_train, y_train, x_valid, y_valid

def run(clf, folded_train_df, fold):

    x_train, y_train, x_valid, y_valid = X_y_of_fold(folded_train_df, "Mutacion", fold)

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    probs = clf.predict_proba(x_valid)

    accuracy = accuracy_score(y_valid, preds)

    AUC = roc_auc_score(y_valid, probs[:,1])

    print(f"Fold={fold}, Accuracy={accuracy}, , AUC={AUC}")

    return clf, accuracy
def voting(clf_list, X, weights):

    pred = np.asarray([clf.predict(X) for clf in clf_list]).T

    pred = np.apply_along_axis(lambda x:np.argmax(np.bincount(x, weights=weights)),

                               axis=1,

                               arr=pred.astype('int'))

    return pred
# Ensemble

models = []

model_weights = []

classifiers = {"ExtremelyRandomizedTree":et_clf}

for classifier in classifiers:

    print(classifier)

    for fold in range(cv):

        clf, weight = run(copy.deepcopy(classifiers[classifier]), folded_train_spreadsheet, fold)

        models.append(clf)

        model_weights.append(weight)



preds = voting(models, x_test, model_weights)

print("Classifier: Ensembled ", "Test Accuracy:", accuracy_score(y_test, preds))