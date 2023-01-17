import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score, classification_report



# Load the images and the labels

train_raw = pd.read_csv("../input/x_train_gr_smpl.csv")

labels = pd.read_csv("../input/y_train_smpl.csv")

train_labels = pd.read_csv("../input/y_train_smpl.csv")



test_raw = pd.read_csv("../input/x_test_gr_smpl.csv")

test_label = labels = pd.read_csv("../input/y_test_smpl.csv")





# Display top 2 rows for each dataframe

display(train_raw.head(2))

display(labels.head(2))





# Train contains the image data and labels in the same table.

data_train = pd.read_csv("../input/x_train_gr_smpl.csv")

data_train['label'] = labels

# Train contains the image data and labels in the same table.

data_test = pd.read_csv("../input/x_test_gr_smpl.csv")

data_test['label'] = test_label
label_paths_train = [

    "../input/y_train_smpl_0.csv",

    "../input/y_train_smpl_1.csv",

    "../input/y_train_smpl_2.csv",

    "../input/y_train_smpl_3.csv",

    "../input/y_train_smpl_4.csv",

    "../input/y_train_smpl_5.csv",

    "../input/y_train_smpl_6.csv",

    "../input/y_train_smpl_7.csv",

    "../input/y_train_smpl_8.csv",

    "../input/y_train_smpl_9.csv",

]



label_paths_test = [

    "../input/y_test_smpl_0.csv",

    "../input/y_test_smpl_1.csv",

    "../input/y_test_smpl_2.csv",

    "../input/y_test_smpl_3.csv",

    "../input/y_test_smpl_4.csv",

    "../input/y_test_smpl_5.csv",

    "../input/y_test_smpl_6.csv",

    "../input/y_test_smpl_7.csv",

    "../input/y_test_smpl_8.csv",

    "../input/y_test_smpl_9.csv",

]




import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from prettytable import PrettyTable

from keras import optimizers

import collections

from keras_tqdm import TQDMNotebookCallback, TQDMCallback



# Sets a threshold to the data and rounds it to 0 or 1. Useful for some metrics.

def set_threshold(data, threshold = 0.5):

    rounded = np.array([1 if x >= threshold else 0 for x in data])

    return rounded



# Function we can use to load labels

def get_label (index=0, paths_array=label_paths_train):

    return pd.read_csv(paths_array [index])





def run_metrics(y_true, y_pred, threshold):

    y_pred_rounded = set_threshold(y_pred, threshold)



    roc_score = round(roc_auc_score(y_true, y_pred), 4)

    f_score = round(f1_score(y_true, y_pred_rounded), 4)

    recall = round(recall_score(y_true, y_pred_rounded), 4)

    precision = round(precision_score(y_true, y_pred_rounded), 4)

    accuracy = round(accuracy_score(y_true, y_pred_rounded), 4)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_rounded).ravel()

    

    

    return {

        "roc_score": roc_score,

        "f_score": f_score,

        "recall": recall,

        "precision": precision,

        "accuracy": accuracy,

        "TP": tp,

        "FP": fp}



def run_metrics_all(predictions_all, file_category, threshold = 0.5, labels_idx=[]):

    x = PrettyTable()

    x.field_names = ["", "Roc_AUC_score", "f score", "recall", "precision", "accuracy", "TP", "FP"]

    for i, (columnName, y_pred) in enumerate(predictions_all.iteritems()):

        if file_category == "test":

            y_true = get_label(index=i, paths_array=label_paths_test)

        else :

            y_true = get_label(index=i, paths_array=label_paths_train)

        

        if len(labels_idx) != 0:

            y_true = y_true.loc[labels_idx]

            

        metrics = run_metrics(y_true, y_pred, threshold=threshold)

        x.add_row([columnName, metrics["roc_score"], metrics["f_score"],

                   metrics["recall"], metrics["precision"], metrics["accuracy"], metrics["TP"], metrics["FP"]])

        

    print(x)




from sklearn.model_selection import KFold



# Initialize 10 fold cross validation

n_splits = 10

random_state = np.random.seed(22135)



kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)



from sklearn.metrics import confusion_matrix

# Slower than expected because we load the file at every run



def random_forest_kfold(n_estimators, max_depth, max_leaf_nodes, min_impurity_decrease ):

    results = pd.DataFrame()

    test_preds = pd.DataFrame()

    features = []

    # Supervised transformation based on random forests

    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, 

                                max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease)

    

    

    for i in range(len(label_paths_test)):

        print("\nRUNNING NET FOR y_train_smpl_{}".format(i))

        # oof -> Out of fold. One single vector with all the validation predictions to

        # then calculate the error upon this predictions.

        oof = np.zeros(data_test.shape[0])

        predictions_test = np.zeros(data_test.shape[0])

        print()

        

        target = get_label(index=i, paths_array=label_paths_test)    



        # K-fold CV

        for epoch, (train_index, val_index) in enumerate(kf.split(data_test.values)):

            #print("\nFOLD {}".format(epoch + 1))

            X_train, X_val = data_test.loc[train_index], data_test.loc[val_index]

            y_train, y_val = target.loc[train_index], target.loc[val_index]

            

            y_train = np.ravel(y_train)

            y_val = np.ravel(y_val)

                        

            rf.fit(X_train, y_train)

            

            # y_pred_rf = label_binarize(y_pred_rf, classes=[0, 1, 2, 3, 4, 5 ,6 ,7 ,8 ,9])

            # y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5 ,6 ,7 ,8 ,9])

            # roc_score = round(roc_auc_score(y_test, y_pred_rf), 4)

            

            # Predict on Val data

            prediction = rf.predict(X_val.astype('float32')).squeeze()            

            features.append(rf.feature_importances_)

            oof[val_index] = prediction

            predictions_test += (rf.predict(data_test)/kf.n_splits).squeeze()

            print("split ", + epoch)

            print(run_metrics(y_val, prediction, threshold=0.5))

            

        test_preds["y_train_smpl_{}".format(i)] = predictions_test 

        results["y_train_smpl_{}".format(i)] = oof 

        # print(run_metrics(target, predictions_test, threshold=0.5))

        

    return results, test_preds, features





predictions, predictions_test, feature_kfold = random_forest_kfold(n_estimators = 100, max_depth = 1000, max_leaf_nodes = 1000, min_impurity_decrease =0)





run_metrics_all(predictions, "test")

def random_forest_split(n_estimators, max_depth, max_leaf_nodes, min_impurity_decrease ):

    # Slower than expected because we load the file at every run

    

    print("Random Forest classifier with train and input files as input")

    results = pd.DataFrame()

    test_preds = pd.DataFrame()    

    # Supervised transformation based on random forests

    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, 

                                max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease)

    

    features = []

    

    for i in range(len(label_paths_train)):

        print("\nRUNNING NET FOR y_train_smpl_{}".format(i))

        # oof -> Out of fold. One single vector with all the validation predictions to

        # then calculate the error upon this predictions.

        oof = np.zeros(test_raw.shape[0])

        predictions_test = np.zeros(test_raw.shape[0])

        print()

        

        target = get_label(index=i, paths_array=label_paths_train)   

        y_train = np.ravel(target)

        

        rf.fit(train_raw, y_train)

        

        

        target_test = get_label(index=i, paths_array=label_paths_test)   

            

        # Predict on Val data

        prediction = rf.predict(test_raw.astype('float32')).squeeze()

        features.append(rf.feature_importances_)

        oof = prediction

        predictions_test += (rf.predict(test_raw)).squeeze()

            

        test_preds["y_train_smpl_{}".format(i)] = predictions_test 

        results["y_train_smpl_{}".format(i)] = oof 

        print(run_metrics(target_test, prediction, threshold=0.5))

        

    return results, test_preds, features

predictions_split, predictions_test_split , features_file = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =0)



run_metrics_all(predictions_split, "test")
def extract_top_indices(my_array, n_indices):

    # This gets the indices for the top N highest values in my_array

    feature_idx = my_array.argsort()[-n_indices:][::-1]

    # convert the indices to string to match column names in the pandas dataframe

    str_columns = [str(i) for i in feature_idx]

    

    return str_columns
npfeatures = np.array(features_file)

for i in range(len(npfeatures)):

    

    top_features = extract_top_indices(npfeatures[i], 10)

    print(top_features)
predictions_split, predictions_test_split , features_split = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =0)

run_metrics_all(predictions_split, "test")
predictions_split, predictions_test_split , features_split = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =1)

run_metrics_all(predictions_split, "test")
predictions_split, predictions_test_split , features_split = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =0)

run_metrics_all(predictions_split, "test")
predictions_split, predictions_test_split , features_split = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =.0001)

run_metrics_all(predictions_split, "test")
predictions_split, predictions_test_split , features_split = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =0.001)

run_metrics_all(predictions_split, "test")
predictions_split, predictions_test_split , features_split = random_forest_split(n_estimators = 100, max_depth = 100, max_leaf_nodes = 100, min_impurity_decrease =0.05)

run_metrics_all(predictions_split, "test")
from sklearn.model_selection import train_test_split





# 4000 values out of 12659 => 0.316

# 9000 values out of 12659 => 0.711



def random_forest_train_instances(n_instances, n_estimators, max_depth, max_leaf_nodes, min_impurity_decrease ):

    print("Random Forest classifier with part of the train file as input")

    results = pd.DataFrame()

    test_preds = pd.DataFrame()    

    # Supervised transformation based on random forests

    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, 

                                max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease)

    

    features = []

    ratio = n_instances/train_raw.shape[0]

    for i in range(len(label_paths_train)):

        print("\nRUNNING NET FOR y_train_smpl_{}".format(i))

        # oof -> Out of fold. One single vector with all the validation predictions to

        # then calculate the error upon this predictions.

        

        target = get_label(index=i, paths_array=label_paths_train)   

        

        x_train, x_test, y_train, y_test = train_test_split(train_raw, target, test_size=ratio, shuffle=True, random_state=3) 

        

        oof = np.zeros(x_test.shape[0])

        predictions_test = np.zeros(x_test.shape[0])

        print()

               

                

        rf.fit(x_train, y_train)

                    

        # Predict on Val data

        prediction = rf.predict(x_test.astype('float32')).squeeze()

        features.append(rf.feature_importances_)

        oof = prediction

        predictions_test += (rf.predict(x_test)).squeeze()

            

        test_preds["y_train_smpl_{}".format(i)] = predictions_test 

        results["y_train_smpl_{}".format(i)] = oof 

        print(run_metrics(y_test, prediction, threshold=0.5))

        

    

        

    return results, test_preds, features
rf = RandomForestClassifier()

ratio = 2000/train_raw.shape[0]

x_train, x_test, y_train, y_test = train_test_split(train_raw, train_labels, test_size=ratio, shuffle=True, random_state=3) 

rf.fit(x_train, y_train)

prediction = rf.predict(x_test)



from sklearn.metrics import confusion_matrix



# Create confusion matrix from test data

conf_matrix_data = confusion_matrix(y_true=y_test, y_pred=prediction)



# Make confusion matrix with colors

plt.figure(figsize=(10,5)) # Size of the plot

ax = sns.heatmap(conf_matrix_data, annot=True) # Creating the colors
rf = RandomForestClassifier(max_depth = 9)

ratio = 9000/train_raw.shape[0]

x_train, x_test, y_train, y_test = train_test_split(train_raw, train_labels, test_size=ratio, shuffle=True, random_state=3) 

rf.fit(x_train, y_train)

prediction = rf.predict(x_test)



from sklearn.metrics import confusion_matrix



# Create confusion matrix from test data

conf_matrix_data = confusion_matrix(y_true=y_test, y_pred=prediction)



# Make confusion matrix with colors

plt.figure(figsize=(10,5)) # Size of the plot

ax = sns.heatmap(conf_matrix_data, annot=True) # Creating the colors
predictions_file, predictions_test_file , features_file = random_forest_train_instances(2000, n_estimators = 100, max_depth = 1000, max_leaf_nodes = 1000, min_impurity_decrease =0)
