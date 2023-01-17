import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from sklearn.metrics import f1_score

import graphviz

from sklearn import tree

from sklearn.model_selection import KFold
# prettify plots

plt.rcParams['figure.figsize'] = [20.0, 5.0]
!ls ../input/**
%%time

test = pd.read_csv('../input/54-features-datasets-clean-drift-noise-free/54-features-from-test_clean_removed_drift_noise.csv')
test = test.fillna(0.0)
%%time

train = pd.read_csv('../input/54-features-datasets-clean-drift-noise-free/54-features-from-train_clean_removed_drift_noise.csv')
train = train.fillna(0.0)
print("train", train.shape)

print("test", test.shape)
train.head()
test.head()
res = 1000

batch_size=500000

sub_sample_size = batch_size/5

margin=200000



def plot_data(column, column_name):

    plt.figure(figsize=(20,5))

    plt.plot(range(0, column.shape[0], res), column[0::res])

    for i in range(11): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')

    for j in range(10): plt.text(j*batch_size+margin,10,str(j+1),size=20)

    plt.xlabel('Row',size=16); plt.ylabel(column_name,size=16); 

    plt.title(f'Training Data {column_name} - 10 batches',size=20)

    plt.show()
plot_data(train.signal, 'Signal')
plot_data(train.open_channels, 'Open Channels')
columns_to_keep_but_are_not_features = [

    'time',

    'batch_index',

    'signal',

    'open_channels'

]



sixteen_features = ["signal_shift-1","maxtominbatch_msignal","abs_maxbatch","medianbatch","abs_minbatch_slices2_msignal","medianbatch_slices2_msignal",

                    "medianbatch_slices2","abs_avgbatch_slices2_msignal","stdbatch_slices2","mean_abs_chgbatch_slices2_msignal","signal_shift+2_msignal",

                    "signal_shift+1","rangebatch_msignal","abs_maxbatch_slices2","mean_abs_chgbatch_slices2","abs_minbatch"]

feature_cols = list(set(sixteen_features) - set(columns_to_keep_but_are_not_features))

print(f"{len(feature_cols)} features: {feature_cols}")
def chain_to_previous_range(indices: tuple, folds: int):

    return indices[1] + folds, indices[1] + folds

    

def generate_range_of_indices(indices: tuple, fold_size: int, max_allowed_value: int):

    start = fold_size * indices[0]

    end = fold_size * (indices[1] + 1)

    if abs(max_allowed_value - end) <= 3:

        end = max_allowed_value



    return np.array(range(start, end))
def generate_nested_folds_batch_ranges(total_folds_size: int, num_of_training_folds: int = 3,

                                       num_of_validation_folds: int = 1, num_of_test_folds: int = 0):

    total_folds_size = int(total_folds_size)

    total_folds = num_of_training_folds + num_of_validation_folds + num_of_test_folds

    each_fold_size = int(round(total_folds_size / total_folds))



    nested_folds_indices = []

    min_training_index = 0

    max_training_index = total_folds - num_of_validation_folds - num_of_test_folds

    for max_training_index_this_fold in range(min_training_index, max_training_index):

        training_indices = (min_training_index, max_training_index_this_fold)

        validation_indices = chain_to_previous_range(training_indices, num_of_validation_folds)

        test_indices = (0, 0)

        if num_of_test_folds > 0:

            test_indices = chain_to_previous_range(validation_indices, num_of_test_folds)

        nested_folds_indices.append([training_indices, validation_indices, test_indices])



    nested_batch_indices = []

    for each_nested_fold_indices in nested_folds_indices:

        training_indices = each_nested_fold_indices[0]

        validation_indices = each_nested_fold_indices[1]

        test_indices = each_nested_fold_indices[2]



        indices = [

            generate_range_of_indices(training_indices, each_fold_size, total_folds_size),

            generate_range_of_indices(validation_indices, each_fold_size, total_folds_size),

        ]

        if num_of_test_folds > 0:

            indices.append(

                generate_range_of_indices(test_indices, each_fold_size, total_folds_size),

            )

        nested_batch_indices.append(indices)



    return nested_batch_indices
print("training,      validation,        test")

generate_nested_folds_batch_ranges(train.shape[0], 5)
print("training,      validation,        test")

generate_nested_folds_batch_ranges(train.shape[0], 5, 1, 1)
training_folds = 5
def get_kfold_enumerator(dataset: pd.DataFrame, folds: int = training_folds):

    return enumerate(KFold(n_splits=folds).split(dataset))
def get_nestedcv_enumerator(dataset: pd.DataFrame, folds: int = training_folds):

    return enumerate(generate_nested_folds_batch_ranges(dataset.shape[0], folds))
def train_with_cross_validation(params, model, X_train_, y_train_, cv_enumerator=get_nestedcv_enumerator):

    total_f1_macro_score = 0.0

    models = []

    best_model = None

    best_f1_macro_score = 0.0

    

    for fold_index, (training_index, validation_index) in cv_enumerator(X_train_):

        X_training_set = X_train_[training_index]

        y_training_set = y_train_[training_index]

        X_validation_set = X_train_[validation_index]

        y_validation_set = y_train_[validation_index]

        model = model.fit(X_training_set, y_training_set)

        models.append(model)

        predictions = model.predict(X_validation_set)

        f1_macro_score = f1_score(y_validation_set, predictions, average='macro')

        if best_f1_macro_score < f1_macro_score:

            best_f1_macro_score = f1_macro_score

            best_model = model

        print(f'fold {fold_index + 1}: macro f1 validation score: {f1_macro_score}, best macro f1 validation score: {best_f1_macro_score}')

        total_f1_macro_score += f1_macro_score



    return models, best_model, total_f1_macro_score/training_folds



def train_model_by_batch(train_df, feature_cols_, first_batch, second_batch, model_type, 

                         class_names=['0', '1'], params={'max_depth':1}, cv_enumerator=get_nestedcv_enumerator):

    a = batch_size * (first_batch - 1); b = (batch_size * first_batch); 

    c = batch_size * (second_batch - 1); d = (batch_size * second_batch)

    left_batch = train_df[a:b];     right_batch = train_df[a:b];



    X_train = np.concatenate([left_batch[feature_cols_].values, right_batch[feature_cols_].values]).reshape((-1,len(feature_cols)))

    y_train = np.concatenate([left_batch.open_channels.values, left_batch.open_channels.values]).reshape((-1,1))

    

    print(f'Training model {model_type} channel')

    model = tree.DecisionTreeClassifier(**params)

    models, best_model, f1_macro_score = train_with_cross_validation(params, model, X_train, y_train, cv_enumerator=cv_enumerator)

    print(f'model {model_type}, average macro f1 validation score = {f1_macro_score}')

    

    tree_graph = tree.export_graphviz(best_model, out_file=None, max_depth = 10, impurity = False, 

                                      feature_names = feature_cols_, class_names = class_names, rounded = True, filled= True)

    return models, f1_macro_score, graphviz.Source(tree_graph) 
import numpy as np

from sklearn.metrics import f1_score

from tensorflow.keras.callbacks import Callback





def macro_f1(y_true, y_pred):

    """

    The Macro F1 metric used in this competition

    :param y_true: The ground truth labels given in the dataset

    :param y_pred: Our predictions

    :return: The Macro F1 Score

    """

    return f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
nestedcv_f1_macro_scores = []

kfold_f1_macro_scores = []
%%time

nestedcv_clf1s, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 1, 2, '1s', cv_enumerator=get_nestedcv_enumerator)

nestedcv_f1_macro_scores.append(f1_macro_score)

graph
%%time

kfold_clf1s, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 1, 2, '1s', cv_enumerator=get_kfold_enumerator)

kfold_f1_macro_scores.append(f1_macro_score)

graph
%%time

nestedcv_clf1f, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 3, 7, '1f', cv_enumerator=get_nestedcv_enumerator)

nestedcv_f1_macro_scores.append(f1_macro_score)

graph
%%time

kfold_clf1f, f1_macro_score, graph = train_model_by_batch(train,feature_cols, 3, 7, '1f', cv_enumerator=get_kfold_enumerator)

kfold_f1_macro_scores.append(f1_macro_score)

graph
%%time

print("Training using NestedCV cross-validation method")

nestedcv_clf3, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 4, 8, '3', 

                                                            class_names=['0','1','2','3'], params={'max_leaf_nodes': 4}, 

                                                            cv_enumerator=get_nestedcv_enumerator)

nestedcv_f1_macro_scores.append(f1_macro_score)

graph
%%time

print("Training using KFold cross-validation method")

kfold_clf3, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 4, 8, '3', 

                                                         class_names=['0','1','2','3'], params={'max_leaf_nodes': 4}, 

                                                         cv_enumerator=get_kfold_enumerator)

kfold_f1_macro_scores.append(f1_macro_score)

graph
%%time

print("Training using NestedCV cross-validation method")

nestedcv_clf5, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 6, 9, '5', 

                                                            class_names=['0','1','2','3','4','5'], params={'max_leaf_nodes': 6}, 

                                                            cv_enumerator=get_nestedcv_enumerator)

nestedcv_f1_macro_scores.append(f1_macro_score)

graph
%%time

print("Training using KFold cross-validation method")

kfold_clf5, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 6, 9, '5', 

                                                            class_names=['0','1','2','3','4','5'], params={'max_leaf_nodes': 6}, 

                                                           cv_enumerator=get_kfold_enumerator)

kfold_f1_macro_scores.append(f1_macro_score)

graph
%%time

print("Training using NestedCV cross-validation method")

nestedcv_clf10, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 5, 10, '10', 

                                                             class_names=[str(x) for x in range(11)], params={'max_leaf_nodes': 255}, 

                                                             cv_enumerator=get_nestedcv_enumerator)

nestedcv_f1_macro_scores.append(f1_macro_score)

graph
%%time

print("Training using KFold cross-validation method")

kfold_clf10, f1_macro_score, graph = train_model_by_batch(train, feature_cols, 5, 10, '10', 

                                                             class_names=[str(x) for x in range(11)], params={'max_leaf_nodes': 255}, 

                                                             cv_enumerator=get_kfold_enumerator)

kfold_f1_macro_scores.append(f1_macro_score)

graph
%%time

nestedcv_sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

kfold_sub = nestedcv_sub.copy()
"""

Training Batches mapped to sub-model types 

1,  2 ==>  1 Slow Open Channel

3,  7 ==>  1 Fast Open Channel

4,  8 ==>  3 Open Channels

6,  9 ==>  5 Open Channels

5, 10 ==> 10 Open Channels

"""



f1_macro_scores = nestedcv_f1_macro_scores

nestedcv_params = [

    [ (0, 1), "Subsample A",     "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]],

    [ (1, 2), "Subsample B",     "Model 3  (3 Open Channels)",     nestedcv_clf3,  f1_macro_scores[2]],

    [ (2, 3), "Subsample C",     "Model 5  (5 Open Channels)",     nestedcv_clf5,  f1_macro_scores[3]],

    [ (3, 4), "Subsample D",     "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]],

    [ (4, 5), "Subsample E",     "Model 1f (1 Fast Open Channel)", nestedcv_clf1f, f1_macro_scores[1]],

    [ (5, 6), "Subsample F",     "Model 10 (10 Open Channels)",    nestedcv_clf10, f1_macro_scores[4]],

    [ (6, 7), "Subsample G",     "Model 5  (5 Open Channels)",     nestedcv_clf5,  f1_macro_scores[3]],

    [ (7, 8), "Subsample H",     "Model 10 (10 Open Channels)",    nestedcv_clf10, f1_macro_scores[4]],

    [ (8, 9), "Subsample I",     "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]],

    [ (9,10), "Subsample J",     "Model 3  (3 Open Channels)",     nestedcv_clf3,  f1_macro_scores[2]],

    [(10,20), "Batches 3 and 4", "Model 1s (1 Slow Open Channel)", nestedcv_clf1s, f1_macro_scores[0]]

]



f1_macro_scores = kfold_f1_macro_scores

kfold_params = [

    [ (0, 1), "Subsample A",     "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]],

    [ (1, 2), "Subsample B",     "Model 3  (3 Open Channels)",     kfold_clf3,  f1_macro_scores[2]],

    [ (2, 3), "Subsample C",     "Model 5  (5 Open Channels)",     kfold_clf5,  f1_macro_scores[3]],

    [ (3, 4), "Subsample D",     "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]],

    [ (4, 5), "Subsample E",     "Model 1f (1 Fast Open Channel)", kfold_clf1f, f1_macro_scores[1]],

    [ (5, 6), "Subsample F",     "Model 10 (10 Open Channels)",    kfold_clf10, f1_macro_scores[4]],

    [ (6, 7), "Subsample G",     "Model 5  (5 Open Channels)",     kfold_clf5,  f1_macro_scores[3]],

    [ (7, 8), "Subsample H",     "Model 10 (10 Open Channels)",    kfold_clf10, f1_macro_scores[4]],

    [ (8, 9), "Subsample I",     "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]],

    [ (9,10), "Subsample J",     "Model 3  (3 Open Channels)",     kfold_clf3,  f1_macro_scores[2]],

    [(10,20), "Batches 3 and 4", "Model 1s (1 Slow Open Channel)", kfold_clf1s, f1_macro_scores[0]]

]
def ensemble_by_geometric_mean(sets_of_predictions,

                               number_of_predictions_per_set: int,

                               min_label_value: int,

                               max_label_value: int) -> np.ndarray:

    result = np.ones(number_of_predictions_per_set)

    for index, each_set_of_predictions in enumerate(sets_of_predictions):

        result *= each_set_of_predictions

    result = result ** (1 / len(sets_of_predictions))

    

    return np.nan_to_num(result, nan=min_label_value, posinf=max_label_value, neginf=min_label_value)

    

def predict_using(models, data):

    predictions = []

    if isinstance(models, list):

        for each_model in models:

            predictions.append(each_model.predict(data))

        return ensemble_by_geometric_mean(predictions, len(data), 0, 10)

    else:

        return np.round(models.predict(data))

    

def create_prediction(reference_dataframe, feature_cols, results_dataframe, params):

    total_score = 0.0

    for each_param in params:

        begin_index, end_index = each_param[0]

        start_batch = int(sub_sample_size * begin_index)

        end_batch = int(sub_sample_size * end_index)

        batch_or_sample_models = each_param[3]

        f1_macro_score = each_param[4]

        X_batch = reference_dataframe[feature_cols]

        X_batch = X_batch.iloc[start_batch:end_batch].values.reshape((-1,len(feature_cols)))

        results_dataframe.iloc[start_batch:end_batch, 1] = predict_using(batch_or_sample_models, X_batch)

        print(f"Predicting for {each_param[1]} ({start_batch} to {end_batch}) of submission with predictions from {each_param[2]} with a F1 Macro score of {f1_macro_score}")

        total_score = total_score + f1_macro_score



    print()

    average_f1_macro_score = total_score/len(params)

    print(f"Average F1 Macro across the {len(params)} subsamples/batches: {average_f1_macro_score}")

    results_dataframe.open_channels = results_dataframe.open_channels.astype(int)

    return results_dataframe, average_f1_macro_score
%%time

nestedcv_sub, nestedcv_average_f1_macro_score = create_prediction(test, feature_cols, nestedcv_sub, nestedcv_params)
%%time

kfold_sub, kfold_average_f1_macro_score = create_prediction(test, feature_cols, kfold_sub, kfold_params)
res = 1000

letters = ['A','B','C','D','E','F','G','H','I','J']



def plot_results(reference_dataframe, results_dataframe):

    plt.figure(figsize=(20,5))

    plt.plot(range(0,reference_dataframe.shape[0],res),results_dataframe.open_channels[0::res])

    for i in range(5): plt.plot([i*batch_size,i*batch_size],[-5,12.5],'r')

    for i in range(21): plt.plot([i*sub_sample_size, i*sub_sample_size],[-5,12.5],'r:')

    for k in range(4): plt.text(k*batch_size + (batch_size/2),10,str(k+1),size=20)

    for k in range(10): plt.text(k*sub_sample_size + 40000,7.5,letters[k],size=16) # 

    plt.title('Test Data Predictions',size=16)

    plt.show()
plot_results(test, nestedcv_sub)
nestedcv_sub.describe()
print(nestedcv_sub.open_channels.describe())

nestedcv_sub.open_channels.hist()
nestedcv_sub
nestedcv_sub[100000:200000]
plot_results(test, kfold_sub)
kfold_sub.describe()
print(kfold_sub.open_channels.describe())

kfold_sub.open_channels.hist()
kfold_sub
kfold_sub[100000:200000]
%%time

!rm sub*nestedcv*.csv || true

submission_filename = f'submission-1-{len(feature_cols)}-features-nestedcv-DecisionTree-f1-macro.csv'

nestedcv_sub.to_csv(submission_filename, index=False, float_format='%0.4f')

print(f'Saved {submission_filename} with Macro F1 validation score of {nestedcv_average_f1_macro_score}')

!ls sub*nestedcv*.csv
%%time

!rm sub*kfold*.csv || true

submission_filename = f'submission-2-{len(feature_cols)}-features-kfold-DecisionTree-f1-macro.csv'

nestedcv_sub.to_csv(submission_filename, index=False, float_format='%0.4f')

print(f'Saved {submission_filename} with Macro F1 validation score of {kfold_average_f1_macro_score}')

!ls sub*kfold*.csv