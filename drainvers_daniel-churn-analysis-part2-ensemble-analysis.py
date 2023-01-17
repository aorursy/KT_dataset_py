import numpy as np

import pandas as pd

import os

import datetime as dt

import seaborn as sns

from matplotlib import pyplot as plt

from IPython.display import display



csv_paths = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        csv_paths.append(os.path.join(dirname, filename))
print(csv_paths)
raw_dataframes = []

for path in csv_paths:

    df = pd.read_csv(path, index_col=0)

    raw_dataframes.append(df)
print("Hard Churn")

display(raw_dataframes[0].head())

print("Soft Churn")

display(raw_dataframes[1].head())
raw_hard_churn = raw_dataframes[0].copy()

raw_hard_churn.info()

raw_hard_churn.isna().sum()
raw_soft_churn = raw_dataframes[1].copy()

raw_soft_churn.info()

raw_soft_churn.isna().sum()
raw_soft_churn = raw_soft_churn[~raw_soft_churn.isnull().any(axis=1)]
raw_hard_churn.corr()
plt.figure(figsize=(14, 7))

plt.title("Feature Correlation (Hard)")

sns.heatmap(raw_hard_churn.loc[:, raw_hard_churn.columns != 'imei_name'].corr(), annot=True)
plt.figure(figsize=(14, 7))

plt.title("Feature Correlation (Soft)")

sns.heatmap(raw_soft_churn.loc[:, raw_soft_churn.columns != 'imei_name'].corr(), annot=True)
from sklearn.model_selection import train_test_split



# Shuffle is disabled and random state here is set for reproducibility



def split_data(dataframe, result_column, test_size):

    X, y = dataframe.loc[:, (dataframe.columns != result_column) & (dataframe.columns != 'imei_name')], dataframe[result_column]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=False)

    return X_train, X_valid, y_train, y_valid



X_hard_train, X_hard_valid, y_hard_train, y_hard_valid = split_data(raw_hard_churn, 'churn', 0.3)

X_soft_train, X_soft_valid, y_soft_train, y_soft_valid = split_data(raw_soft_churn, 'churn', 0.3)
print("Hard Churn (Train) Predictors")

display(X_hard_train.head())

print("Hard Churn (Train) Target")

display(y_hard_train.head())
hard_training_counts = y_hard_train.value_counts()

hard_testing_counts = y_hard_valid.value_counts()

soft_training_counts = y_soft_train.value_counts()

soft_testing_counts = y_soft_valid.value_counts()



print("Churn:Retain Ratio (Hard, Train) =", hard_training_counts[1] / hard_training_counts[0])

print("Churn:Retain Ratio (Hard, Valid) =", hard_testing_counts[1] / hard_testing_counts[0])

print("Churn:Retain Ratio (Soft, Train) =", soft_training_counts[1] / soft_training_counts[0])

print("Churn:Retain Ratio (Soft, Valid) =", soft_testing_counts[1] / soft_testing_counts[0])
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate

import pickle

# Random state here is set for reproducibility



hard_models = {

    'RandomForestClassifier': {

        'model': RandomForestClassifier,

        'param': {

            'n_estimators': 155,

            'criterion': 'gini',

            'min_samples_leaf': 10,

            'max_features': 'sqrt',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier': {

        'model': ExtraTreesClassifier,

        'param': {

            'n_estimators': 143,

            'criterion': 'gini',

            'min_samples_leaf': 10,

            'max_features': 'sqrt',

            'random_state': 0

        }

    },

    'AdaBoostClassifier': {

        'model': AdaBoostClassifier,

        'param': {

            'n_estimators': 170,

            'learning_rate': 0.14046548511432905,

            'random_state': 0

        }

    },

    'GradientBoostingClassifier': {

        'model': GradientBoostingClassifier,

        'param': {

            'n_estimators': 161,

            'learning_rate': 0.021480690032076512,

            'subsample': 1.0,

            'min_samples_leaf': 4,

            'max_features': 'log2',

            'random_state': 0

        }

    }

}



soft_models = {

    'RandomForestClassifier': {

        'model': RandomForestClassifier,

        'param': {

            'n_estimators': 167,

            'criterion': 'gini',

            'min_samples_leaf': 6,

            'max_features': 'log2',

            'random_state': 0

        }

    },

    'ExtraTreesClassifier': {

        'model': ExtraTreesClassifier,

        'param': {

            'n_estimators': 125,

            'criterion': 'gini',

            'min_samples_leaf': 5,

            'max_features': 'sqrt',

            'random_state': 0

        }

    },

    'AdaBoostClassifier': {

        'model': AdaBoostClassifier,

        'param': {

            'n_estimators': 198,

            'learning_rate': 0.9928786250309561,

            'random_state': 0

        }

    },

    'GradientBoostingClassifier': {

        'model': GradientBoostingClassifier,

        'param': {

            'n_estimators': 166,

            'learning_rate': 0.7479432334085875,

            'subsample': 0.9,

            'min_samples_leaf': 1,

            'max_features': 'log2',

            'random_state': 0

        }

    }

}



ground_truth = {

    "hard": {

        "train": y_hard_train,

        "valid": y_hard_valid

    },

    "soft": {

        "train": y_soft_train,

        "valid": y_soft_valid

    }

}



preds = {

    "hard": {

        "train": {},

        "valid": {}

    },

    "soft": {

        "train": {},

        "valid": {}

    }

}



# For ROC Curve

preds_proba = {

    "hard": {

        "train": {},

        "valid": {}

    },

    "soft": {

        "train": {},

        "valid": {}

    }

}
scores = {

    "hard": {

        "train": pd.DataFrame(),

        "valid": pd.DataFrame()

    },

    "soft": {

        "train": pd.DataFrame(),

        "valid": pd.DataFrame()

    }

}
for current_model in hard_models:

    clf = hard_models[current_model]['model'](**hard_models[current_model]['param'])

    clf.fit(X_hard_train, y_hard_train)



    # Pickle model

    model_filename = 'Daniel_HardChurn_{}.pickle'.format(current_model)

    with open(model_filename, 'wb') as outfile:

        pickle.dump(clf, outfile)



    preds["hard"]["train"][current_model] = clf.predict(X_hard_train)

    preds["hard"]["valid"][current_model] = clf.predict(X_hard_valid)



    preds_proba["hard"]["train"][current_model] = clf.predict_proba(X_hard_train)

    preds_proba["hard"]["valid"][current_model] = clf.predict_proba(X_hard_valid)



for current_model in soft_models:

    clf = soft_models[current_model]['model'](**soft_models[current_model]['param'])

    clf.fit(X_soft_train, y_soft_train)



    # Pickle model

    model_filename = 'Daniel_SoftChurn_{}.pickle'.format(current_model)

    with open(model_filename, 'wb') as outfile:

        pickle.dump(clf, outfile)



    preds["soft"]["train"][current_model] = clf.predict(X_soft_train)

    preds["soft"]["valid"][current_model] = clf.predict(X_soft_valid)



    preds_proba["soft"]["train"][current_model] = clf.predict_proba(X_soft_train)

    preds_proba["soft"]["valid"][current_model] = clf.predict_proba(X_soft_valid)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
def metric_score(y_true, y_pred, y_pred_proba, churn_type, feature_type, model):

    global scores

    accuracy = accuracy_score(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])

    roc_auc = auc(fpr, tpr)

    f1 = f1_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    best_threshold = thresholds[np.argmax(tpr - fpr)]

    

    score = {'Accuracy': [accuracy], 'AUC': [roc_auc], 'F1':[f1], 'Precision':[precision], 'Recall':[recall], 'Best Threshold':[best_threshold]}



    print('Metrics for {} churn ({}) [{}]'.format(churn_type, feature_type, model))

    print("Accuracy :", accuracy)

    print("ROC AUC  :", roc_auc)

    print("F1       :", f1)

    print("Precision:", precision)

    print("Recall   :", recall)

    print("Threshold:", best_threshold, "(Best)")

    

    scores[churn_type][feature_type] = pd.concat([scores[churn_type][feature_type], pd.DataFrame(score, index=[model])])
def plot_confusion_matrix(y_true, y_pred, y_pred_proba, churn_type, feature_type, model):

    plt.figure(figsize = (20,7))

    plt.subplots_adjust(right=1)

    plt.subplot(1, 2, 1)



    data = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))



    print('Confusion matrix for {} churn ({}) [{}]'.format(churn_type, feature_type, model))

    print(df_cm)



    df_cm.index.name = 'Actual'

    df_cm.columns.name = 'Predicted'

    plt.title('Confusion matrix for {} churn ({}) [{}]'.format(churn_type, feature_type, model))

    sns.set(font_scale=1.4) # Label size

    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}) # Font size



    plt.subplot(1, 2, 2, facecolor='aliceblue')



    # Calculate ROC Curve, we take only the positive outcomes from the probabilities

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])

    roc_auc = auc(fpr, tpr)



    # We use Youden's J statistic for ease

    J = tpr - fpr

    idx = np.argmax(J)



    plt.title('ROC Curve : {} churn ({}) [{}]'.format(churn_type, feature_type, model))

    plt.plot(fpr, tpr, 'r', label='AUC = %0.3f' % roc_auc)

    plt.scatter(fpr[idx], tpr[idx], marker='o', color='r', label='Best threshold = %0.3f' % thresholds[idx])

    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'b--')

    plt.xlim([-0.01, 1.01])

    plt.ylim([-0.01, 1.01])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')



    plt.show()
for churn_type, churn_dict in preds.items():

    for feature_type, models in churn_dict.items():

        for model, results in models.items():

            y_valid = ground_truth[churn_type][feature_type]

            y_pred_proba = preds_proba[churn_type][feature_type][model]

            metric_score(y_valid, results, y_pred_proba, churn_type, feature_type, model)

            print('')

            plot_confusion_matrix(y_valid, results, y_pred_proba, churn_type, feature_type, model)
for churn_type, churn_dict in scores.items():

    for feature_type, df in churn_dict.items():

        print("{} churn ({})".format(churn_type, feature_type))

        display(df.style.set_properties(**{'background-color': 'black', 'color': 'white'}).highlight_max(color='green' if feature_type == 'train' else 'purple'))
def confusion_aggregate(X, y_pred, y_true):

    X_valid = X.copy()

    X_valid['Prediction'] = y_pred

    X_valid['Ground Truth'] = y_true



    same = X_valid[X_valid['Ground Truth'] == X_valid['Prediction']]

    diff = X_valid[X_valid['Ground Truth'] != X_valid['Prediction']]



    TP = same[same['Prediction'] == 1]

    TN = same[same['Prediction'] != 1]

    FP = diff[diff['Prediction'] == 1]

    FN = diff[diff['Prediction'] != 1]

    

    TP_agg = TP.describe()

    TN_agg = TN.describe()

    FP_agg = FP.describe()

    FN_agg = FN.describe()



    con_agg = pd.DataFrame()

    con_agg['Mean TP'] = TP_agg.loc['mean']

    con_agg['Mean FP'] = FP_agg.loc['mean']

    con_agg['Mean TN'] = TN_agg.loc['mean']

    con_agg['Mean FN'] = FN_agg.loc['mean']

    con_agg['TP-FP'] = con_agg['Mean TP'] - con_agg['Mean FP']

    con_agg['TN-FN'] = con_agg['Mean TN'] - con_agg['Mean FN']



    return con_agg
hard_train_ca = confusion_aggregate(X_hard_train, preds['hard']['train']['RandomForestClassifier'], ground_truth['hard']['train'])

hard_valid_ca = confusion_aggregate(X_hard_valid, preds['hard']['valid']['RandomForestClassifier'], ground_truth['hard']['valid'])

soft_train_ca = confusion_aggregate(X_soft_train, preds['soft']['train']['RandomForestClassifier'], ground_truth['soft']['train'])

soft_valid_ca = confusion_aggregate(X_soft_valid, preds['soft']['valid']['RandomForestClassifier'], ground_truth['soft']['valid'])



print('Hard Churn (Train)')

display(hard_train_ca)

print('Hard Churn (Valid)')

display(hard_valid_ca)

print('Soft Churn (Train)')

display(soft_train_ca)

print('Soft Churn (Valid)')

display(soft_valid_ca)
with open("Daniel_HardChurn_RandomForestClassifier.pickle", "rb") as infile:

    rf_clf_hard = pickle.load(infile)



with open("Daniel_SoftChurn_RandomForestClassifier.pickle", "rb") as infile:

    rf_clf_soft = pickle.load(infile)
# Only used to verify that the features are indeed important



from sklearn.feature_selection import SelectFromModel



sel_hard = SelectFromModel(rf_clf_hard, prefit=True)

selected_features_hard = X_hard_train.columns[(sel_hard.get_support())]

print(list(np.array(selected_features_hard)))



sel_soft = SelectFromModel(rf_clf_soft, prefit=True)

selected_features_soft = X_soft_train.columns[(sel_soft.get_support())]

print(list(np.array(selected_features_soft)))
from sklearn.inspection import permutation_importance



def show_feature_importance(model, X, y, churn_type, feature_type):

    feature_names = X.columns

    tree_feature_importances = model.feature_importances_

    sorted_idx = tree_feature_importances.argsort()

    

    y_ticks = np.arange(0, len(feature_names))

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1.barh(y_ticks, tree_feature_importances[sorted_idx])

    ax1.set_yticklabels(feature_names[sorted_idx])

    ax1.set_yticks(y_ticks)

    ax1.set_title("Feature Importance (MDI) [{}_{}]".format(churn_type, feature_type))

    

    #   result = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=2)

    #   sorted_idx = result.importances_mean.argsort()



    #   ax2.boxplot(result.importances[sorted_idx].T, vert=False, labels=feature_names[sorted_idx])

    #   ax2.set_title("Permutation Importance [{}_{}]".format(churn_type, feature_type))

    

    plt.show()



def get_top3(X, model):

    sorted_idx = model.feature_importances_.argsort()

    return list(np.array(model.feature_importances_)[sorted_idx])[-3:][::-1], list(np.array(X.columns)[sorted_idx])[-3:][::-1]
show_feature_importance(rf_clf_hard, X_hard_train, y_hard_train, 'hard', 'train')

show_feature_importance(rf_clf_soft, X_soft_train, y_soft_train, 'soft', 'train')



hard_top3_coef, hard_top3 = get_top3(X_hard_train, rf_clf_hard)

soft_top3_coef, soft_top3 = get_top3(X_soft_train, rf_clf_soft)



print("Top 3 Features (Hard Churn):")

print(dict(zip(hard_top3, hard_top3_coef)))

print("Top 3 Features (Soft Churn):")

print(dict(zip(soft_top3, soft_top3_coef)))
print("Hard Churn")

display(hard_valid_ca.loc[hard_top3, :])

print("Soft Churn")

display(soft_valid_ca.loc[soft_top3, :])
df_plot = X_soft_valid.copy().loc[:, soft_top3]

df_plot['prediction'] = preds['soft']['valid']['RandomForestClassifier']

df_plot['churn'] = y_soft_valid

df_plot
def categories(p, c):

    if p == c:

        if p == 1:

            return 'g'

        else:

            return 'b'

    else:

        if p == 1:

            return 'r'

        else:

            return 'm'
df_plot['category'] = df_plot.apply(lambda x: categories(x['prediction'], x['churn']), axis=1)

df_plot
from mpl_toolkits.mplot3d import Axes3D



fig3d = plt.figure(figsize=(30,15))

fig3dax = fig3d.add_subplot(111, projection='3d')



for index, row in df_plot.iterrows():

    xs = row[soft_top3[2]]

    ys = row[soft_top3[1]]

    zs = row[soft_top3[0]]

    m = row['category']

    fig3dax.scatter(xs, ys, zs, c=m)



fig3dax.set_xlabel(soft_top3[2] + " (x)")

fig3dax.set_ylabel(soft_top3[1] + " (y)")

fig3dax.set_zlabel(soft_top3[0] + " (z)")



plt.show()
def confusion_aggregate_proba(X, y_true, y_pred, y_proba):

    X_valid = X.copy()

    X_valid['Prediction'] = y_pred

    X_valid['Ground Truth'] = y_true

    X_valid['Churn+ Probability'] = y_proba[:, 1]

    X_valid['Churn- Probability'] = y_proba[:, 0]



    same = X_valid[X_valid['Ground Truth'] == X_valid['Prediction']]

    diff = X_valid[X_valid['Ground Truth'] != X_valid['Prediction']]



    TP = same[same['Prediction'] == 1]

    TN = same[same['Prediction'] != 1]

    FP = diff[diff['Prediction'] == 1]

    FN = diff[diff['Prediction'] != 1]

    

    dict_con_agg = {}

    

    for i in range(0, 10, 1):

        low = round(i * 0.1, 1)

        high = round(i * 0.1 + 0.1, 1)



        curr_TP = TP[(TP['Churn+ Probability'] > low) & (TP['Churn+ Probability'] <= high)]

        curr_FP = FP[(FP['Churn+ Probability'] > low) & (FP['Churn+ Probability'] <= high)]

        curr_TN = TN[(TN['Churn- Probability'] > low) & (TN['Churn- Probability'] <= high)]

        curr_FN = FN[(FN['Churn- Probability'] > low) & (FN['Churn- Probability'] <= high)]

        

        con_agg = pd.DataFrame()

        con_agg['TP'] = curr_TP.describe().loc['count']

        con_agg['FP'] = curr_FP.describe().loc['count']

        con_agg['TN'] = curr_TN.describe().loc['count']

        con_agg['FN'] = curr_FN.describe().loc['count']

        con_agg['Mean TP'] = curr_TP.describe().loc['mean']

        con_agg['Mean FP'] = curr_FP.describe().loc['mean']

        con_agg['Mean TN'] = curr_TN.describe().loc['mean']

        con_agg['Mean FN'] = curr_FN.describe().loc['mean']



        con_agg['Delta Mean+'] = con_agg['Mean TP'] - con_agg['Mean FP']

        con_agg['Delta Mean-'] = con_agg['Mean TN'] - con_agg['Mean FN']

        

        dict_con_agg[i] = con_agg

        

    return dict_con_agg



curr_model = 'RandomForestClassifier'
churn_type = 'hard'

feature_type = 'train'



y_pred_proba = preds_proba[churn_type][feature_type][curr_model]

y_pred = preds[churn_type][feature_type][curr_model]

dict_hard_train = confusion_aggregate_proba(X_hard_train, y_hard_train, y_pred, y_pred_proba)



for key, item in dict_hard_train.items():

    low = round(key * 0.1, 1)

    high = round(key * 0.1 + 0.1, 1)

    print("Range: {}-{}".format(low, high))

    display(item.loc[hard_top3, : ])
churn_type = 'hard'

feature_type = 'valid'



y_pred_proba = preds_proba[churn_type][feature_type][curr_model]

y_pred = preds[churn_type][feature_type][curr_model]

dict_hard_valid = confusion_aggregate_proba(X_hard_valid, y_hard_valid, y_pred, y_pred_proba)



for key, item in dict_hard_valid.items():

    low = round(key * 0.1, 1)

    high = round(key * 0.1 + 0.1, 1)

    print("Range: {}-{}".format(low, high))

    display(item.loc[hard_top3, : ])
churn_type = 'soft'

feature_type = 'train'



y_pred_proba = preds_proba[churn_type][feature_type][curr_model]

y_pred = preds[churn_type][feature_type][curr_model]

dict_soft_train = confusion_aggregate_proba(X_soft_train, y_soft_train, y_pred, y_pred_proba)



for key, item in dict_soft_train.items():

    low = round(key * 0.1, 1)

    high = round(key * 0.1 + 0.1, 1)

    print("Range: {}-{}".format(low, high))

    display(item.loc[soft_top3, : ])
churn_type = 'soft'

feature_type = 'valid'



y_pred_proba = preds_proba[churn_type][feature_type][curr_model]

y_pred = preds[churn_type][feature_type][curr_model]

dict_soft_valid = confusion_aggregate_proba(X_soft_valid, y_soft_valid, y_pred, y_pred_proba)



for key, item in dict_soft_valid.items():

    low = round(key * 0.1, 1)

    high = round(key * 0.1 + 0.1, 1)

    print("Range: {}-{}".format(low, high))

    display(item.loc[soft_top3, : ])