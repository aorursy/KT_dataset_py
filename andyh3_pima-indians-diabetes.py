import warnings



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from fancyimpute.knn import KNN

from sklearn import metrics, svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,

                                     cross_val_predict, cross_val_score,

                                     train_test_split)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier



warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.isnull().any()
df.describe()
features_with_zero_values = ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']

df[features_with_zero_values] = df[features_with_zero_values].replace(0, np.nan)

data = KNN(k=5).fit_transform(df.values)

df = pd.DataFrame(data, columns=df.columns)
df.head()
df.hist(figsize=(12, 12))

plt.show()
sns.pairplot(data=df, diag_kind='kde', hue='Outcome', vars=df.columns[:-1])

plt.show()
sns.heatmap(df.corr(), annot=True)

plt.show()
X = df.drop('Outcome', axis=1)

y = df['Outcome']



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

X.head()
X.hist(figsize=(12, 12))

plt.show()
random_state = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
models = [

    ('SVC (Linear)', svm.SVC(kernel='linear', probability=True)),

    ('LR', LogisticRegression()),

    ('SVC (RBF)', svm.SVC(kernel='rbf', probability=True)),

    ('RFC', RandomForestClassifier())

]



models.extend(

    [

        (f'KNN (n={i})', KNeighborsClassifier(i))

        for i in range(1, 9)

    ]

)



cv = StratifiedKFold(n_splits=10, random_state=random_state)



model_metrics_map = {}

    

for model_descriptor, model in models:

    print(f"Computing metrics for model {model_descriptor}")

    # Fit mode

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Cross val predict class

    cv_y_pred = cross_val_predict(model, X_train, y_train, cv=cv)

    model_metrics_map.setdefault(model_descriptor, {})['cv_y_pred'] = cv_y_pred

    # Cross val predict probability

    cv_y_prob_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')

    model_metrics_map.setdefault(model_descriptor, {})['cv_y_prob_pred'] = cv_y_prob_pred

    # Train accuracy

    train_accuracy = metrics.accuracy_score(y_test, y_pred)

    model_metrics_map.setdefault(model_descriptor, {})['train_accuracy'] = train_accuracy

    # Test accuracy using cross validation

    test_cv_accuracies = cross_val_score(model, X_train, y_train, cv=cv)

    model_metrics_map.setdefault(model_descriptor, {})['test_cv_accuracies'] = test_cv_accuracies

    test_mean_cv_accuracy = test_cv_accuracies.mean()

    model_metrics_map.setdefault(model_descriptor, {})['test_mean_cv_accuracy'] = test_mean_cv_accuracy

    test_std_cv_accuracy = test_cv_accuracies.std()

    model_metrics_map.setdefault(model_descriptor, {})['test_std_cv_accuracy'] = test_std_cv_accuracy

    # Train ROC AUC

    train_roc_auc = metrics.roc_auc_score(y_test, y_pred)

    model_metrics_map.setdefault(model_descriptor, {})['train_roc_auc'] = train_roc_auc

    # Test ROC AUC using cross validation

    test_roc_aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    model_metrics_map.setdefault(model_descriptor, {})['test_roc_aucs'] = test_roc_aucs

    test_mean_auc = test_roc_aucs.mean()

    model_metrics_map.setdefault(model_descriptor, {})['test_mean_auc'] = test_mean_auc

    test_std_auc = test_roc_aucs.std()

    model_metrics_map.setdefault(model_descriptor, {})['test_std_auc'] = test_std_auc

    # Confusion matrix

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    model_metrics_map.setdefault(model_descriptor, {})['confusion_matrix'] = confusion_matrix

    # Test Precision

    test_precisions = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision')

    model_metrics_map.setdefault(model_descriptor, {})['test_precisions'] = test_precisions

    test_mean_precision = test_precisions.mean()

    model_metrics_map.setdefault(model_descriptor, {})['test_mean_precision'] = test_mean_precision

    test_std_precision = test_precisions.std()

    model_metrics_map.setdefault(model_descriptor, {})['test_std_precision'] = test_std_precision

    # Train Precision

    train_precision = metrics.precision_score(y_train, cv_y_pred)

    model_metrics_map.setdefault(model_descriptor, {})['train_precision'] = train_precision

    # Test Recall

    test_recalls = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')

    model_metrics_map.setdefault(model_descriptor, {})['test_recalls'] = test_recalls

    test_mean_recall = test_recalls.mean()

    model_metrics_map.setdefault(model_descriptor, {})['test_mean_recall'] = test_mean_recall

    test_std_recall = test_recalls.std()

    model_metrics_map.setdefault(model_descriptor, {})['test_std_recall'] = test_std_recall

    # Train Recall

    train_recall = metrics.recall_score(y_train, cv_y_pred)

    model_metrics_map.setdefault(model_descriptor, {})['train_recall'] = train_recall

    # Test F1 Score

    test_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

    model_metrics_map.setdefault(model_descriptor, {})['test_f1_scores'] = test_f1_scores

    test_mean_f1_score = test_f1_scores.mean()

    model_metrics_map.setdefault(model_descriptor, {})['test_mean_f1_score'] = test_mean_f1_score

    test_std_f1_score = test_f1_scores.std()

    model_metrics_map.setdefault(model_descriptor, {})['test_std_f1_score'] = test_std_f1_score

    # Train F1 Score

    train_f1_score = metrics.f1_score(y_train, cv_y_pred)

    model_metrics_map.setdefault(model_descriptor, {})['train_f1_score'] = train_f1_score

    

model_metrics_df = pd.DataFrame(model_metrics_map).T

model_metrics_df.head()



core_metrics = [

    'train_accuracy', 

    'train_roc_auc',

    'train_precision',

    'train_recall',

    'train_f1_score',

    'test_mean_cv_accuracy', 

    'test_mean_auc',

    'test_mean_precision',

    'test_mean_recall',

    'test_mean_f1_score'

]

core_metrics_df = model_metrics_df[core_metrics]

core_metrics_df



def highlight_max(s):

    is_max = s == s.max()

    return ['background-color: green' if v else '' for v in is_max]



core_metrics_df.style.apply(highlight_max)

def plot_confusion_matrices(df):

    fig, ax = plt.subplots(2, 6, figsize=(24, 8))

    index_ax_map = {

        6 * i + j: ax[i, j]

        for i in range(2)

        for j in range(6)

    }

    confusion_matrices = df['confusion_matrix'].values

    model_descriptors = df.index.values

    for i, o in enumerate(zip(model_descriptors, confusion_matrices)):

        model_descriptor, confusion_matrix = o

        sns.heatmap(pd.DataFrame(confusion_matrix), ax=index_ax_map[i], annot=True, fmt='g')

        index_ax_map[i].set_title(model_descriptor)

        if i > 5:

            index_ax_map[i].set_xlabel("Predicted")

        if i in (0, 6):

            index_ax_map[i].set_ylabel("Actual")

    plt.show()





plot_confusion_matrices(model_metrics_df)
best_models_df = model_metrics_df.loc[['RFC', 'KNN (n=7)'], :]

best_models_df.head()
def plot_roc_auc_curve(df, y_true):

    fig, ax = plt.subplots(figsize=(12, 8))

    y_scores = df['cv_y_prob_pred'].values

    auc_scores = df['train_roc_auc'].values

    model_descriptors = df.index.values

    for model_descriptor, y_score, auc_score in zip(model_descriptors, y_scores, auc_scores):

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score[:, 1])

        ax.plot(fpr, tpr, label=f"{model_descriptor} (AUC={auc_score:.3f})")

    ax.legend()

    ax.plot([0, 1], 'k--')

    ax.set_xlim([0.0, 1.0])

    ax.set_ylim([0.0, 1.0])

    plt.show()

    



plot_roc_auc_curve(best_models_df, y_train)
def plot_precision_recall_vs_threshold(df, y_true):

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    index_ax_map = {

        0: ax[0],

        1: ax[1]

    }

    proba_preds = df['cv_y_prob_pred'].values

    model_descriptors = df.index.values

    for i, o in enumerate(zip(model_descriptors, proba_preds)):

        model_descriptor, proba_pred = o

        precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, proba_pred[:, 1])

        index_ax_map[i].plot(thresholds, precisions[:-1], "b--", label="Precision")

        index_ax_map[i].plot(thresholds, recalls[:-1], "g-", label="Recall")

        index_ax_map[i].set_ylim([0, 1])

        index_ax_map[i].set_title(model_descriptor)

        index_ax_map[i].set_xlabel('Recall')

        index_ax_map[i].set_ylabel('Precision')

        

        

plot_precision_recall_vs_threshold(best_models_df, y_train)