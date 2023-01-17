import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



pd.reset_option('^display.', silent=True)



# Load data

X_train = pd.read_csv('/kaggle/input/employee-attrition/employee_attrition_train.csv')

X_test = pd.read_csv('/kaggle/input/employee-attrition/employee_attrition_test.csv')



# Make target numerical

X_train.Attrition = X_train.Attrition.apply(lambda x: 0 if x == 'No' else 1)



# Split target and predictors

y_train = X_train['Attrition']

num_train = len(X_train)

X_train.drop(['Attrition'], axis=1, inplace=True)



df = pd.concat([X_train, X_test], ignore_index=True)
df.info()
df.describe()
# Detect if data is imbalanced

print(y_train.value_counts())
pd.set_option('mode.chained_assignment', None)



# Fill missing values for DailyRate with median

daily_rates = df.groupby(['Gender', 'Education', 'JobLevel']).DailyRate

f = lambda x: x.fillna(x.median())

df.DailyRate = daily_rates.transform(f)



# Fill missing values for age with median

ages = df.groupby(['Gender', 'Education']).Age

f = lambda x: x.fillna(x.median())

df.Age = ages.transform(f)



# Set missing values for travel to Non-Travel

df.BusinessTravel[df.BusinessTravel.isnull()] = 'Non-Travel'



# Set missing values for DistanceFromHome to median

df.DistanceFromHome[df.DistanceFromHome.isnull()] = np.around(df.DistanceFromHome.mean())



# Set missing values for MaritalStatus to Married

df.MaritalStatus[df.MaritalStatus.isnull()] = 'Married'



# Save indices of categorial features

categorical_features_indices = np.where(df.dtypes == 'object')[0]
# Split the df into train and test set

X_train = df.iloc[:num_train,:]

X_test = df.iloc[num_train:,:]



# Make a training and validation set

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.75, stratify=y_train, random_state=0)



# Imbalanced data, so set weight

pos_weight = sum(y_train.values == 0)/sum(y_train.values == 1)
import catboost

params = {"iterations": 1000,

          "learning_rate": 0.1,

          "scale_pos_weight": pos_weight,

          "eval_metric": "AUC",

          "custom_loss": "Accuracy",

          "loss_function": "Logloss",

          "od_type": "Iter",

          "od_wait": 30,

          "logging_level": "Verbose",

          "random_seed": 0

}



train_pool = catboost.Pool(X_train, y_train, cat_features=categorical_features_indices)

valid_pool = catboost.Pool(X_valid, y_valid, cat_features=categorical_features_indices)



model = catboost.CatBoostClassifier(**params)

model.fit(train_pool, eval_set=valid_pool, plot=False)
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.metrics import precision_score, recall_score

from catboost.utils import get_roc_curve, select_threshold



def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')



y_pred = model.predict(X_valid)

print(f"Confusion Matrix:\n {confusion_matrix(y_valid, y_pred)}\n")

print(f"Classification Report:\n {classification_report(y_valid, y_pred)}\n")



y_pred = model.predict(X_valid)

print(f"Accuracy on validation set: {accuracy_score(y_valid, y_pred)}")

print(f"Precision on validation set: {precision_score(y_valid, y_pred)}")

print(f"Recall on validation set: {recall_score(y_valid, y_pred)}")



fpr_train, tpr_train, _ = get_roc_curve(model, train_pool)

fpr_valid, tpr_valid, _ = get_roc_curve(model, valid_pool)



plt.figure(figsize=(8,6))

plot_roc_curve(fpr_train, tpr_train, "Training ROC")

plot_roc_curve(fpr_valid, tpr_valid, "Validation ROC")

plt.legend(loc="lower right")

plt.title("ROC plot")

plt.ylabel("TPR")

plt.xlabel("FPR")

plt.show()
# Get feature importances

model.get_feature_importance(train_pool, fstr_type=catboost.EFstrType.FeatureImportance, prettified=True)
# Plot feature importances

importances = model.get_feature_importance(train_pool, fstr_type=catboost.EFstrType.FeatureImportance)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,12))

plt.title('Feature importance for CatBoost classifier')

plt.barh(X_train.columns[indices][::-1], importances[indices][::-1])
interactions = model.get_feature_importance(train_pool, fstr_type=catboost.EFstrType.Interaction)

feature_interaction = [[X_train.columns[interaction[0]], X_train.columns[interaction[1]], interaction[2]] for interaction in interactions]

feature_interaction_df = pd.DataFrame(feature_interaction, columns=['feature1', 'feature2', 'interaction_strength'])

feature_interaction_df.head(10)
pd.Series(index=zip(feature_interaction_df['feature1'], feature_interaction_df['feature2']), data=feature_interaction_df['interaction_strength'].values, name='interaction_strength').head(10)[::-1].plot(kind='barh', figsize=(12,12))
import shap

shap_values = model.get_feature_importance(train_pool, fstr_type=catboost.EFstrType.ShapValues)

shap.initjs()

shap.summary_plot(shap_values[:, :-1], X_train, feature_names=X_train.columns.tolist())
shap.summary_plot(shap_values[:, :-1], X_train, feature_names=X_train.columns.tolist(), plot_type="bar")
# Helper function to plot shap values

def shap_plot(j):

    explainerModel = shap.TreeExplainer(model)

    shap_values_Model = explainerModel.shap_values(X_test)

    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], X_test.iloc[[j]])

    return(p)



shap_plot(0)
shap_plot(10)
shap_plot(45)
shap_plot(49)
shap_plot(50)
y_test_preds = model.predict(X_test)

y_test_probas = model.predict_proba(X_test)



print(f"First 20 predictions on test set: {y_test_preds[:10]}")

print(f"First 20 dropout probabilities: {y_test_probas[:10]}")

print(f"Number of predicated dropouts: {np.sum(y_test_preds == 1)}")

print(f"Number of predicated non-dropouts: {np.sum(y_test_preds == 0)}")