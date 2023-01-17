import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



pd.reset_option('^display.', silent=True)



# Load data

X_train = pd.read_csv("/kaggle/input/donorsprediction/Raw_Data_for_train_test.csv")

X_test = pd.read_csv("/kaggle/input/donorsprediction/Predict_donor.csv")



# Split target and predictors

y_train = X_train['TARGET_B']

num_train = len(X_train)

X_train.drop(['TARGET_B'], axis=1, inplace=True)

df = pd.concat([X_train, X_test], ignore_index=True)
df.head()
# Show the columns types

df.dtypes.value_counts()

categorical_columns = df.select_dtypes('object').columns

print(len(df.columns)-len(df.select_dtypes('object').columns),'numerical columns:')

print([i for i in list(df.columns) if i not in list(df.select_dtypes('object').columns)], '\n')

print(len(df.select_dtypes('object').columns),'categorical columns:')

print(list(df.select_dtypes('object').columns))
df.info()
df.describe()
pd.set_option('mode.chained_assignment', None)



# Delete unused variables CONTROL_NUMBER and TARGET_D

df = df.drop(['CONTROL_NUMBER', 'TARGET_D'], axis=1)



# Fill missing values for age with median

ages = df.groupby(['DONOR_GENDER']).DONOR_AGE

f = lambda x: x.fillna(x.median())

df.DONOR_AGE = ages.transform(f)



# Fill missing values for income group with median

income = df.groupby(['DONOR_AGE', 'DONOR_GENDER']).INCOME_GROUP

f = lambda x: x.fillna(x.median())

df.INCOME_GROUP = income.transform(f)

df.INCOME_GROUP = df.INCOME_GROUP.fillna(4)



# Use zero for missing SES values

df.SES[df.SES == '?'] = 0



# Use zero missing cluster

df.CLUSTER_CODE[df.CLUSTER_CODE == '.'] = 0



# Use mean value S for missing URBANICITY

df.URBANICITY[df.URBANICITY == '?'] = 'S'



# Fill missing values for wealth rating with median

wealth = df.groupby(['DONOR_AGE', 'INCOME_GROUP']).WEALTH_RATING

f = lambda x: x.fillna(x.median())

df.WEALTH_RATING = wealth.transform(f)

df.WEALTH_RATING = df.WEALTH_RATING.fillna(5)



# Use mean value for missing MONTHS_SINCE_LAST_PROM_RESP

df.MONTHS_SINCE_LAST_PROM_RESP[df.MONTHS_SINCE_LAST_PROM_RESP.isnull()] = 19



# Save indices of categorial features

categorical_features_indices = np.where(df.dtypes == 'object')[0]
# Split the df into train and test set

X_train = df.iloc[:num_train,:]

X_test = df.iloc[num_train:,:]



# Make a training and validation set

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.75, stratify=y_train, random_state=0)



# Calculate pos weight

pos_weight = sum(y_train.values == 0)/sum(y_train.values == 1)
import catboost

params = {"iterations": 10000,

          "learning_rate": 0.1,

          "scale_pos_weight": pos_weight,

          "eval_metric": 'AUC',

          "custom_loss": 'Accuracy',

          "loss_function": "Logloss",

          "boosting_type": 'Ordered',

          'od_type': 'Iter',

          'od_wait': 30,

          "use_best_model": True,

          "logging_level": 'Verbose',

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
model.get_feature_importance(train_pool, fstr_type=catboost.EFstrType.FeatureImportance, prettified=True)
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
y_test_preds = model.predict(X_test)

y_test_probas = model.predict_proba(X_test)



print(f"20 first predictions on test set: {y_test_preds[:20]}")

print(f"20 first probability dists: {y_test_probas[:20]}")

print(f"Number of predicated donors: {np.sum(y_test_preds == 1)}")

print(f"Number of predicated non-donors: {np.sum(y_test_preds == 0)}")