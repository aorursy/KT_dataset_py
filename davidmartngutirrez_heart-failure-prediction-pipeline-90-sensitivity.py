# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing libraries

import pandas as pd

import numpy as np

import plotly.graph_objs as go

from matplotlib import pyplot as plt

from mlxtend.plotting import plot_confusion_matrix
# Prepare your file

parent_dir: str = os.path.join('/kaggle', 'input', 'heart-failure-clinical-data')

dataset_name: str = "heart_failure_clinical_records_dataset.csv"

dataset_path: str = os.path.join(parent_dir, dataset_name)

print(f"Dataset directory: {dataset_path}")
# Load data

heart_failure_df: pd.DataFrame = pd.read_csv(dataset_path)

heart_failure_df.head()
def plot_survival_vs_binary_variable(data: pd.DataFrame, predicted_col: str, response_col: str, unique_labels: list):

    # Preprocess the data

    positive = data[data[predicted_col]==1]

    negative = data[data[predicted_col]==0]

    

    # Extract values and labels

    data_values = zip([positive, negative, positive, negative.copy()],

                      [0,0,1,1])

    values = [i[i[response_col]==j].shape[0] for i, j in data_values]

    

    # Extract labels

    labels = [f"{label} - Survived" for label in unique_labels] + [f"{label} - not Survived" for label in unique_labels]

    

    # Plot Figure

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])

    fig.update_layout(

        title_text=f"Analysis on Survival - {predicted_col.title()}")

    fig.show()
# By Sex

predicted_col: str = "sex"

response_col: str = "DEATH_EVENT"

unique_labels: list = ["Male", "Female"]



# Call Function

plot_survival_vs_binary_variable(data=heart_failure_df, 

                                 predicted_col=predicted_col,

                                 response_col=response_col,

                                 unique_labels=unique_labels)
# By Diabetes

predicted_col: str = "diabetes"

response_col: str = "DEATH_EVENT"

unique_labels: list = ["Yes", "No"]



# Call Function

plot_survival_vs_binary_variable(data=heart_failure_df, 

                                 predicted_col=predicted_col,

                                 response_col=response_col,

                                 unique_labels=unique_labels)
# By anaemia

predicted_col: str = "anaemia"

response_col: str = "DEATH_EVENT"

unique_labels: list = ["Yes", "No"]



# Call Function

plot_survival_vs_binary_variable(data=heart_failure_df, 

                                 predicted_col=predicted_col,

                                 response_col=response_col,

                                 unique_labels=unique_labels)
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import RobustScaler

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score
response_variable: str = "DEATH_EVENT"



# Get Predictive columns

predictive_variables: list = list(heart_failure_df.columns)

predictive_variables.remove(response_variable)

print(predictive_variables)

# Prepare data (Cross-validation purposes)

X: pd.DataFrame = heart_failure_df[predictive_variables]

y: pd.Series = heart_failure_df[[response_variable]]

test_size: float = 0.25



# Split data into separate sets (Training purposes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



print(f"X_train shape: {X_train.shape}\n")

print(f"X_test shape: {X_test.shape}\n")

print(f"y_train shape: {y_train.shape}\n")

print(f"y_test shape: {y_test.shape}\n")
# Scaler

scaler: RobustScaler = RobustScaler()

n_features: int = int(X.shape[1])



# Numerical transformer

numerical_transformer:Pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median',

                                                                            fill_value=-99)),

                                                 ('scaler', scaler)

                                                 ])

# Preprocessor Transformer

preprocessor_transformer: ColumnTransformer = ColumnTransformer(transformers=[

    ('num', numerical_transformer, predictive_variables)])

# Ensemble model (stacking)



random_state=123

estimators: list = [('RF', RandomForestClassifier(max_features="sqrt",

                                                  n_estimators=150,

                                                  random_state=random_state)),

                    ('GB', GradientBoostingClassifier(max_depth=5,

                                                     random_state=random_state)),

                    ('SVM', SVC(C=2))]

final_estimator: LogisticRegression = LogisticRegression()



ensemble_model: StackingClassifier = StackingClassifier(estimators=estimators,

                                                        final_estimator=final_estimator

                                                        )

print(ensemble_model)
# Complete pipeline

steps: list = [("preprocessor", preprocessor_transformer),

               ('classifier', ensemble_model)]

ml_model: Pipeline = Pipeline(steps=steps)

print(ml_model)
from sklearn.model_selection import cross_validate

y_np: np.array = np.array(y).ravel()

cv: int = 5

scoring: tuple = ('balanced_accuracy', 'f1', 'precision', 'recall', 'roc_auc')

scores = cross_validate(ml_model, X, y_np, cv=cv, scoring=scoring, return_train_score=True)

for metric_name, score in scores.items():

    print(f"{metric_name} mean: {np.mean(score)}, {metric_name} std: {np.std(score)}")
# Train the model

ml_model.fit(X=X_train, y=np.array(y_train).ravel())
y_pred=ml_model.predict(X_test)

print(f"Ensemble Model score: {ml_model.score(X_test, y_test)}")
# Prepare Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure()

plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)

plt.title("Ensemble Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
# Compute 

TP: float = cm[0,0]

FP: float = cm[1,0]

FN: float = cm[0,1]

TN: float = cm[1,1]



sensitivity = (TP / (TP + FN))

print(f"Sensitivity: {sensitivity}")

specificity = ( TN / (TN + FP))

print(f"Specificity: {specificity}")