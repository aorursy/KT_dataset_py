# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier ,VotingClassifier

import matplotlib.pyplot as plt

import seaborn as sns

# roc curve and auc score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
# loading dataset 

training_v2 = pd.read_csv("../input/widsdatathon2020/training_v2.csv")

test = pd.read_csv("../input/widsdatathon2020/unlabeled.csv")
# creating independent features X and dependant feature Y

y = training_v2['hospital_death']

X = training_v2

X = training_v2.drop('hospital_death',axis = 1)

test = test.drop('hospital_death',axis = 1)
# Remove Features with more than 75 percent missing values

train_missing = (X.isnull().sum() / len(X)).sort_values(ascending = False)

train_missing = train_missing.index[train_missing > 0.75]

X = X.drop(columns = train_missing)

test = test.drop(columns = train_missing)
categoricals_features = ['hospital_id','ethnicity','gender','hospital_admit_source','icu_admit_source','icu_stay_type','icu_type','apache_3j_bodysystem','apache_2_bodysystem']

X = X.drop(columns = categoricals_features)

test = test.drop(columns = categoricals_features)
# Imputation transformer for completing missing values.

my_imputer = SimpleImputer()

new_data = pd.DataFrame(my_imputer.fit_transform(X))

test_data = pd.DataFrame(my_imputer.fit_transform(test))

new_data.columns = X.columns

test_data.columns = test.columns

X= new_data

test = test_data
# Split into training and validation set

X_train, valid_features, Y_train, valid_y = train_test_split(X, y, test_size = 0.25, random_state = 1)
# Gradient Boosting Classifier

GBC = GradientBoostingClassifier(random_state=1)

# Random Forest Classifier

RFC = RandomForestClassifier(n_estimators=100)

# Voting Classifier with soft voting 

votingC = VotingClassifier(estimators=[('rfc', RFC),('gbc',GBC)], voting='soft')

votingC = votingC.fit(X_train, Y_train)
predict_y = votingC.predict(valid_features)
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
probs = votingC.predict_proba(valid_features)

probs = probs[:, 1]

auc = roc_auc_score(valid_y, probs)

fpr, tpr, thresholds = roc_curve(valid_y, probs)

plot_roc_curve(fpr, tpr)

print("AUC-ROC :",auc)

test1 = test.copy()

test1["hospital_death"] = votingC.predict(test)

test1[["encounter_id","hospital_death"]].to_csv("submission5.csv",index=False)

test1[["encounter_id","hospital_death"]].head()