import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1}).astype(int)

df['Embarked'].fillna('S', inplace=True)

df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

df_test['Embarked'].fillna('S', inplace=True)

df_test['Embarked'] = df_test['Embarked'].map(

    {'S': 0, 'C': 1, 'Q': 2}).astype(int)

df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

df_test['Has_Cabin'] = df_test["Cabin"].apply(

    lambda x: 0 if type(x) == float else 1)





encoder = LabelEncoder()

df_board = df['Embarked'].apply(str)

df_board_test = df_test['Embarked'].apply(str)

df_board_encoded = encoder.fit_transform(df_board)

df_board_test_encoded = encoder.fit_transform(df_board_test)

# df_test['encoded_embark'] = df_board_test_encoded

# df['encoded_embark'] = df_board_encoded

df['Age'].isnull().values.any()  # Checks for nans in age and returns true

df_test['Age'].isnull().values.any()  # Checks for nans in age and returns true

med = df['Age'].median()

med_test = df_test['Age'].median()



df['Age'].fillna(med, inplace=True)

df_test['Age'].fillna(med_test, inplace=True)



# Checks for nans in Fare and returns true

df_test['Fare'].isnull().values.any()

med = df_test['Fare'].median()

df_test['Fare'].fillna(med, inplace=True)



y = df['Survived']

X = df.drop(['Survived', 'PassengerId', 'Ticket',

             'Name', 'Cabin', 'Embarked'], axis=1)

X_test = df_test.drop(['PassengerId', 'Ticket',

                       'Name', 'Cabin', 'Embarked'], axis=1)





def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)





def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--",

             label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.xlabel("Threshold", fontsize=16)

    plt.legend(loc="upper left", fontsize=16)

    plt.ylim([0, 1])



model = SVC(kernel="rbf", C=10)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model.fit(X_scaled, y)



scores = cross_val_score(model, X, y,

                         scoring="neg_mean_squared_error", cv=2,

                         verbose=1, n_jobs=-1)



rmse_scores = np.sqrt(-scores)

rmse_scores.mean()

rmse_scores.std()





y_scores = cross_val_predict(model, X, y, cv=3)



precisions, recalls, thresholds = precision_recall_curve(y, y_scores)



plt.figure(figsize=(8, 4))

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.xlim([0, 1])

plt.show()



fpr, tpr, thresholds = roc_curve(y, y_scores)



plt.figure(figsize=(8, 6))

plot_roc_curve(fpr, tpr)

plt.show()



roc_auc_score(y, y_scores)

X_test_scaled = scaler.fit_transform(X_test)



ytest = model.predict(X_test_scaled)

df_test['Survived'] = ytest

print('Test shape OK') if df_test.shape[0] == ytest.shape[0] else print('Oops')

df_test[['PassengerId', 'Survived']].to_csv(

    'titanic_submission.csv.gz', index=False, compression='gzip')


