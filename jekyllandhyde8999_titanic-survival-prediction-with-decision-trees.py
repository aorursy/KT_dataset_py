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
import pickle

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



from sklearn import tree, metrics



sns.set_style('whitegrid')

fig_size = plt.rcParams['figure.figsize']



%matplotlib inline
class Mask:

    def __init__(self, df):

        self.df = df

    

    @property

    def SURVIVED_MASK(self):

        return self.df.Survived == 1

    

    @property

    def NOT_SURVIVED_MASK(self):

        return self.df.Survived == 0

    

    @property

    def CHILDREN_MASK(self):

        return self.df.Age <= 15

    

    @property

    def ADULT_MASK(self):

        return np.logical_and(self.df.Age <= 45, self.df.Age > 15)

    

    @property

    def SENIOR_ADULT_MASK(self):

        return 45 < self.df.Age

    

    @property

    def MALE_MASK(self):

        return self.df.Sex == 'male'

    

    @property

    def FEMALE_MASK(self):

        return self.df.Sex == 'female'

    

    @property

    def FIRST_CLASS_MASK(self):

        return self.df.Pclass == 1

    

    @property

    def SECOND_CLASS_MASK(self):

        return self.df.Pclass == 2

    

    @property

    def THIRD_CLASS_MASK(self):

        return self.df.Pclass == 3
def label_age_group(x: float) -> str:

    if x <= 15:

        return "Child"

    if x < 45:

        return "Adult"

    return "SeniorAdult"





def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # filling age missing values with median

    df['Age'] = df.Age.fillna(df.Age.median())

    

    # grouping age values

    df['AgeGroup'] = df.Age.apply(label_age_group)



    return df
TRAIN_FILE = "/kaggle/input/titanic/train.csv"

TEST_FILE = "/kaggle/input/titanic/test.csv"

SUBMISSION_FILE = "/kaggle/input/titanic/gender_submission.csv"
train_df = pd.read_csv(TRAIN_FILE).drop(["Cabin", "Embarked"], axis=1)

train_df.info()

train_df.head()
# Preprocessing

train_df = preprocess_data(train_df)

train_mask = Mask(train_df)
nrows=1

ncols = 2

column = "Pclass"

fig, ax = plt.subplots(nrows, ncols, figsize=(fig_size[0] * ncols * 1.25, fig_size[1] * nrows * 1.25))

_ = sns.countplot(x=column, data=train_df, ax=ax[0])

_ = sns.countplot(x=column, hue="Survived", data=train_df, ax=ax[1])

_ = plt.tight_layout()
nrows=1

ncols = 2

column = "Sex"

fig, ax = plt.subplots(nrows, ncols, figsize=(fig_size[0] * ncols * 1.25, fig_size[1] * nrows * 1.25))

_ = sns.countplot(x=column, data=train_df, ax=ax[0])

_ = sns.countplot(x=column, hue="Survived", data=train_df, ax=ax[1])

_ = plt.tight_layout()
fig = plt.figure(figsize=(fig_size[0] * 2, fig_size[1] * 1.5))



gs0 = gridspec.GridSpec(1, 2, figure=fig)

ax0 = fig.add_subplot(gs0[0, 0])

gs01 = gs0[0, 1].subgridspec(2, 1)

ax1 = fig.add_subplot(gs01[0, 0])

ax2 = fig.add_subplot(gs01[1, 0])



_ = sns.distplot(train_df.Age, ax=ax0, kde=False)

_ = ax0.set_xlabel("Age")

_ = sns.distplot(train_df.loc[train_mask.SURVIVED_MASK].Age, ax=ax1, label='survived', kde=False, color='green')

_ = sns.distplot(train_df.loc[train_mask.NOT_SURVIVED_MASK].Age, ax=ax2, label='not survived', kde=False, color='red')

_ = ax1.legend()

_ = ax2.legend()

_ = plt.tight_layout()
nrows=1

ncols = 2

fig, ax = plt.subplots(nrows, ncols, figsize=(fig_size[0] * ncols * 1.25, fig_size[1] * nrows * 1.25))

_ = sns.countplot(x="AgeGroup", data=train_df, ax=ax[0])

_ = sns.countplot(x="AgeGroup", data=train_df, hue=train_df.Survived, ax=ax[1])

_ = plt.tight_layout()
print(f"Children that survived = {train_df.loc[np.logical_and(train_mask.SURVIVED_MASK, train_mask.CHILDREN_MASK)].shape[0]}")

print(f"Children that did not survive = {train_df.loc[np.logical_and(train_mask.NOT_SURVIVED_MASK, train_mask.CHILDREN_MASK)].shape[0]}")

print()

print(f"Adults that survived = {train_df.loc[np.logical_and(train_mask.SURVIVED_MASK, train_mask.ADULT_MASK)].shape[0]}")

print(f"Adults that did not survive = {train_df.loc[np.logical_and(train_mask.NOT_SURVIVED_MASK, train_mask.ADULT_MASK)].shape[0]}")

print()

print(f"Senior adults that survived = {train_df.loc[np.logical_and(train_mask.SURVIVED_MASK, train_mask.SENIOR_ADULT_MASK)].shape[0]}")

print(f"Senior adults that did not survive = {train_df.loc[np.logical_and(train_mask.NOT_SURVIVED_MASK, train_mask.SENIOR_ADULT_MASK)].shape[0]}")
fig = plt.figure(figsize=(fig_size[0] * 2, fig_size[1] * 1.5))



gs0 = gridspec.GridSpec(1, 2, figure=fig)

ax0 = fig.add_subplot(gs0[0, 0])

gs01 = gs0[0, 1].subgridspec(2, 1)

ax1 = fig.add_subplot(gs01[0, 0])

ax2 = fig.add_subplot(gs01[1, 0])



_ = sns.distplot(train_df.Fare.values, ax=ax0, kde=False)

_ = ax0.set_xlabel("Fare")

_ = sns.distplot(train_df.loc[train_mask.SURVIVED_MASK].Fare, ax=ax1, label='survived', kde=False, color='green')

_ = sns.distplot(train_df.loc[train_mask.NOT_SURVIVED_MASK].Fare, ax=ax2, label='not survived', kde=False, color='red')

_ = ax1.legend()

_ = ax2.legend()

_ = plt.tight_layout()
nrows = 2

ncols = 3

fig, ax = plt.subplots(nrows, ncols, figsize=(fig_size[0] * ncols, fig_size[1] * nrows))

_ = sns.distplot(train_df.loc[np.logical_and(train_mask.FIRST_CLASS_MASK, train_mask.SURVIVED_MASK)].Fare, kde=False, ax=ax[0, 0])

_ = sns.distplot(train_df.loc[np.logical_and(train_mask.SECOND_CLASS_MASK, train_mask.SURVIVED_MASK)].Fare, kde=False, ax=ax[0, 1], color='orange')

_ = sns.distplot(train_df.loc[np.logical_and(train_mask.THIRD_CLASS_MASK, train_mask.SURVIVED_MASK)].Fare, kde=False, ax=ax[0, 2], color='green')

_ = sns.distplot(train_df.loc[np.logical_and(train_mask.FIRST_CLASS_MASK, train_mask.NOT_SURVIVED_MASK)].Fare, kde=False, ax=ax[1, 0])

_ = sns.distplot(train_df.loc[np.logical_and(train_mask.SECOND_CLASS_MASK, train_mask.NOT_SURVIVED_MASK)].Fare, kde=False, ax=ax[1, 1], color='orange')

_ = sns.distplot(train_df.loc[np.logical_and(train_mask.THIRD_CLASS_MASK, train_mask.NOT_SURVIVED_MASK)].Fare, kde=False, ax=ax[1, 2], color='green')

_ = ax[0, 0].set_xlabel("")

_ = ax[0, 1].set_xlabel("")

_ = ax[0, 2].set_xlabel("")

_ = ax[1, 0].set_xlabel("First Class Fare", fontsize=13)

_ = ax[1, 1].set_xlabel("Second Class Fare", fontsize=13)

_ = ax[1, 2].set_xlabel("Third Class Fare", fontsize=13)

_ = ax[0, 0].set_ylabel("Survived", fontsize=13)

_ = ax[1, 0].set_ylabel("Not Survived", fontsize=13)



_ = plt.tight_layout()
# Gathering

input_cols = ["Pclass", "Sex", "AgeGroup", "Fare"]

target_cols = ["Survived"]



input_data = train_df.loc[:, input_cols]

target_data = train_df.loc[:, target_cols].values



input_data = pd.get_dummies(input_data)

input_dummies = input_data.values
# Intialize the classifier

dtc = tree.DecisionTreeClassifier()



# Train the classifier with our training data

dtc = dtc.fit(input_dummies, target_data)



# Save the model in a pickle file

with open('titanic_decision_tree_model.pkl', mode='wb') as f:

    pickle.dump(dtc, f)
y_pred = dtc.predict(input_dummies)

y_true = np.squeeze(target_data.T)



accuracy = sum(y_pred == y_true) / y_pred.shape[0]

conf_matrix = metrics.confusion_matrix(y_true, y_pred)

precision = metrics.precision_score(y_true, y_pred)

recall = metrics.recall_score(y_true, y_pred)

f1 = metrics.f1_score(y_true, y_pred)



print(f"Accuracy: {accuracy * 100:.2f}%")

print(f"Precision: {precision:.2f}")

print(f"Recall: {recall:.2f}")

print(f"F1-Score: {f1:.2f}")



scale = 1.5

_ = plt.figure(figsize=(fig_size[0] * scale, fig_size[1] * scale))

ax = sns.heatmap(conf_matrix, annot=True, fmt='d')

_ = ax.set_xticklabels(["Not Survived", "Survived"], fontsize=13)

_ = ax.set_yticklabels(["Not Survived", "Survived"], fontsize=13)

_ = ax.set_xlabel("Predicted Labels", fontsize=15)

_ = ax.set_ylabel("Actual Labels", fontsize=15)
tree_text = tree.export_text(dtc, feature_names=list(input_data.columns))

print(tree_text)

with open("titanic_decision_tree_text.log", mode='w') as f:

    f.write(tree_text)
# scale = 2.5

# fig = plt.figure(figsize=(fig_size[0] * scale, fig_size[1] * scale))



# _ = tree.plot_tree(dtc, filled=True)

import graphviz

dot = tree.export_graphviz(dtc, out_file=None, filled=True, feature_names=list(input_data.columns), class_names=["Not Survived", "Survived"])

graph = graphviz.Source(dot, format='svg')

_ = graph.render('titanic_decision_tree_graphviz')
test_df = pd.read_csv(TEST_FILE).drop(["Cabin", "Embarked"], axis=1)

test_df.info()

test_df.head()
test_df = preprocess_data(test_df)

test_mask = Mask(test_df)

test_df.info()
test_df.loc[test_df.Fare.isnull(), :]
test_df.loc[test_df.Fare.isnull(), "Fare"] = test_df.loc[test_mask.THIRD_CLASS_MASK].Fare.mean()

test_df.info()
# Gathering

test_input_cols = ["Pclass", "Sex", "AgeGroup", "Fare"]



test_input_data = test_df.loc[:, test_input_cols]



test_input_data = pd.get_dummies(test_input_data)

test_input_dummies = test_input_data.values

test_input_data
# Feeding the test data to the model

y_pred_test = dtc.predict(test_input_dummies)
# Logging the predictions into the submission file

submission_df = pd.read_csv(SUBMISSION_FILE)

submission_df.Survived = y_pred_test

submission_df.set_index(['PassengerId']).to_csv("submission.csv")