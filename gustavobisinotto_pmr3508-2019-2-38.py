

import numpy as np 

import pandas as pd

import seaborn as sns

from sklearn import preprocessing

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from time import time

from sklearn.model_selection import cross_val_score 
data_frame = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv", sep=r'\s*,\s*', engine='python',

na_values="?")

test_df = pd.read_csv('../input/adult-pmr3508/test_data.csv', sep=r'\s*,\s*', engine='python', 

na_values='?')

train_df = data_frame.copy(deep=True)

train_df.shape
train_df.head()
num_missing = []

for column in train_df.columns:

    aux_missing = []

    sum_missing = 0

    sum_missing += train_df[column].isnull().sum() + train_df[column].isna().sum()

    aux_missing.append(column)

    aux_missing.append(100*sum_missing/train_df.shape[0])

    num_missing.append(aux_missing)



missing_df = pd.DataFrame(num_missing, columns = ['Feature', 'Frequency']).sort_values(by=['Frequency'],

                                                                                       ascending=False) 

print("Frequency of Missing Data on Features")

missing_df.head()
train_df['occupation'].value_counts().plot(kind = 'pie')
train_df['workclass'].value_counts().plot(kind = 'pie')
train_df['native.country'].value_counts().plot(kind = 'pie')
train_df.loc[train_df['native.country'].isna(), 'native.country'] = "United-States"

train_df.loc[train_df['native.country'].isnull(), 'native.country'] = "United-States"



train_df.loc[train_df['workclass'].isna(), 'workclass'] = "Private"

train_df.loc[train_df['workclass'].isnull(), 'workclass'] = "Private"



train_df.loc[train_df['occupation'].isna(), 'occupation'] = "Unknown"

train_df.loc[train_df['occupation'].isnull(), 'occupation'] = "Unknown"



test_df.loc[test_df['native.country'].isna(), 'native.country'] = "United-States"

test_df.loc[test_df['native.country'].isnull(), 'native.country'] = "United-States"



test_df.loc[test_df['workclass'].isna(), 'workclass'] = "Private"

test_df.loc[test_df['workclass'].isnull(), 'workclass'] = "Private"



test_df.loc[test_df['occupation'].isna(), 'occupation'] = "Unknown"

test_df.loc[test_df['occupation'].isnull(), 'occupation'] = "Unknown"



num_missing = []

for column in train_df.columns:

    aux_missing = []

    sum_missing = 0

    sum_missing += train_df[column].isnull().sum() + train_df[column].isna().sum()

    aux_missing.append(column)

    aux_missing.append(100*sum_missing/train_df.shape[0])

    num_missing.append(aux_missing)



missing_df = pd.DataFrame(num_missing, columns = ['Feature', 'Frequency']).sort_values(by=['Frequency'],

                                                                                       ascending=False) 

print("Frequency of Missing Data on Features")

missing_df.head()
train_df.loc[train_df['native.country'] != "United-States", 'native.country'] = 0

train_df.loc[train_df['native.country'] == "United-States", 'native.country'] = 1



test_df.loc[test_df['native.country'] != "United-States", 'native.country'] = 0

test_df.loc[test_df['native.country'] == "United-States", 'native.country'] = 1



train_df.loc[train_df['age'] <= 10, 'age'] = 10

train_df.loc[(train_df['age'] > 10) & (train_df['age'] < 15), 'age'] = 15

train_df.loc[(train_df['age'] > 15) & (train_df['age'] < 20), 'age'] = 20

train_df.loc[(train_df['age'] > 20) & (train_df['age'] < 25), 'age'] = 25

train_df.loc[(train_df['age'] > 25) & (train_df['age'] < 30), 'age'] = 30

train_df.loc[(train_df['age'] > 30) & (train_df['age'] < 35), 'age'] = 35

train_df.loc[(train_df['age'] > 35) & (train_df['age'] < 40), 'age'] = 40

train_df.loc[(train_df['age'] > 40) & (train_df['age'] < 45), 'age'] = 45

train_df.loc[(train_df['age'] > 45) & (train_df['age'] < 50), 'age'] = 50

train_df.loc[(train_df['age'] > 50) & (train_df['age'] < 55), 'age'] = 55

train_df.loc[(train_df['age'] > 55) & (train_df['age'] < 60), 'age'] = 60

train_df.loc[(train_df['age'] > 60) & (train_df['age'] < 65), 'age'] = 65

train_df.loc[(train_df['age'] > 65), 'age'] = 70



test_df.loc[test_df['age'] <= 10, 'age'] = 10

test_df.loc[(test_df['age'] > 10) & (test_df['age'] < 15), 'age'] = 15

test_df.loc[(test_df['age'] > 15) & (test_df['age'] < 20), 'age'] = 20

test_df.loc[(test_df['age'] > 20) & (test_df['age'] < 25), 'age'] = 25

test_df.loc[(test_df['age'] > 25) & (test_df['age'] < 30), 'age'] = 30

test_df.loc[(test_df['age'] > 30) & (test_df['age'] < 35), 'age'] = 35

test_df.loc[(test_df['age'] > 35) & (test_df['age'] < 40), 'age'] = 40

test_df.loc[(test_df['age'] > 40) & (test_df['age'] < 45), 'age'] = 45

test_df.loc[(test_df['age'] > 45) & (test_df['age'] < 50), 'age'] = 50

test_df.loc[(test_df['age'] > 50) & (test_df['age'] < 55), 'age'] = 55

test_df.loc[(test_df['age'] > 55) & (test_df['age'] < 60), 'age'] = 60

test_df.loc[(test_df['age'] > 60) & (test_df['age'] < 65), 'age'] = 65

test_df.loc[(test_df['age'] > 65), 'age'] = 70



train_df_num = train_df.apply(preprocessing.LabelEncoder().fit_transform)

test_df_num = test_df.apply(preprocessing.LabelEncoder().fit_transform)

train_df_num.head()
train_df['sex'].value_counts().plot(kind = 'pie')
train_df['age'].value_counts().plot(kind = 'pie')
train_df['race'].value_counts().plot(kind = 'pie')
def bar_plots(attribute):

    fig, axes = plt.subplots(nrows=1,ncols=2)

    low_df = train_df[train_df['income'] == '<=50K']

    high_df = train_df[train_df['income'] == '>50K']

    low_df[attribute].value_counts().plot(ax = axes[0],kind='bar', subplots=True)

    high_df[attribute].value_counts().plot(ax = axes[1],kind='bar', subplots=True)

    axes[0].title.set_text('<=50k')

    axes[1].title.set_text('>50k')

    plt.show()
bar_plots('sex')
bar_plots('age')
bar_plots('occupation')
bar_plots('education')
bar_plots('marital.status')
sns.heatmap(train_df_num.corr(), annot=True, vmin=-1, vmax=1)
train_df_num.corr()
sns.boxplot(x='income', y='age', data=pd.concat([train_df['age'], train_df['income']], axis=1), notch = True)
sns.boxplot(x='income', y='education.num', data=pd.concat([train_df['education.num'], train_df['income']], axis=1), notch = True)
sns.boxplot(x='income', y='hours.per.week', data=pd.concat([train_df['hours.per.week'], train_df['income']], axis=1), notch = True)
X_train = train_df_num.iloc[:, 1:15] # não considerando o atributo Id

Y_train = train_df.iloc[:,15]



X_test = test_df_num.iloc[:, 1:15]
elapsed_train_time = []

elapsed_test_time = []

accuracy = []
start_time = time()

logistic = LogisticRegression(solver = 'liblinear')

logistic_cv = cross_val_score(logistic, X_train, Y_train, cv = 10)



logistic.fit(X_train, Y_train)

final_time = time()



elapsed_train_time.append(final_time-start_time)

accuracy.append(logistic_cv.mean())

logistic_cv.mean()

start_time = time()

Y_test_logistic = logistic.predict(X_test)

final_time = time()



elapsed_test_time.append(final_time-start_time)
start_time = time()

rand_forest = RandomForestClassifier(n_estimators = 700, max_depth = 10)

rand_forest_cv = cross_val_score(rand_forest, X_train, Y_train, cv = 10)



rand_forest.fit(X_train, Y_train)

final_time = time()



elapsed_train_time.append(final_time-start_time)

accuracy.append(rand_forest_cv.mean())

rand_forest_cv.mean()
start_time = time()

Y_test_rand_forest = rand_forest.predict(X_test)

final_time = time()



elapsed_test_time.append(final_time-start_time)
start_time = time()

ada_boost = AdaBoostClassifier()

ada_boost_cv = cross_val_score(ada_boost, X_train, Y_train, cv = 10)



ada_boost.fit(X_train, Y_train)

final_time = time()



elapsed_train_time.append(final_time-start_time)

accuracy.append(ada_boost_cv.mean())

ada_boost_cv.mean()
start_time = time()

Y_test_ada_boost = ada_boost.predict(X_test)

final_time = time()



elapsed_test_time.append(final_time-start_time)
x_label = ["Logistic Regression", "Random Forest", "AdaBoost"]

plt.figure()

plt.title('Cross Validation Accuracy')

plt.scatter(x_label, accuracy)

plt.grid(True)

plt.show()
plt.figure()

plt.title('Elapsed Training Time [s]')

plt.scatter(x_label, elapsed_train_time)

plt.grid(True)

plt.show()
plt.figure()

plt.title('Elapsed Testing Time [s]')

plt.scatter(x_label, elapsed_test_time)

plt.grid(True)

plt.show()
elapsed_train_time
predict_y_logistic = pd.DataFrame(Y_test_logistic, columns=['Income'])

predict_y_rand_forest = pd.DataFrame(Y_test_rand_forest, columns=['Income'])

predict_y_ada_boost = pd.DataFrame(Y_test_ada_boost, columns=['Income'])



predict_y_logistic.to_csv("subimission_logistic_regression.csv", index_label = 'Id')

predict_y_rand_forest.to_csv("subimission_random_forest.csv", index_label = 'Id')

predict_y_ada_boost.to_csv("subimission_ada_boost.csv", index_label = 'Id')