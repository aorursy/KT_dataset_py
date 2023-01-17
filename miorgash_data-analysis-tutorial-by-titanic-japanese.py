# data wrangling

import numpy as np

import pandas as pd

import pandas_profiling as pdp

from collections import Counter



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.core.display import display



# modeling

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate



# evaluation

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
# Load

train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")



# fundamental statistics

display(train_df.describe(include='all'))



# generate detailed report(train)

pdp.ProfileReport(train_df)
# 前処理； raw データのうち数値型のデータをとりあえず使う（考えること少ないから）

# train データから更に validation 用のデータを切り分ける

X = train_df.loc[:, ['Fare', 'Age']].fillna(0)

y = train_df.loc[:, ['Survived']]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=123)



# train/valid データにおける陰性・陽性比率の比較

print('train:\n', y_train['Survived'].value_counts() / y_train.shape[0])

print('valid:\n', y_valid['Survived'].value_counts() / y_valid.shape[0])



# だいたい同じくらいになるので OK
# とりあえずモデルを作る

model = SVC(random_state=123)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)



# 予測値

print(y_pred)
# 精度（Accuracy）

print('score:', accuracy_score(y_valid, y_pred))
# 混合行列

sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d')
train_df['Survived'].value_counts()
# 元データ

train_df.head()
# 補完前の分布

train_df.Age.hist()
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))

guess_ages



for i in range(0, 2):

    for j in range(0, 3):

        guess_df = train_df[(train_df['Sex'] == i) & \

                              (train_df['Pclass'] == j+1)]['Age'].dropna()

        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



for i in range(0, 2):

    for j in range(0, 3):

        train_df.loc[ (train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j+1),\

                'Age'] = guess_ages[i,j]



train_df['Age'] = train_df['Age'].astype(int)



train_df.head()
train_df[train_df['Age'].notnull()].shape
# 補完後の分布

train_df.Age.hist()
df = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

plt.bar(df['Pclass'], df['Survived'])

plt.title('survival ratio by Pclasses')

plt.show()
# 例

print(pd.get_dummies([1, 2, 3]))
# Cabin は欠損が多い（687/891）

train_df.Cabin.value_counts(dropna=False).sort_values(ascending=False)
# Ticket は謎が多い（件数が最大のもので 7 件しかない，Fare と同じ値が入っているものがある， etc...）

train_df.Ticket.value_counts(dropna=False).sort_values(ascending=False)
# drop

train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId', 'Name', 'Embarked'], axis=1)

train_df.head()
# train/validation に分割

X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=123)



# 予測

model = SVC(random_state=123)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)



# 予測結果

print(y_pred)

print('freq:', Counter(y_pred), '/ratio:', np.array([v for v in Counter(y_pred).values()]) / y_pred.shape[0])

print('score:', accuracy_score(y_valid, y_pred))

sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d')
# before

train_df.describe(include = 'all')
scaler = StandardScaler()

train_df.iloc[:, 1:] = scaler.fit_transform(train_df.iloc[:, 1:])

train_df.head()
# after

train_df.describe(include = 'all')
# train/validation に分割

X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=123)



# 予測

model = SVC(random_state=123)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)



# 予測結果

print(y_pred)

print('freq:', Counter(y_pred), '/ratio:', np.array([v for v in Counter(y_pred).values()]) / y_pred.shape[0])

print('score:', accuracy_score(y_valid, y_pred))

sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d')
def show_results_info(cv_results):

    print('test scores:', cv_results['test_score'])

    print('max:', cv_results['test_score'].max())

    print('mean:', cv_results['test_score'].mean())

    print('min:', cv_results['test_score'].min())

    print('\n')



X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]



cv = KFold(n_splits = 5, shuffle = True, random_state = 123)

cv_results = cross_validate(model, X, y, cv = cv)

print('results of KFold')

show_results_info(cv_results)



cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 123)

cv_results = cross_validate(model, X, y, cv = cv)

print('results of StratifiedKFold')

show_results_info(cv_results)
# サンプリング

X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=123, stratify = y, test_size = .25)

# 予測

model = SVC(random_state=123)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)



# 予測結果

print(y_pred)

print('freq:', Counter(y_pred), '/ratio:', np.array([v for v in Counter(y_pred).values()]) / y_pred.shape[0])

print('score:', accuracy_score(y_valid, y_pred))

sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d')
len(C_array)
train_scores = []

valid_scores = []

C_array = np.arange(.5, 10.5, .5)



for C in C_array:



    model = SVC(C = C, random_state=123)

    model.fit(X_train, y_train)

    

    y_pred_train = model.predict(X_train)

    train_scores.append(accuracy_score(y_train, y_pred_train))

    

    y_pred_valid = model.predict(X_valid)

    valid_scores.append(accuracy_score(y_valid, y_pred_valid))



sns.lineplot(x = C_array, y = train_scores, color = 'skyblue', label = 'train')

sns.lineplot(x = C_array, y = valid_scores, color = 'orange', label = 'valid')

plt.show()
train_scores = []

valid_scores = []

gamma_array = np.arange(.05, 1.05, .05)



for gamma in gamma_array:



    model = SVC(C = 3, gamma = gamma, random_state=123)

    model.fit(X_train, y_train)

    

    y_pred_train = model.predict(X_train)

    train_scores.append(accuracy_score(y_train, y_pred_train))

    

    y_pred_valid = model.predict(X_valid)

    valid_scores.append(accuracy_score(y_valid, y_pred_valid))



sns.lineplot(x = gamma_array, y = train_scores, color = 'skyblue', label = 'train')

sns.lineplot(x = gamma_array, y = valid_scores, color = 'orange', label = 'valid')

plt.show()
model = SVC(C = 3, gamma = .17, random_state=123)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)



# 予測結果

print(y_pred)

print('freq:', Counter(y_pred), '/ratio:', np.array([v for v in Counter(y_pred).values()]) / y_pred.shape[0])

print('score:', accuracy_score(y_valid, y_pred))

sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d')
valid_score_matrix = np.zeros([C_array.shape[0], gamma_array.shape[0]])



for i, gamma in enumerate(gamma_array):

    for j, C in enumerate(C_array):

        model = SVC(C = C, gamma = gamma, random_state = 123)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)

        valid_score_matrix[i, j] = accuracy_score(y_valid, y_pred)



valid_score_df = pd.DataFrame(valid_score_matrix, index = gamma_array, columns = C_array)

valid_score_df.head()
plt.figure(figsize=(16, 16))

sns.heatmap(valid_score_df, annot = True, fmt = '.4f')

plt.xlabel('C')

plt.ylabel('gamma')

plt.show()
model = SVC(C = 3.5, gamma = 0.15, random_state=123) # test score: 0.66985

# model = SVC(C = 8.0, gamma = 0.1, random_state=123) # test score: 0.58851

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)



# 予測結果

print(y_pred)

print('freq:', Counter(y_pred), '/ratio:', np.array([v for v in Counter(y_pred).values()]) / y_pred.shape[0])

print('score:', accuracy_score(y_valid, y_pred))

sns.heatmap(confusion_matrix(y_valid, y_pred), annot=True, fmt='d')
def preprocess(df):

    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    guess_ages = np.zeros((2,3))

    guess_ages



    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = df[(df['Sex'] == i) & \

                                  (df['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0, 2):

        for j in range(0, 3):

            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    df['Age'] = df['Age'].astype(int)



    # drop

    df = df.drop(['Ticket', 'Cabin', 'PassengerId', 'Name', 'Embarked'], axis=1)

    

    # scaling

    scaler = StandardScaler()

    df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

    df.head()

    return df
# passengerId は Submit に必要なのでとっておく

passenger_id = test_df["PassengerId"].copy()



# 前処理（test データには Fare == nan のサンプルが存在するので暫定処置として fillna）

X_test = preprocess(test_df.fillna(0))

X_test.head()
y_pred = model.predict(X_test)

print(y_pred)

print('freq:', Counter(y_pred), '/ratio:', np.array([v for v in Counter(y_pred).values()]) / y_pred.shape[0])
submission_df = pd.DataFrame({

        "PassengerId": passenger_id,

        "Survived": y_pred

    })

display(submission_df.head())

submission_df.to_csv('./submission.csv', index=False)