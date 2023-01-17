import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline
df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values="?", index_col=['Id'])

df_train.info()

df_train.sample(5)
df_train.isnull().sum().sort_values(ascending=False)
df_train.shape
df_train.groupby(['workclass']).size().sort_values(ascending = False).plot(kind='bar', figsize=(12, 5))
df_train['workclass'] = df_train['workclass'].fillna(df_train['workclass'].mode()[0])
df_train.groupby(['occupation']).size().sort_values(ascending = False).plot(kind='bar', figsize=(12, 5))
df_train.groupby(['native.country']).size().sort_values(ascending = False).plot(kind="bar", figsize=(12, 5))
df_train['native.country'] = df_train['native.country'].fillna(df_train['native.country'].mode()[0])
df_train = df_train.dropna()

df_train.shape
df_train["income"] = df_train["income"].map({"<=50K": 0, ">50K":1})

df_train["sex.num"] = df_train["sex"].map({"Male": 1, "Female":0})
df_train['income'].corr(df_train['sex.num'])
df_train['native.country'].unique()
len(df_train['native.country'].unique())
europe = ['Hungary', 'Ireland', 'England', 'France', 'Portugal', 'Scotland', 'Italy', 'Dominican-Republic', 'Germany', 'Greece', 'Yugoslavia', 'Poland', 'Holand-Netherlands']

africa = ['South', 'Cambodia', 'Laos']

south_central_america = ['Jamaica', 'Mexico', 'Guatemala', 'El-Salvador', 'Haiti', 'Philippines', 'Cuba', 'Puerto-Rico', 'Nicaragua', 'Ecuador', 'Columbia', 'Peru', 'Honduras', 'Trinadad&Tobago']

north_america = ['Canada']

asia = ['Iran', 'India', 'Taiwan', 'Vietnam', 'Hong', 'Japan', 'China', 'Thailand', 'Outlying-US(Guam-USVI-etc)']
len(europe)+len(africa)+len(south_central_america)+len(north_america)+len(asia)
def nationality(country):

    if country in europe:

        return 'Europe'

    elif country in africa:

        return 'Africa'

    elif country in south_central_america:

        return 'SouthCentralAmerica'

    elif country in north_america:

        return 'NorthAmerica'

    elif country in asia:

        return 'Asia'

    elif country == 'United-States':

        return 'United-States'

    else:

        print(country)

    
df_train['nationality'] = df_train['native.country'].apply(nationality)
df_train.groupby(['nationality']).mean()['income'].sort_values(ascending=True)
def nationalityNum(nationality):

    if nationality == 'SouthCentralAmerica':

        return 0

    if nationality == 'Africa':

        return 1

    if nationality == 'United-States':

        return 2

    if nationality == 'Europe':

        return 3

    if nationality == 'Asia':

        return 4

    if nationality == 'NorthAmerica':

        return 5
df_train['nationality.num'] = df_train['nationality'].apply(nationalityNum)
df_train['income'].corr(df_train['nationality.num'])
df_train.groupby(['race']).size().sort_values(ascending = False).plot(kind="bar")
df_train.groupby(['race']).mean()['income'].sort_values(ascending=True)
def raceNum(race):

    if race == 'Other':

        return 0

    elif race == 'Amer-Indian-Eskimo':

        return 1

    elif race == 'Black':

        return 2

    elif race == 'White':

        return 3

    elif race == 'Asian-Pac-Islander':

        return 4
df_train['race.num'] = df_train['race'].apply(raceNum)
df_train['income'].corr(df_train['race.num'])
df_train.groupby(['relationship']).mean()['income'].sort_values(ascending=True)
def relationshipNum(relationship):

    if relationship == 'Own-child':

        return 1

    elif relationship == 'Other-relative':

        return 2

    elif relationship == 'Unmarried':

        return 3

    elif relationship == 'Not-in-family':

        return 4

    elif relationship == 'Husband':

        return 5

    elif relationship == 'Wife':

        return 6
df_train['relationship.num'] = df_train['relationship'].apply(relationshipNum)
df_train.groupby(['occupation']).mean()['income'].sort_values(ascending=True)
def occupationNum(occupation):

    if occupation == 'Priv-house-serv':

        return 0

    elif occupation == 'Other-service':

        return 1

    elif occupation == 'Handlers-cleaners':

        return 2

    elif occupation == 'Armed-Forces':

        return 3

    elif occupation == 'Farming-fishing':

        return 4

    elif occupation == 'Machine-op-inspct':

        return 5

    elif occupation == 'Adm-clerical':

        return 6

    elif occupation == 'Transport-moving':

        return 7

    elif occupation == 'Craft-repair':

        return 8

    elif occupation == 'Sales':

        return 9

    elif occupation == 'Tech-support':

        return 10

    elif occupation == 'Protective-serv':

        return 11

    elif occupation == 'Prof-specialty':

        return 12

    elif occupation == 'Exec-managerial':

        return 13
df_train['occupation.num'] = df_train['occupation'].apply(occupationNum)
df_train.groupby(['workclass']).mean()['income'].sort_values(ascending=True)
def workclassNum(workclass):

    if workclass == 'Without-pay' or workclass == 'Never-worked':

        return 0

    elif workclass == 'Private':

        return 1

    elif workclass == 'State-gov':

        return 2

    elif workclass == 'Self-emp-not-inc':

        return 3

    elif workclass == 'Local-gov':

        return 4

    elif workclass == 'Federal-gov':

        return 5

    elif workclass == 'Self-emp-inc':

        return 6
df_train['workclass.num'] = df_train['workclass'].apply(workclassNum)
df_train.groupby(['education.num']).mean()['income'].sort_values(ascending=True)
def educationNumCor(educationNum):

    if educationNum == 1:

        return 0

    elif educationNum == 2:

        return 1

    elif educationNum == 3:

        return 2

    elif educationNum == 5:

        return 3

    elif educationNum == 7:

        return 4

    elif educationNum == 4:

        return 5

    elif educationNum == 6:

        return 6

    elif educationNum == 8:

        return 7

    elif educationNum == 9:

        return 8

    elif educationNum == 10:

        return 9

    elif educationNum == 12:

        return 10

    elif educationNum == 11:

        return 11

    elif educationNum == 13:

        return 12

    elif educationNum == 14:

        return 13

    elif educationNum == 16:

        return 14

    elif educationNum == 15:

        return 15
df_train['education.num.cor'] = df_train['education.num'].apply(educationNumCor)
df_train['income'].corr(df_train['fnlwgt'])
df_train = df_train.drop(['fnlwgt'], axis=1)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 9))



ax[0].pie(df_train['sex.num'].value_counts(), autopct='%1.1f%%')

ax[0].set_title("Presence of each sex")

ax[0].legend(["Male", "Female"], title="Income")



ax[1].pie(df_train[df_train['sex.num'] == 1]['income'].value_counts(), autopct='%1.1f%%')

ax[1].set_title("Male income")

ax[1].legend(["<=50K", ">50K"], title="Income")



ax[2].pie(df_train[df_train['sex.num'] == 0]['income'].value_counts(), autopct='%1.1f%%')

ax[2].set_title("Female income")

ax[2].legend(["<=50K", ">50K"], title="Income")
df_train.groupby(['education', 'sex']).mean()['income']
df_train.sort_values(['occupation', 'sex']).groupby(['occupation', 'sex']).mean()['income']
df_train.groupby(['education']).mean()['income'].sort_values(ascending=True).plot(kind='bar')
fig, ax = plt.subplots(figsize=(20, 13))

sns.heatmap(df_train.corr(), annot=True, vmin=-1, vmax=1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

import statistics as st
used_columns = ['sex.num',

                'education.num.cor',

                'capital.gain',

                'capital.loss',

                'hours.per.week',

                'nationality.num',

                'occupation.num',

                'relationship.num',

                'workclass.num',

                'race.num',

                'income']

base = df_train[used_columns]
X = base.drop(['income'], axis = 1)

Y = base['income']
k_min = 10

k_max = 30

folds = 10



score_mean_best = 0

score_min_best = 0

score_max_best = 0



scores = []



for i in range(k_min, k_max+1):

    knn = KNeighborsClassifier(n_neighbors=i)

    score_k = cross_val_score(knn, X, Y, cv=folds)

    scores.append([i, score_k])

    

    if st.mean(score_k) > score_mean_best:

        score_mean_best = st.mean(score_k)

        print_aux = "  --  Best k until now"

        k_best = i

    else:

        print_aux = ""

    

    print("For k =", i, " ---  Score:", round(st.mean(score_k), 4)*100, "%", "+/-", round(st.pstdev(score_k), 4)*100, print_aux)



k = k_best

print("\n", k, "is the best k found in the range", k_min, "to", k_max)
knn = KNeighborsClassifier(n_neighbors=k).fit(X, Y)
df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values="?", index_col=['Id'])

df_test.info()

df_test.sample(5)
df_test.isnull().sum().sort_values(ascending=False)
df_test.shape
df_test['workclass'] = df_test['workclass'].fillna(df_test['workclass'].mode()[0])

df_test['native.country'] = df_test['native.country'].fillna(df_test['native.country'].mode()[0])

df_test['occupation'] = df_test['occupation'].fillna(df_test['occupation'].mode()[0])

df_test.shape
df_test["sex.num"] = df_test["sex"].map({"Male": 1, "Female":0})



df_test['nationality'] = df_test['native.country'].apply(nationality)

df_test['nationality.num'] = df_test['nationality'].apply(nationalityNum)



df_test['race.num'] = df_test['race'].apply(raceNum)



df_test['relationship.num'] = df_test['relationship'].apply(relationshipNum)



df_test['occupation.num'] = df_test['occupation'].apply(occupationNum)



df_test['workclass.num'] = df_test['workclass'].apply(workclassNum)



df_test['education.num.cor'] = df_test['education.num'].apply(educationNumCor)



df_test = df_test.drop(['fnlwgt'], axis=1)
used_columns.remove('income')

base_test = df_test[used_columns]
X_prev = base_test

X_prev.info()
predictions = pd.DataFrame({'Income': knn.predict(X_prev)})

predictions['Income'] = predictions['Income'].map({0 : "<=50K", 1 : ">50K"})
df_train["income"] = df_train["income"].map({"<=50K": 0, ">50K":1})

predictions.to_csv("submission.csv", index = True, index_label = 'Id')