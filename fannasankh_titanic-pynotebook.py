# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import minmax_scale

import seaborn as sns

from sklearn.feature_selection import RFECV



from sklearn.model_selection import GridSearchCV





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



holdout = pd.read_csv("../input/test.csv")

holdout_shape = holdout.shape



train = pd.read_csv("../input/train.csv")

train_shape = train.shape
sex_pivot = train.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

plt.show()



Pclass_pivot = train.pivot_table(index="Pclass",values="Survived")

Pclass_pivot.plot.bar()

plt.show()
def process_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df
cut_points = [-1,0,5,12,18,35,60,100]

label_names = ["Missing","Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]



train = process_age(train,cut_points,label_names)

holdout = process_age(holdout,cut_points,label_names)



Age_categories_pivot = train.pivot_table(index="Age_categories",values="Survived")

Age_categories_pivot.plot.bar()

plt.show()
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df
column_names = ['Age_categories', 'Pclass', 'Sex']



for name in column_names:

    train = create_dummies(train,name);

    holdout = create_dummies(holdout,name);

    

print(train.columns)
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']





lr = LogisticRegression()

lr.fit(train[columns], train['Survived'])
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',

       'Age_categories_Missing','Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior']



all_x = train[columns]

all_y = train['Survived']



train_X, test_X, train_y, test_y = train_test_split(

    all_x, all_y, test_size=0.2,random_state=0)
lr = LogisticRegression()

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)



accuracy = accuracy_score(test_y, predictions)

print(accuracy)
lr = LogisticRegression()

scores = cross_val_score(lr, all_x, all_y, cv=10)

accuracy = np.mean(scores)



print(scores)

print(accuracy)
lr = LogisticRegression()

lr.fit(all_x, all_y)

holdout_predictions = lr.predict(holdout[columns])
#holdout_ids = holdout["PassengerId"]

#submission_df = {"PassengerId": holdout_ids,

#                 "Survived": holdout_predictions}

#submission = pd.DataFrame(submission_df)
#holdout_ids = holdout["PassengerId"]

#submission_df = {"PassengerId": holdout_ids,

#                 "Survived": holdout_predictions}

#submission = pd.DataFrame(submission_df)

#submission.to_csv('titanic.csv', index=False)
columns = ['SibSp','Parch','Fare','Cabin','Embarked']

train[columns].describe(include='all',percentiles=[])
holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())



train["Embarked"] = train["Embarked"].fillna("S");

holdout ["Embarked"] = holdout["Embarked"].fillna("S");



train = create_dummies(train, "Embarked")

holdout = create_dummies(holdout, "Embarked")



columns = ["SibSp", "Parch", "Fare"]

new_columns = ["SibSp_scaled", "Parch_scaled", "Fare_scaled"]



for column in new_columns:

    train[column] = '';

    holdout[column] = '';

    

train[new_columns] = minmax_scale(train[columns])

holdout[new_columns] = minmax_scale(holdout[columns])

columns = ['Age_categories_Missing', 'Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'SibSp_scaled', 'Parch_scaled', 'Fare_scaled']



lr = LogisticRegression()

lr.fit(train[columns], train["Survived"])



coefficients = lr.coef_



feature_importance = pd.Series(coefficients[0], index=columns)



feature_importance.plot.barh()



ordered_feature_importance = feature_importance.abs().sort_values()

ordered_feature_importance.plot.barh()

plt.show()
lr = LogisticRegression()

scores = cross_val_score(lr, train[columns], train["Survived"], cv=10)

accuracy = np.mean(scores)

print(accuracy)
columns = ['Age_categories_Infant', 'SibSp_scaled', 'Sex_female', 'Sex_male',

       'Pclass_1', 'Pclass_3', 'Age_categories_Senior', 'Parch_scaled']



all_X = train[columns]

all_y = train['Survived']



lr = LogisticRegression()

lr.fit(all_X,all_y)

holdout_predictions = lr.predict(holdout[columns])
#holdout_ids = holdout["PassengerId"]

#submission_df = {"PassengerId": holdout_ids,

#                 "Survived": holdout_predictions}

#submission = pd.DataFrame(submission_df)



#holdout_ids = holdout["PassengerId"]

#submission_df = {"PassengerId": holdout_ids,

#                 "Survived": holdout_predictions}

#submission = pd.DataFrame(submission_df)

#submission.to_csv('titanic.csv', index=False)
def process_fare(df,cut_points,label_names):

    df["Fare"] = df["Fare"].fillna(-0.5)

    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)

    return df
cut_points = [0,12,50,100,1000]

label_names = ["0-12","12-50","50-100","100+"]



train  = process_fare(train ,cut_points,label_names)

holdout  = process_fare(holdout ,cut_points,label_names)



train  = create_dummies(train , "Fare_categories")

holdout = create_dummies(holdout, "Fare_categories")



titles = {

    "Mr" :         "Mr",

    "Mme":         "Mrs",

    "Ms":          "Mrs",

    "Mrs" :        "Mrs",

    "Master" :     "Master",

    "Mlle":        "Miss",

    "Miss" :       "Miss",

    "Capt":        "Officer",

    "Col":         "Officer",

    "Major":       "Officer",

    "Dr":          "Officer",

    "Rev":         "Officer",

    "Jonkheer":    "Royalty",

    "Don":         "Royalty",

    "Sir" :        "Royalty",

    "Countess":    "Royalty",

    "Dona":        "Royalty",

    "Lady" :       "Royalty"

}
extracted_titles = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

train["Title"] = extracted_titles.map(titles)

extracted_titles = holdout["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

holdout["Title"] = extracted_titles.map(titles)
train["Cabin_type"] = train["Cabin"].str[0]

train["Cabin_type"] = train["Cabin_type"].fillna("Unknown")



holdout["Cabin_type"] = holdout["Cabin"].str[0]

holdout["Cabin_type"] = train["Cabin_type"].fillna("Unknown")
for column in ["Title","Cabin_type"]:

    train = create_dummies(train,column)

    holdout = create_dummies(holdout,column)
def plot_correlation_heatmap(df):

    corr = df.corr()

    

    sns.set(style="white")

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)





    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
columns = ['Age_categories_Missing', 'Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'SibSp_scaled', 'Parch_scaled', 'Fare_categories_0-12',

       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',

       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',

       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',

       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_T', 'Cabin_type_Unknown']
plot_correlation_heatmap(train[columns])
columns = ['Age_categories_Missing', 'Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Young Adult',

       'Age_categories_Adult', 'Age_categories_Senior', 'Pclass_1', 'Pclass_3',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',

       'Parch_scaled', 'Fare_categories_0-12', 'Fare_categories_50-100',

       'Fare_categories_100+', 'Title_Miss', 'Title_Mr', 'Title_Mrs',

       'Title_Officer', 'Title_Royalty', 'Cabin_type_B', 'Cabin_type_C',

       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',

       'Cabin_type_T', 'Cabin_type_Unknown']
all_X = train[columns]

all_y = train["Survived"]
lr = LogisticRegression()

selector = RFECV(lr,cv=10)

selector.fit(all_X,all_y)
optimized_columns = all_X.columns[selector.support_]

print(optimized_columns)
all_X = train[optimized_columns]

all_y = train["Survived"]



lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

accuracy = np.mean(scores)

print(accuracy)
lr = LogisticRegression()

lr.fit(all_X,all_y)

holdout_predictions = lr.predict(holdout[optimized_columns])



holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv('titanic.csv', index=False)
all_X = train.drop(['Survived','PassengerId'],axis=1)[columns]

all_y = train['Survived']
lr = LogisticRegression()

scores = cross_val_score(lr, all_X, all_y, cv=10)

accuracy_lr = np.mean(scores)

print(accuracy_lr)
knn = KNeighborsClassifier(n_neighbors=1)

scores = cross_val_score(knn, all_X, all_y, cv=10)

accuracy_knn = np.mean(scores)

print(accuracy_knn)
%matplotlib inline
def plot_dict(dictionary):

    pd.Series(dictionary).plot.bar(figsize=(9,6),

                                   ylim=(0.78,0.83),rot=0)

    plt.show()
knn_scores = dict()
for k in range(1,50,2):

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, all_X, all_y, cv=10)

    accuracy_knn = np.mean(scores)

    knn_scores[k] = accuracy_knn
plot_dict(knn_scores)
hyperparameters = {

    "n_neighbors": range(1,20,2),

    "weights": ["distance", "uniform"],

    "algorithm": ['brute'],

    "p": [1,2]

}
grid = GridSearchCV(knn, param_grid=hyperparameters, cv=10)

grid.fit(all_X, all_y)
best_params = grid.best_params_

best_score = grid.best_score_

print(best_params)

print(best_score)
best_knn = grid.best_estimator_
best_knn.fit(all_X,all_y)

holdout_no_id = holdout.drop(['PassengerId'],axis=1)

holdout_predictions = best_knn.predict(holdout_no_id[columns])
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv('submission_knn.csv', index=False)
clf = RandomForestClassifier(random_state=1)

scores = cross_val_score(clf, all_X, all_y, cv=10)

accuracy_rf = np.mean(scores)

print(accuracy_rf)
hyperparameters = {"criterion": ["entropy", "gini"],

                   "max_depth": [5, 10],

                   "max_features": ["log2", "sqrt"],

                   "min_samples_leaf": [1, 5],

                   "min_samples_split": [3, 5],

                   "n_estimators": [6, 9]

}
clf = RandomForestClassifier(random_state=1)
grid = GridSearchCV(clf, param_grid=hyperparameters, cv=10)

grid.fit(all_X, all_y)
best_params = grid.best_params_

best_score = grid.best_score_

print(best_params)

print(best_score)
best_rf = grid.best_estimator_
best_rf.fit(all_X,all_y)

holdout_no_id = holdout.drop(['PassengerId'],axis=1)

holdout_predictions = best_rf.predict(holdout_no_id[columns])
holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)



holdout_ids = holdout["PassengerId"]

submission_df = {"PassengerId": holdout_ids,

                 "Survived": holdout_predictions}

submission = pd.DataFrame(submission_df)

submission.to_csv('submission_rf.csv', index=False)