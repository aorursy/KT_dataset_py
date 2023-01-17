# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')   



train_df.describe(include='all')
train_df.head()
## clean existing features

def clean_data(df):

    clean_df = df

    

    # age

    clean_df['Age'] = clean_df['Age'].fillna(clean_df['Age'].median())



    # sex

    clean_df.loc[clean_df['Sex'] == 'male', 'Sex'] = 0

    clean_df.loc[clean_df['Sex'] == 'female', 'Sex'] = 1



    # embarked

    clean_df['Embarked'] = clean_df['Embarked'].fillna('S')

    clean_df.loc[clean_df["Embarked"] == "S", "Embarked"] = 0

    clean_df.loc[clean_df["Embarked"] == "C", "Embarked"] = 1

    clean_df.loc[clean_df["Embarked"] == "Q", "Embarked"] = 2



    # fare

    clean_df['Fare'] = clean_df['Fare'].fillna(clean_df['Fare'].median())

    

    ## add new features



    # family size

    clean_df['FamilySize'] = clean_df['SibSp'] + clean_df['Parch']



    # name length

    clean_df['NameLength'] = clean_df['Name'].apply(lambda x: len(x))

    return clean_df



clean_train_df = clean_data(train_df)

clean_test_df = clean_data(test_df)



clean_train_df.describe(include='all')

# find out which features are the most predictive



import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize"]



selector = SelectKBest(f_classif, k=5)

selector.fit(clean_train_df[predictors], clean_train_df["Survived"])



scores = -np.log10(selector.pvalues_)



plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation='vertical')

plt.show()



# looks like Pclass, Sex, and Fare are the best, so lets make a model with those
# lets do cross validation on random forest predictions

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier



predictors = ["Pclass", "Sex", "Fare"]



alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)



kf = model_selection.KFold(n_splits=3, random_state=1)



scores = model_selection.cross_val_score(alg, clean_train_df[predictors], clean_train_df["Survived"], cv=kf)



# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
# train model on training set

alg.fit(clean_train_df[predictors], clean_train_df["Survived"])



# make predictions on test set

predictions = alg.predict(clean_test_df[predictors])



# create a csv file of PassengerId, Survived

submission = pd.DataFrame({

        "PassengerId": clean_test_df["PassengerId"],

        "Survived": predictions

    })



submission.to_csv('titanic1.csv', index=False)

# Verify our file was written

print(check_output(["ls"]).decode("utf8"))



print(check_output(["head", "titanic1.csv"]).decode("utf8"))



# submission page: https://www.kaggle.com/c/titanic/submissions/attach

# files can be downloaded here: https://www.kaggle.com/ericgustavson/titanic/simple-random-forest/output