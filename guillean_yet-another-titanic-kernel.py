import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read in all of the data

all_data = pd.read_csv('../input/train.csv')

all_data.head()
all_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
def prepare_data(data):

    all_data = data.copy()

    

    # One-hot encode the embarked status

    one_hot_embarked = pd.get_dummies(all_data.Embarked)

    all_data = all_data.join(one_hot_embarked)

    all_data.drop('Embarked', axis=1, inplace=True)

    

    # One-hot encode the passenger class

    one_hot_pclass = pd.get_dummies(all_data.Pclass)

    all_data = all_data.join(one_hot_pclass)

    all_data.drop('Pclass', axis=1, inplace=True)

    

    # Encode if the person had any siblings

    all_data.SibSp = all_data.SibSp > 0

    

    # Encode if the person had an assigned cabin

    all_data.Cabin = all_data.Cabin.isnull()

    

    # Encode whether or not the person was female

    all_data.Sex = all_data.Sex == 'female'

    

    # Make a new category for ages which are unknown

    all_data['age_unknown'] = all_data.Age.isnull()

    all_data.replace(np.nan, 0, inplace=True)

    

    return all_data
all_data = prepare_data(all_data)

target = all_data.Survived

all_data.drop('Survived', axis=1, inplace=True)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
n_folds = 5

kf = KFold(n_splits=n_folds, random_state=1)

best_score, best_l = -1, -1

for l in np.logspace(-5, 1, 20):

    curr_tot = 0

    for train_idx, test_idx in kf.split(all_data):

        X_train, X_test = all_data.iloc[train_idx,:], all_data.iloc[test_idx,:]

        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

        lgr = LogisticRegression(C=l)

        lgr.fit(X_train, y_train)

        curr_tot += lgr.score(X_test, y_test)

    

    curr_tot /= n_folds

    print('Current score {} with C={}'.format(curr_tot, l))

    if curr_tot > best_score:

        best_l, best_score = l, curr_tot

        

lgr_final = LogisticRegression(C=best_l)

lgr_final.fit(all_data, target)
test_df = pd.read_csv('../input/test.csv')

test_df = prepare_data(test_df)



# Get the PassengerIds we need for submission

output_df = test_df[['PassengerId']]



predict_df = pd.DataFrame(lgr_final.predict(test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)), dtype=np.int16)



output_df = output_df.join(predict_df, how='right')

output_df.columns = ['PassengerId', 'Survived']
output_df.head()