# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')



dfMerge = train.merge(labels, how="left", on=[ "game_session"])
dfmergeDropped = dfMerge.drop(columns=['game_session','event_id','installation_id_x', 'title_y','timestamp','event_data','event_count','installation_id_y','num_correct','num_incorrect','num_correct','accuracy'])

dfmergeDropped = dfmergeDropped.dropna()

dfmergeDropped
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

test = test.groupby(['installation_id']).last()

test.reset_index(level=0, inplace=True)

test
test
testDropped = test.drop(columns=['game_session','event_id','timestamp','event_data','event_count','installation_id'])

testDropped = testDropped.rename(columns={'title': "title_x"})

testDropped
del [train]
def handle_non_numerical_data(df):

    columns = df.columns.values



    for column in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]



        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()

            unique_elements = set(column_contents)

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x+=1



            df[column] = list(map(convert_to_int, df[column]))



    return df



handle_non_numerical_data(testDropped)

handle_non_numerical_data(dfmergeDropped)
dfmergeDropped = dfmergeDropped.dropna()
dfmergeDropped['accuracy_group'] = pd.to_numeric(dfmergeDropped.accuracy_group,downcast='signed')
dfmergeDropped
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor





Y_train = dfmergeDropped[('accuracy_group')]

X_train = dfmergeDropped.drop(columns=['accuracy_group'])



X_test = testDropped

X_train.shape, Y_train.shape, X_test.shape



# Create the model with 10 trees

clf = RandomForestRegressor(n_estimators = 20, random_state = 31415)



# Fit on training data

clf.fit(X_train, Y_train)



prediction = clf.predict(X_test)







submission = pd.DataFrame({

        "installation_id": test["installation_id"],

        "accuracy_group": prediction

    })



#submission.to_csv("submission.csv", index=False)
submission
import numpy as np

submission["accuracy_group"] = round(submission["accuracy_group"])



submission["accuracy_group"] = np.array(submission["accuracy_group"],dtype=int)



submission

import matplotlib.pyplot as plt



plt.rcParams.update({'font.size': 16})



se = submission.groupby(['accuracy_group'])['accuracy_group'].count()

se.plot.bar(stacked=True, rot=0, figsize=(12,10))

plt.title("Counts of accuracy group")

plt.show()
submission
submission = pd.DataFrame({

        "installation_id": submission["installation_id"],

        "accuracy_group": submission['accuracy_group']

    })



submission.to_csv("submission.csv", index=False)