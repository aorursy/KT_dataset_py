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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.reset_index(inplace=True)

df
measure_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

labels = ['Survived']



train_df_x = df[measure_features]

train_df_y = df[labels]
clean_enc_dic = {'Sex': {'male':0, 'female':1}, 'Embarked': {'S': 0, 'C':1, 'Q':2}}

train_df_x.replace(clean_enc_dic, inplace=True)

train_df_x
train_df_x.isna().sum()
for x in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:

    train_df_x[x].replace(np.NaN, train_df_x[x].median(), inplace=True)



train_df_x['Age'].replace(np.NaN, train_df_x['Age'].mean(), inplace=True)

    

train_df_x.isnull().sum()
from sklearn.svm import SVC

model = SVC(kernel='linear')

model.fit(train_df_x, train_df_y)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



test_df.reset_index(inplace=True)



test_df_x = test_df[measure_features]



test_df_x.replace(clean_enc_dic, inplace=True)



for x in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:

    test_df_x[x].replace(np.NaN, test_df_x[x].median(), inplace=True)



test_df_x['Age'].replace(np.NaN, test_df_x['Age'].mean(), inplace=True)
predict_y = model.predict(test_df_x)

len(predict_y)
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived':predict_y})

output.head()
output.to_csv('my_submission.csv', index=False)

print("Submission Saved")