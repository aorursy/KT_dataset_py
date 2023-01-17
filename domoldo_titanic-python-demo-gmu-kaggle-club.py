import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for making graphs

import os



from sklearn.model_selection import train_test_split



for dirname, _, filenames in os.walk('/kaggle/input/titanic'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/titanic/train.csv")

train_df.head()
sns.distplot(train_df[(train_df["Survived"]==0) & (train_df["Age"].isna() == False)].Age, bins=17, kde=False)

sns.distplot(train_df[(train_df["Survived"]==1) & (train_df["Age"].isna() == False)].Age, bins=17, kde=False)
def transform_data(df_in):

    df_out = df_in.copy()

    df_out["age_imputed"] = df_out.Age

    df_out["age_imputed"].fillna(value=df_out.Age.mean(), inplace=True)

    return df_out
train_filled = transform_data(train_df)



print(sum(train_filled.Age.isna()))

print(sum(train_filled.age_imputed.isna()))
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()

logmodel.fit(np.array(train_filled.age_imputed).reshape(-1,1),train_filled.Survived)
test_df = pd.read_csv("../input/titanic/test.csv")

test_df.head()
test_filled = transform_data(test_df)

predictions = logmodel.predict(np.array(test_filled.age_imputed).reshape(-1,1))
predictions
submission = pd.DataFrame(test_filled.PassengerId)

submission["Survived"] = predictions

submission.to_csv("submission.csv", index=False)