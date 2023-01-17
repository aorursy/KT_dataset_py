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



train_df = pd.read_csv("../input/train.csv")

test_df   = pd.read_csv("../input/test.csv")



# in the data Embarked, Sex features are not numerical values. In order to let the classifier

# to interpret this properly, it is better to convert this to numerical value.

# In order to do this, we will convert these columns' type to category and then convert it to their code

# The mechanism behind it is to covert it to `category` type, and then use the codes that are

# assigned to the category.



train_df["Embarked"] = train_df["Embarked"].astype('category').cat.codes

train_df["Sex"] = train_df["Sex"].astype('category').cat.codes



# we drop the PassengerId, Cabin, Ticket, Name features.

# PassengerId seems irrelevant with the survival.

# As for cabin, ticket feature there were too many NaN values.

# I am not sure how to incorporate this incomplete feature to the prediction so just left them out.

# The name was not taken into consideration as well since it seems very irrelevant to survival

train_df2=train_df.drop(["PassengerId","Cabin","Ticket","Name"],axis=1)



# The age feature seemed like a useful feature but there were several missing

# rows that did not have this value.

# For simplicity, we will use the average value and fill the missing parts with this



agemean = train_df2["Age"].mean()

agefillvalue = int(round(agemean))

train_df2["Age"] = train_df2["Age"].fillna(agefillvalue)



# nicely slice the train_df into input and output that will be given to train the classifier



train_y = train_df2["Survived"]

train_x = train_df2.drop("Survived",axis=1)




















