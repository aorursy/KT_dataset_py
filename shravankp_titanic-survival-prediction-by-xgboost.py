import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv('../input/train.csv',index_col='PassengerId')

test_df = pd.read_csv("../input/test.csv",index_col='PassengerId')
train_df = train_df.drop(['Name','Ticket','Cabin'],axis=1)

test_df = test_df.drop(['Name','Ticket','Cabin'],axis=1)

train_df.select_dtypes(include=object)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cols = ['Sex',"Embarked"]

for i in cols:

    le.fit(list(train_df[i].values.astype('str'))+list(test_df[i].values.astype('str')))

    train_df[i] = le.transform(list(train_df[i].values.astype('str')))

    test_df[i] = le.transform(list(test_df[i].values.astype('str')))

# help(le)
train_df.head()
y = train_df['Survived']

X = train_df.drop('Survived',axis=1)
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

pipe = make_pipeline(XGBClassifier())

scores = cross_val_score(pipe,X,y,scoring='neg_mean_absolute_error')

print(scores)

pipe.fit(X,y)

predictions = pipe.predict(test_df)
sub = pd.DataFrame({"PassengerId":test_df.index,"Survived":predictions})


sub.to_csv("./submission.csv",index=False)

# from sklearn.ensemble.partial_dependence import plot_partial_dependence
print(sub)