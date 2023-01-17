import numpy as np

import pandas as pd

from sklearn.naive_bayes import GaussianNB
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
def preprocess_titanic(df):

    newdf = pd.DataFrame()

    

    newdf['Sex'] = pd.Series([1 if s == 'male' else 0 for s in df.Sex], name = 'Sex' )

    newdf['Age'] = df.Age.fillna( df.Age.mean() )

    newdf['Fare'] = df.Fare.fillna( df.Fare.mean() )



    pclass = pd.get_dummies( df.Pclass , prefix='Pclass' )

    for c in pclass:

        newdf[c] = pclass[c]

    return newdf
train_data = preprocess_titanic(train)

train_labels = train.Survived
model = GaussianNB()

model.fit(train_data,train_labels)

print ('Score: {}'.format(model.score(train_data, train_labels)))
test_data = preprocess_titanic(test)

prediction = model.predict(test_data)



submission = pd.DataFrame()

submission['PassengerId'] = test.PassengerId

submission['Survived'] = pd.Series(prediction)

submission.to_csv("kaggle.csv", index=False)

submission.head()