import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv').fillna(" ")

train = train.dropna(how="any").reset_index(drop=True)
train.head()
train.groupby("is_duplicate")['id'].count().plot.bar()
from sklearn.feature_extraction.text import CountVectorizer
Bag = CountVectorizer(max_df=0.999, min_df=50, max_features=300, 

                                      analyzer='char', ngram_range=(1,2), 

                                      binary=True, lowercase=True)
Bag.fit(pd.concat((train.question1,train.question2)).unique())
question1 = Bag.transform(train['question1'])

question2 = Bag.transform(train['question2'])
X = -(question1 != question2).astype(int)

y = train['is_duplicate'].values
logisticRegressor = linear_model.LogisticRegression(C=0.1, solver='sag', 

                                                    class_weight={1: 0.472008228977, 0: 1.30905513329})

logisticRegressor.fit(X, y)
gamma_0 = 1.30905513329

gamma_1 = 0.472008228977

def link_function(x):

    return gamma_1*x/(gamma_1*x + gamma_0*(1 - x))



support = np.linspace(0, 1, 1000)

values = link_function(support)



fig, ax = plt.subplots()

ax.plot(support, values)

ax.set_title('Link transformation', fontsize=20)

ax.set_xlabel('x')

ax.set_ylabel('f(x)')

plt.show()
test = pd.read_csv('../input/test.csv')

test.ix[test['question1'].isnull(),['question1','question2']] = ' '

test.ix[test['question2'].isnull(),['question1','question2']] = ' '
%%time

test.ix[test['question1'].isnull(),['question1','question2']] = ' '

test.ix[test['question2'].isnull(),['question1','question2']] = ' '



Question1 = Bag.transform(test['question1'])

Question2 = Bag.transform(test['question2'])



X_test = -(Question1 != Question2).astype(int)



seperators= [750000,1500000]

testPredictions1 = logisticRegressor.predict_proba(X_test[:seperators[0],:])[:,1]

testPredictions2 = logisticRegressor.predict_proba(X_test[seperators[0]:seperators[1],:])[:,1]

testPredictions3 = logisticRegressor.predict_proba(X_test[seperators[1]:,:])[:,1]

testPredictions = np.hstack((testPredictions1,testPredictions2,testPredictions3))
submissionName = 'quora_submission'



submission = pd.DataFrame()

submission['test_id'] = test['test_id']

submission['is_duplicate'] = testPredictions

submission.to_csv(submissionName + '.csv', index=False)