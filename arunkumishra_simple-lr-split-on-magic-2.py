import numpy as np, pandas as pd, os

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from statistics import mean
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
summary = train.describe().T

summary[summary['max']>20]
# Helper method to find feature columns relevant to each value of magic

def find_feature_columns(df):

    corr_data = df.corr()

    cor_target = abs(corr_data['target'])

    # reached 0.075 by trial and error

    relevant_features = cor_target[cor_target>=0.075]

    relevant_features.sort_values(ascending=False)

    columns = list(relevant_features.index)

    columns.append('wheezy-copper-turtle-magic')

    columns.append('id')



    test_columns = columns.copy()

    test_columns.remove('target')

    return columns, test_columns
splitvalues = train['wheezy-copper-turtle-magic'].unique()

len(splitvalues)
# LR model

logModel = LogisticRegression(solver='liblinear',penalty='l2')
# validating approach

scores=[]

for i in splitvalues:

    train_split = train[train['wheezy-copper-turtle-magic']==i]

    columns, test_columns = find_feature_columns(train_split)

    train_split = train_split[columns]

    y = train_split['target']

    X = train_split.drop('id',axis=1)

    X = X.drop('wheezy-copper-turtle-magic',axis=1)

    X = X.drop('target',axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

    logModel.fit(X_train,y_train)

    predictions = logModel.predict(X_valid)

    score = f1_score(y_valid,predictions)

    scores.append(score)
print('Average F1 score: ' + str(mean(scores)))
# Setup output dataframe

output = pd.DataFrame(columns=['id','target'])
#model

for i in splitvalues:

    output_split = pd.DataFrame(columns=['id','target'])

    

    train_split = train[train['wheezy-copper-turtle-magic']==i]

    columns, test_columns = find_feature_columns(train_split)

    train_split = train_split[columns]

    

    test_split = test[test_columns]

    test_split = test_split[test_split['wheezy-copper-turtle-magic']==i]

    test_split2 = test_split.drop('id',axis=1)

    test_split2 = test_split2.drop('wheezy-copper-turtle-magic',axis=1)

    

    y = train_split['target']

    X = train_split.drop('id',axis=1)

    X = X.drop('wheezy-copper-turtle-magic',axis=1)

    X = X.drop('target',axis=1)

    

    logModel.fit(X,y)

    predictions = logModel.predict(test_split2)

 

    output_split['id'] = test_split['id']

    output_split['target'] = predictions



    output = pd.concat([output,output_split])
# write to submission file

output.to_csv('submission.csv',index=False)