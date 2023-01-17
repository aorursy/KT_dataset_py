import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
def prepare(df, labels=True):
    inp = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin']].copy()
    inp['Cabin'] = inp.Cabin.isnull()
    inp['Sex'] = inp.Sex == 'male'
    if labels:
        return inp, df['Survived']
    return inp
    
# initialize data
train_data = pd.read_csv('../input/train.csv')
train_inp, train_label = prepare(train_data.iloc[:800], labels=True)
val_inp, val_label = prepare(train_data.iloc[800:], labels=True)

test_data = pd.read_csv('../input/test.csv')
# specify the training parameters 
model = CatBoostClassifier(iterations=2, depth=2, learning_rate=1, loss_function='Logloss', logging_level='Verbose')
#train the model
model.fit(train_inp.values, train_label.values, cat_features=[0, 1, 3, 4, 6])
# make the prediction using the resulting model
preds_class = model.predict(val_inp.values)
preds_proba = model.predict_proba(val_inp.values)

(val_label == preds_class).mean()
result = pd.read_csv('../input/gender_submission.csv')
result['Survived'] = model.predict(prepare(test_data, labels=False).values).astype('int')
result.to_csv('result.csv', index=False)

