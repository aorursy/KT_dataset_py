# Package imports
import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X = train[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]
Y, Y_label= pd.factorize(train['type'])
X_pred  = test.drop(["id", "color"], axis=1)
Y = pd.DataFrame(Y)
Y.columns = ['type']
#제일 큰 컬럼 인덱스 찾기
X['max_feature'] = X.iloc[:,0:4].idxmax(axis=1)
X['min_feature'] = X.iloc[:,0:4].idxmin(axis=1)
#X['max-min'] = X.iloc[:,0:3].max(axis=1)-X.iloc[:,0:3].min(axis=1)
#X.sort_values("max-min", ascending=False)
#X['max_feature'] == 'rotting_flesh' and X['min_feature'] == 'has_soul'
X[(X.max_feature == 'rotting_flesh')&(X.min_feature == 'has_soul')]
#df1 = df[(df.a != -1) & (df.b != -1)]
X['isGoast'] = np.where((X.max_feature == 'rotting_flesh')&(X.min_feature == 'has_soul'), 1, 0)
#df['color'] = np.where(df['Set']=='Z', 'green', 'red')
X.drop(columns=["max_feature", "min_feature"], inplace=True)
X, X_test = train_test_split(X, test_size=0.2, random_state=42)
Y, Y_test = train_test_split(Y, test_size=0.2, random_state=42)
train_data = lightgbm.Dataset(X, label=Y)
test_data  = lightgbm.Dataset(X_test, label=Y_test)
parameters = {
    'application': 'multiclass',
    'objective': 'multiclass',
    'num_class':3,
    'metric': 'multi_logloss',
    'boosting': 'dart',
    #'boosting': 'rf',
    'num_leaves': 50,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.001,
    'max_depth' : 6
}
model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=500)
prediction = model.predict(X_pred)
prediction = Y_label[np.argmax(prediction, axis=1)]
submission = pd.DataFrame(prediction)
submission.columns = ["type"]
submission = pd.concat([test['id'], submission["type"]], axis=1)
submission.to_csv("submission_lightGBM.csv", index=False)
submission
