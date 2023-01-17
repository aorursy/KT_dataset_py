import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')

df_test.head()

df_train.head()
df_train.shape
df_train.isnull().sum()
df_train.describe()
sb.heatmap(df_train.corr())
g = sb.FacetGrid(df_train,col='target')

g.map(plt.hist,'target')
# Heatmap shows no correlation between some of the columns.



drop_cols_train = df_train.columns[df_train.columns.str.startswith('ps_calc')]

drop_cols_test = df_test.columns[df_test.columns.str.startswith('ps_calc')]

Xtrain = df_train.drop(drop_cols_train,axis=1)

train_X = Xtrain.drop(['id','target'],axis=1)

Xtest = df_test.drop(drop_cols_test,axis=1)

test_X = Xtest.drop(['id'],axis=1)

y = df_train['target']
import xgboost as xgb

model = xgb.XGBClassifier()

xgb_param = model.get_xgb_params()

xgb_param
gbm = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05, objective='binary:logistic').fit(train_X, y)

predictions = gbm.predict_proba(test_X,ntree_limit=10)
res = pd.DataFrame(predictions)

prob = res[1]

prob
result = pd.DataFrame({'id':df_test['id'],'target':prob})
result.to_csv('probabilities.csv',index=False)
result[:5]