import pandas as pd
from IPython.display import display, HTML
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)






train = pd.read_csv('../input/minor-project-2020/train.csv')
test = pd.read_csv('../input/minor-project-2020/test.csv')

label = train['target']
train2 = train.drop(columns = ['target', 'id'])
display(train2.head(10))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


params = {"objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        'lambda_l1': 8.363957625616022e-05, 'lambda_l2': 5.1576139520241994e-05, 'num_leaves': 24, 'feature_fraction': 0.858469319864406, 'min_data': 46, 'max_depth': 94, 'num_boost_round': 86, 'learning_rate': 0.0007961842606486752, 'min_child_samples': 37, 'scale_pos_weight': 386.944723690289}
x_train, x_test, y_train, y_test = train_test_split(train2,label,test_size = 0.25, random_state = 0)
d_train = lgb.Dataset(x_train, label = y_train)
gbm = lgb.train(params, d_train)
preds = gbm.predict(test.drop(columns = ['id']))
y_pred = preds
df = pd.DataFrame(data = y_pred)
df['target'] = df[0]
df.drop(columns = [0], inplace = True)
display(df.head(10))
test['target'] = df.target
print(test.head(10))
print(len(test))
test[['id', 'target']].to_csv('./output.csv' ,index = False)