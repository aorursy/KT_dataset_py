import pandas as pd

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



import numpy as np



import os



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.linear_model import ElasticNet



from sklearn.metrics import roc_auc_score
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
target_column = 'target'

column_to_drop = 'id'



train_set, validate_set = train_test_split(train, test_size = 0.2, random_state = 1)



x_train = train_set.drop([target_column, column_to_drop], axis = 1).copy()

y_train = train_set[target_column].copy()



x_validate = validate_set.drop([target_column, column_to_drop], axis = 1).copy()

y_validate = validate_set[target_column].copy()



x_total = train.drop([target_column, column_to_drop], axis = 1).copy()

y_total = train[target_column].copy()



x_test = test.drop(column_to_drop, axis = 1).copy()
en = ElasticNet(alpha = 0.085, l1_ratio = 0.5)



en.fit(x_train, y_train)
y_validate_p = en.predict(x_validate)

roc_auc_score(y_validate, y_validate_p)
en = ElasticNet(alpha = 0.085, l1_ratio = 0.5)



en.fit(x_total, y_total)
submission = pd.DataFrame(test.id)

y_test_p = pd.Series(en.predict(x_test))

submission['target'] = y_test_p

submission.to_csv("submission.csv", index = False)