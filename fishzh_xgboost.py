import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
### load datasets
file_path = '../input/'
train = pd.read_csv(file_path + 'train.csv')
test = pd.read_csv(file_path + 'test.csv')

Id_train=train['Id']
Id_test=test['Id']

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
features_train=train.drop('Cover_Type', axis=1)
labels_train=train['Cover_Type']
### data split
x_train,x_vali,y_train,y_vali = train_test_split(features_train,
                                                 labels_train,
                                                 test_size = 0.1,
                                                 random_state = 33)
### fit model for train data
model = XGBClassifier(learning_rate=0.1,
                      n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                       min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                      colsample_btree=0.8,       # 随机选择80%特征建立决策树
                      objective='multi:softmax', # 指定损失函数
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=27            # 随机数
                      )
model.fit(x_train,
          y_train,
          eval_set = [(x_vali,y_vali)],
          eval_metric = "mlogloss",
          early_stopping_rounds = 10,
          verbose = True)


y_pred = model.predict(test)
submit_df = pd.DataFrame()
submit_df['Cover_Type'] = y_pred
submit_df['Id'] = Id_test
submit_df=submit_df.loc[:, ['Id', 'Cover_Type']]
submit_df.to_csv('xgboost.csv', index=False)
print(submit_df.shape)
# y_valipred = model.predict(x_vali)
# accuracy = accuracy_score(y_vali,y_valipred)
# print("accuarcy: %.2f%%" % (accuracy*100.0))
# print(accuracy)
# ### plot feature importance
# fig,ax = plt.subplots(figsize=(15,15))
# plot_importance(model,
#                 height=0.5,
#                 ax=ax,
#                  max_num_features=64)
# plt.show()

# ### make prediction for test data
# y_pred = model.predict(x_test)

# ### model evaluate
# accuracy = accuracy_score(y_test,y_pred)
# print("accuarcy: %.2f%%" % (accuracy*100.0))
# """
#  66 95.74%
#  67 """