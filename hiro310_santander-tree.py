import numpy as np

import pandas as pd
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.head()
test.head()
train.isnull().sum()
test.isnull().sum()
from sklearn import tree
# train_object = train["target"].values



train_object = train.iloc[:200000, 1].values
print(train_object)
# train_explan = train[["var_0","var_1","var_199"]].values



# train_explan = train.iloc[:200000, range(2,22)].values



# train_explan = train[["var_0","var_1","var_6","var_22","var_26","var_53","var_99","var_110","var_133","var_179","var_190"]].values



train_explan = train[[

    "var_0",

    "var_1",

    "var_6",

    "var_22",

    "var_26",

    "var_53",

    "var_99",

    "var_110",

    "var_133",

    "var_179",

    "var_190",

    "var_12",

    "var_13",

    "var_21",

    "var_34",

    "var_76",

    "var_80",

    "var_81",

    "var_115",

    "var_139",

    "var_146",

    "var_148",

    "var_165",

    "var_166",

    "var_174",

    "var_198"

]].values
print(train_explan)
sctp_tree = tree.DecisionTreeClassifier()
sctp_tree = sctp_tree.fit(train_explan, train_object)
print(sctp_tree)
# test_explan = test.iloc[:200000, range(2,22)].values



# test_explan = test[["var_0","var_1","var_6","var_22","var_26","var_53","var_99","var_110","var_133","var_179","var_190"]].values



# test_explan = test[["var_12","var_13","var_21","var_34","var_76","var_80","var_81","var_115","var_139","var_146","var_148","var_165","var_166","var_174","var_198"]].values



test_explan = test[[

    "var_0",

    "var_1",

    "var_6",

    "var_22",

    "var_26",

    "var_53",

    "var_99",

    "var_110",

    "var_133",

    "var_179",

    "var_190",

    "var_12",

    "var_13",

    "var_21",

    "var_34",

    "var_76",

    "var_80",

    "var_81",

    "var_115",

    "var_139",

    "var_146",

    "var_148",

    "var_165",

    "var_166",

    "var_174",

    "var_198"

]].values
print(test_explan)
prediction = sctp_tree.predict(test_explan)
print(prediction)
id_code = np.array(test["ID_code"]).astype(str)
print(id_code)
answer = pd.DataFrame(prediction, id_code, columns = ["target"])
print(answer)
answer.to_csv("answer_1903042026.csv", index_label = ["ID_code"])