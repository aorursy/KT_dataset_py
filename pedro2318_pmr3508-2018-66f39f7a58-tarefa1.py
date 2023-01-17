import pandas as pd
import sklearn
adult = pd.read_csv("../input/ep12018/treino.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adult.shape
adult.head()
nadult = adult.dropna()
testAdult = pd.read_csv("../input/ep12018/teste.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
nTestAdult = testAdult.dropna()
Xadult = nadult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadult = nadult[["income"]]
XtestAdult = nTestAdult[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
YtesteAdult = nTestAdult[["income"]]
