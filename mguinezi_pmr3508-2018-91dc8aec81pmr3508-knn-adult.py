import pandas as pd
import sklearn
import matplotlib
adults = pd.read_csv("../input/adultsdb/train_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
testAdults = pd.read_csv("../input/adultsdb/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
adults.shape
adults.head()
adults["native.country"].value_counts()
adults["age"].value_counts().plot(kind="bar")
nadults = adults.dropna()
nadults.head()
Xadults = adults[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
Yadults = adults.income
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xadults,Yadults,cv=10)
scores
knn.fit(Xadults,Yadults)
ntestAdults = testAdults.dropna()
XtestAdults = testAdults[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
YtestPrediction = knn.predict(XtestAdults)
YtestPrediction
arr1= testAdults.iloc[:,0].values
arr1 = arr1.ravel()
dataset = pd.DataFrame({'Id':arr1[:],'income':YtestPrediction[:]})
dataset.to_csv("submition.csv", index = False)
len(YtestPrediction)
