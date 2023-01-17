import numpy as np
import pandas as pd
dftrain = pd.read_csv("../input/train.csv")
dftest = pd.read_csv("../input/test.csv")
print(dftrain.shape,dftest.shape)
dftrain.head(5)
train_y = dftrain['Cover_Type']
train_x = dftrain.drop(columns=['Cover_Type','Id'])
test_x = dftest.drop(columns=['Id'])
print(train_y.shape,train_x.shape,test_x.shape)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
frames = [train_x,test_x]
all_df = pd.concat(frames).values
all_df = min_max_scaler.fit_transform(all_df)
all_df = pd.DataFrame(all_df)
print(all_df.shape)
all_df.head(5)
train_X = all_df[:15120]
test_X = all_df[15120:]
print(train_X.shape,test_X.shape)
"""
from sklearn.svm import SVC
model = SVC() # 0.6696
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() #0.7196
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50) #0.8526
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() #0.5526
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier() #0.7785
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5) #0.78303
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=20) #0.5839

from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators=[('knn',clf1),('rfc',clf2),('dtc',clf3),('lr',clf4)],voting='hard') #0.82
"""
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(train_X,train_y)
score = model.score(train_X,train_y)
print(score)
predictions = model.predict(test_X)
results = pd.DataFrame(predictions,index=range(15121,581013))
results.to_csv("sol.csv",index=True,index_label="Id",header=['Cover_Type'])
