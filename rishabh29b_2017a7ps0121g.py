import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from scipy import stats

# Calculate euclidean distance between x1 and x2. You can assume both x1 and x2 are numpy arrays

def distance(x1, x2):

    res = np.sqrt(np.sum(np.square(np.subtract(x1,x2))))

    return res



# Implement knn algorithm. Return majority label for given test_sample and k

def knn(X_train, y_train, test_sample, k):

    dist = []

    for i,pt in enumerate(X_train):

        dist.append([distance(test_sample,pt),y_train[i]])

    dist.sort()

    dist = dist[:k]

    n = [i[1] for i in dist]

    return int(stats.mode(n)[0])

 

# Return class of each test sample predicted by knn 

def predict(X_train, y_train, X_test, k):

    p = []

    for i in X_test:

        p.append(knn(X_train, y_train, i, k))

    return p
df = pd.read_csv('../input/eval-lab-1-f464-v2/train.csv')

dfTest = pd.read_csv('../input/eval-lab-1-f464-v2/test.csv')



dfNew = df.fillna(df.mean())

df = df.sample(frac=1).reset_index(drop=True)

dfTest = dfTest.fillna(df.mean())

# numerical_features = ["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"]

# numerical_features = ["feature1","feature2","feature4","feature8","feature9","feature10","feature11"]

# numerical_features = ["feature1","feature3","feature6","feature9","feature11"]

numerical_features = ["feature1","feature2","feature3","feature5","feature6","feature7","feature8","feature9","feature11"]

categorical_features = ["type"]
# Train

numDf = dfNew[numerical_features]

catDf = pd.get_dummies(dfNew[categorical_features])

finalDat = np.concatenate([numDf,catDf.values],axis=1)

# finalDat = np.concatenate([numDf],axis=1)

x_train = finalDat[:4000]

y_train = dfNew.iloc[:4000,13]

x_val = finalDat[4000:]

y_val = dfNew.iloc[4000:,13]



# Test

numTestDf = dfTest[numerical_features]

catTestDf = pd.get_dummies(dfTest[categorical_features])

x_test = np.concatenate([numTestDf,catTestDf.values],axis=1)

# x_test = np.concatenate([numTestDf],axis=1)
from sklearn import neighbors as skn

clf = skn.NearestCentroid() 

clf.fit(x_train, y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=200)

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier

from sklearn.pipeline import Pipeline

nca = NeighborhoodComponentsAnalysis(random_state=42)

knn = KNeighborsClassifier(n_neighbors=100)

nca_pipe = Pipeline([('nca', nca), ('knn', knn)])

nca_pipe.fit(x_train, y_train.values) 

print(nca_pipe.score(x_val, y_val)) 
# x_train = finalDat

# y_train = dfNew.iloc[:,13]

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=1000,criterion='entropy',max_depth=25, min_samples_split=7,min_samples_leaf=1

                            ,oob_score=True,random_state=42)#class_weight='balanced')

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
# x_train = finalDat

# y_train = dfNew.iloc[:,13]

from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(n_estimators=1000,max_depth=10, min_samples_split=7,min_samples_leaf=2

                            ,oob_score=True,random_state=42,warm_start=True)

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

y_pred = np.round(y_pred)

print(accuracy_score(y_val,y_pred))
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(n_estimators=7)

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=7)

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.gaussian_process import GaussianProcessClassifier

clf = GaussianProcessClassifier()

clf = clf.fit(x_train,y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn import svm

clf = svm.SVC(gamma='scale',decision_function_shape = "ovr",kernel='poly',)#class_weight='balanced')

clf.fit(x_train, y_train.values)

# y_pred = clf.decision_function(x_val)

# y_pred = np.argmax(y_pred,axis=1)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn import svm

clf = svm.LinearSVC(multi_class='ovr')#class_weight='balanced')

clf.fit(x_train, y_train.values)

y_pred = clf.decision_function(x_val)

y_pred = np.argmax(y_pred,axis=1)

# y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
# Extremely slow

from sklearn.neighbors import NeighborhoodComponentsAnalysis,KNeighborsClassifier

from sklearn.pipeline import Pipeline

nca = NeighborhoodComponentsAnalysis(random_state=42)

knn = KNeighborsClassifier(n_neighbors=100)

nca_pipe = Pipeline([('nca', nca), ('knn', knn)])

nca_pipe.fit(x_train, y_train.values) 

print(nca_pipe.score(x_val, y_val)) 
# y_pred = predict(x_train, y_train.values, x_val, 100)

# print(accuracy_score(y_val,y_pred))y_pred = clf.predict(x_val)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.feature_selection import RFE



clf = svm.SVC(gamma='scale',decision_function_shape = "ovr",kernel='linear',)#class_weight='balanced')

rfe = RFE(estimator=clf, n_features_to_select=1, step=1)

rfe.fit(x_train, y_train.values)

plt.matshow(ranking, cmap=plt.cm.Blues)

plt.colorbar()

plt.title("Ranking of pixels with RFE")

plt.show()
from sklearn.experimental import enable_hist_gradient_boosting  # noqa

from sklearn.ensemble import HistGradientBoostingClassifier



clf = HistGradientBoostingClassifier(l2_regularization=0.3,loss='categorical_crossentropy',

                                     max_bins=250,tol=0.0001,max_depth=25,scoring='loss',

                                     random_state=42,max_iter=15)

clf.fit(x_train, y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
# x_train = finalDat

# y_train = dfNew.iloc[:,13]

from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(n_estimators=5000,max_depth=25,random_state=42)

clf.fit(x_train, y_train.values)

y_pred = clf.predict(x_val)

y_pred = np.round(y_pred)

print(accuracy_score(y_val,y_pred))
x_train = finalDat

y_train = dfNew.iloc[:,13]

from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(solver='adam',epsilon=1e-6, activation='relu', alpha=1e-3,hidden_layer_sizes=(7,15), random_state=1,max_iter=500)

clf.fit(x_train, y_train.values)

y_pred = clf.predict(x_val)

y_pred = np.round(y_pred)

print(accuracy_score(y_val,y_pred))
from sklearn.naive_bayes import ComplementNB

clf = ComplementNB()

clf.fit(x_train, y_train.values)

y_pred = clf.predict(x_val)

print(accuracy_score(y_val,y_pred))
from sklearn.ensemble import VotingRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor



clf1 = MLPRegressor(solver='adam',epsilon=1e-6, activation='relu', alpha=1e-3,hidden_layer_sizes=(7,15), random_state=1,max_iter=500)

clf2 = HistGradientBoostingRegressor(l2_regularization=0.3)

clf3 = GradientBoostingRegressor(n_estimators=500,max_depth=25,random_state=42)

clf4 = RandomForestRegressor(n_estimators=500,max_depth=25, min_samples_split=7,min_samples_leaf=2,oob_score=True,random_state=42)

eclf = VotingRegressor(estimators=[('dt', clf1), ('hgb', clf2), ('gb', clf3), ('rf', clf4)], weights=[2, 1, 2, 1])

eclf.fit(x_train, y_train.values)

y_pred = eclf.predict(x_val)

y_pred = np.round(y_pred)

print(accuracy_score(y_val,y_pred))
# id,f6,f3,f1,f11,f9,f7,f2,f5,f8,f4,f10

# f6,f3,f1,f9 

from sklearn.feature_selection import RFE

rfe = RFE(estimator=clf, n_features_to_select=1, step=1)

rfe.fit(x_train, y_train.values)

rfe.ranking_
y_test = clf.predict(x_test)

y_test = np.round(y_test).astype(int)

out = np.concatenate([[dfTest['id'].values], [y_test]],axis=0)

out = out.T

out = pd.DataFrame(out)

out.to_csv('hist2.csv',index=False,header=['id','rating'])
!ls