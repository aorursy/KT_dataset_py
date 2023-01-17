import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df['GP-MIN'] = df['GP'] * df['MIN']
df['GP-PTS'] = kdf['GP'] * df['PTS']
#df['Value'] = df['GP-MIN'] / df['PTS']
#df.Name.nunique()
#df = df.drop_duplicates(subset='Name', keep=False)
df
y = df['TARGET_5Yrs'].as_matrix();
del df['Name']
del df['PlayerID']
del df['TARGET_5Yrs']
X = df.as_matrix().astype(np.float)
tmp = np.where(np.isnan(X))
for i in tmp[0]:
    X[i][8] = 0
df = pd.read_csv('test.csv')
result = df['PlayerID']
del df['Name']
del df['PlayerID']
df['GP-MIN'] = df['GP'] * df['MIN']
df['GP-PTS'] = df['GP'] * df['PTS']
#df['Value'] = df['GP-MIN'] / df['PTS']
df = df.as_matrix()
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
#df = min_max_scaler.transform(df)
#from sklearn.decomposition import PCA
#clfx = PCA(0.96) #keep 95% of variance
#X_trans = clfx.fit_transform(X)
#test_trans = clfx.transform(df)
X_trans = X
test_trans = df
print(X.shape)
print(X_trans.shape)
print(df.shape)
print(test_trans.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=.1, random_state=333, stratify=y)
import matplotlib.pyplot as plt
for i in range(20):
    
    x_index = i;
    colors = ['blue', 'red']
    print(i)
    for lab, col in zip(range(2), colors): #[(0, 'blue'), (1, 'red')]

        plt.hist(X[y==lab, x_index], color=col)
        plt.show()
    print("==========================================")
acc = []
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors=160)
clf1.fit(X_train, y_train)
y_pred = clf1.predict(X_test)
print(np.mean(y_pred == y_test))
acc.append(np.mean(y_pred == y_test))
from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression(max_iter=230, random_state=22)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print(np.mean(y_pred == y_test))
acc.append(np.mean(y_pred == y_test))
from sklearn.neural_network import MLPClassifier
clf3 = MLPClassifier()
clf3.fit(X_train, y_train)
y_pred = clf3.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(n_estimators=120)
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
print(np.mean(y_pred == y_test))
acc.append(np.mean(y_pred == y_test))
from sklearn.ensemble import AdaBoostClassifier
clf5 = AdaBoostClassifier(n_estimators=120)
clf5.fit(X_train, y_train)
y_pred = clf5.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.ensemble import ExtraTreesClassifier
clf6 = ExtraTreesClassifier(n_estimators=120)
clf6.fit(X_train, y_train)
y_pred = clf6.predict(X_test)
print(np.mean(y_pred == y_test))
acc.append(np.mean(y_pred == y_test))
from sklearn.ensemble import BaggingClassifier
clf7 = BaggingClassifier()
clf7.fit(X_train, y_train)
y_pred = clf7.predict(X_test)
print(np.mean(y_pred == y_test))
acc.append(np.mean(y_pred == y_test))
from sklearn import svm
clf8 = svm.SVC()
clf8.fit(X_train, y_train)
y_pred = clf8.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.naive_bayes import BernoulliNB
clf9 = BernoulliNB()
clf9.fit(X_train, y_train)
y_pred = clf9.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.tree import DecisionTreeClassifier
clf10 = DecisionTreeClassifier(criterion='gini',max_depth=300)
clf10.fit(X_train, y_train)
y_pred = clf10.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.tree import DecisionTreeRegressor
clf11 = DecisionTreeRegressor()
clf11.fit(X_train, y_train)
y_pred = clf11.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.tree import ExtraTreeClassifier
clf12 = ExtraTreeClassifier(criterion='entropy',max_depth=300)
clf12.fit(X_train, y_train)
y_pred = clf12.predict(X_test)
print(np.mean(y_pred == y_test))
acc.append(np.mean(y_pred == y_test))
from sklearn.tree import ExtraTreeRegressor
clf13 = ExtraTreeRegressor()
clf13.fit(X_train, y_train)
y_pred = clf13.predict(X_test)
print(np.mean(y_pred == y_test))
#acc.append(np.mean(y_pred == y_test))
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('KNeighborsClassifier', clf1), ('LogisticRegression', clf2), ('MLPClassifier', clf3), 
                                     ('RandomForestClassifier', clf4), ('AdaBoostClassifier', clf5), ('ExtraTreesClassifier', clf6),
                                     ('BaggingClassifier', clf7), ('svm', clf8), ('BernoulliNB', clf9), ('DecisionTreeClassifier', clf10), 
                                     ('ExtraTreeClassifier', clf12)], voting='hard', weights=acc)
eclf1 = eclf1.fit(X_train, y_train)
y_p = eclf1.predict(X_test)
print(np.mean(y_p == y_test))
result = result.as_matrix()
dic = {"PlayerID" : result}
result = pd.DataFrame(dic)
#y_pred = clf7.predict(test_trans)
y_pred = eclf1.predict(test_trans)
y_pred
result['TARGET_5Yrs'] = y_pred
result = result.set_index('PlayerID')
result.to_csv("RES.csv")
c = pd.read_csv("RES.csv")
c
pd.read_csv('sample_submission.csv')

