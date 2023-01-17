import pandas as pd
df = pd.read_csv('train.csv')
del df['PlayerID']
del df['Name']
df['GPMIN'] = df['GP'] * df['MIN']
df['3PAP3P'] = df['3PA'] * df['3P%']
df['FTAFT%'] =df['FTA'] * df['FT%']

y=df['TARGET_5Yrs']
del df['TARGET_5Yrs']

y = y.as_matrix()
X = df.as_matrix()
import numpy as np
t = np.where(np.isnan(X))
for i in range(len(t[0])):
    X[t[0][i]][t[1][i]] = 0
np.where(np.isnan(X))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)
print(X_test.shape)
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
clf = KNeighborsClassifier(n_neighbors=150)
clf.fit(X_train, y_train)
y_pre = clf.predict(X_test)
print(np.mean(y_pre==y_test))

df = pd.read_csv('test.csv')
del df['PlayerID']
del df['Name']
df['GPMIN'] = df['GP'] * df['MIN']
df['3PAP3P'] = df['3PA'] * df['3P%']
df['FTAFT%'] = df['FTA'] * df['FT%']

df
test=df.as_matrix()
test
from sklearn.neighbors import KNeighborsClassifier
clv=KNeighborsClassifier(n_neighbors=100)
clv.fit(X, y)
y_pre = clv.predict(test)
y_pre
sub = pd.read_csv('sample_submission.csv')
sub
del sub['TARGET_5Yrs']
sub['TARGET_5Yrs'] =y_pre
sub = sub.set_index('PlayerID')
sub
sub.to_csv('ANS1.csv')
pd.read_csv('ANS1.csv')

