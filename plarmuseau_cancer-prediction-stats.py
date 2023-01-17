import pandas as pd

import numpy as np
df = pd.read_csv("../input/data.csv",header = 0)

df = df.drop("id",1)

df = df.drop("radius_se",1)

df = df.drop("texture_se",1)

df = df.drop("perimeter_se",1)

df = df.drop("area_se",1)

df = df.drop("smoothness_se",1)

df = df.drop("compactness_se",1)

df = df.drop("concavity_se",1)

df = df.drop("concave points_se",1)

df = df.drop("symmetry_se",1)

df = df.drop("fractal_dimension_se",1)

df = df.drop("Unnamed: 32",1)

new_columns = df.columns.values; new_columns[18] = 'concavepoints_worst'; df.columns = new_columns



#Volume worst estimation of cancer create a estimate worst 3D volume estimate

temp = np.log(df.radius_worst*df.area_worst)

df['Volume_ln'] = temp.values

#

temp = np.log(df.concavepoints_worst*df.concavity_worst*df.compactness_worst+1)

df['Concave_ln'] = temp.values

#cancer fractal- symmetry  divided by volume

temp = -np.log(df.fractal_dimension_worst*df.symmetry_worst/np.log(df.radius_mean*df.area_mean))

df['FractVol_ln'] = temp.values

# all unrelated

temp = np.log(df.radius_worst*df.perimeter_worst*df.concavepoints_worst+1)

df['RaPeCo_ln'] = temp.values

#show malignancy binary

d = {'M' : 2, 'B' : -1}

df['binair'] = df['diagnosis'].map(d)



df.describe()
import seaborn as sns

sns.pairplot(df, hue="diagnosis", size=2)
cols = list(df.columns[1:31])

corr_matrix = df[cols].corr()

heatmap = sns.heatmap(corr_matrix,cbar=True,annot=True,square=True,fmt='.1f',annot_kws={'size': 7},yticklabels=cols,xticklabels=cols,cmap='Dark2')


test=df[::-567]

test=np.array(test)[1,:]

features = list(df.columns[1:25])

print(features)

from sklearn import tree

testset = test[1:25]

x = df[features]

y = df["binair"]



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)

output = Tree.predict(x)



Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)

output = Forest.predict(x)



from sklearn import svm

svc = svm.SVC(kernel='linear',C=1).fit(x,y)

output = svc.predict(x)



      

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .4,random_state=0)



Kfold = KFold(len(df),n_folds = 10,shuffle = False)

print("KfoldCrossVal score using Decision Tree is %s" %cross_val_score(Tree,x,y,cv=10).mean())



Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using Random Forest is %s" %cross_val_score(Forest,x,y,cv=10).mean())



Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using SVM is %s" %cross_val_score(svc,x,y,cv=10).mean())

from sklearn import metrics



dt = Tree.fit(X_train,y_train)

y_pred = dt.predict(X_test)

metrics.accuracy_score(y_test,y_pred)





rf = Forest.fit(X_train,y_train)

y_pred = rf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)



sm = svc.fit(X_train,y_train)

y_pred = sm.predict(X_test)

metrics.accuracy_score(y_test,y_pred)
from sklearn.ensemble import RandomForestClassifier
Forest = RandomForestClassifier(n_estimators = 10)

Forest = Forest.fit(x,y)

output = Forest.predict(x)

print(output)
from sklearn import svm
svc = svm.SVC(kernel='linear',C=1).fit(x,y)
output = svc.predict(x)

print(output)
from sklearn import cross_validation

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size= .4,random_state=0)
Kfold = KFold(len(df),n_folds = 10,shuffle = False)

print("KfoldCrossVal score using Decision Tree is %s" %cross_val_score(Tree,x,y,cv=10).mean())
Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using Random Forest is %s" %cross_val_score(Forest,x,y,cv=10).mean())
Kfold = KFold(len(df),n_folds=10,shuffle=False)

print("KfoldCrossVal score using SVM is %s" %cross_val_score(svc,x,y,cv=10).mean())
from sklearn import metrics
dt = Tree.fit(X_train,y_train)

y_pred = dt.predict(X_test)

metrics.accuracy_score(y_test,y_pred)
rf = Forest.fit(X_train,y_train)

y_pred = rf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)
sm = svc.fit(X_train,y_train)

y_pred = sm.predict(X_test)

metrics.accuracy_score(y_test,y_pred)