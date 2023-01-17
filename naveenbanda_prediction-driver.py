import pandas as pd

import numpy as np
dataset=pd.read_csv("../input/train.csv")
dataset.convert_objects(convert_numeric=True)
print (dataset.head())
cols=list(dataset)

lenAttr=len(cols)

print (lenAttr)

print (cols)
cols.insert(lenAttr-1,cols.pop(cols.index('target')))
print (cols)
cat=[]

binary=[]

con_ord=[]

for i in range(1,lenAttr-1):

    if str(cols[i][-3:len(cols[i])])=="bin":

        binary.append(cols[i])

    elif str(cols[i][-3:len(cols[i])])=="cat":

        cat.append(cols[i])

    else:

        con_ord.append(cols[i])

print (cat)

print (binary)

print (con_ord)
cols=['id']+cat+binary+con_ord+['target']

print (cols)
dataset=dataset.loc[:,cols]

print (dataset.head())
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 58].values
print (np.count_nonzero(dataset.iloc[:, 1:14].values==-1))

print (np.sum(dataset.iloc[:, 15:31].values==-1))

print (np.sum(dataset.iloc[:, 32:57].values==-1))

n_cat=len(cat)

n_bin=len(binary)

n_con_ord=len(con_ord)

print (n_cat,n_bin,n_con_ord)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 1:n_cat])

X[:, 1:n_cat] = imputer.transform(X[:, 1:n_cat])
print (np.count_nonzero(X[:, 1:14]==-1))
imputer = Imputer(missing_values = -1, strategy = 'median', axis = 0)

imputer = imputer.fit(X[:, 32:57])

X[:, 32:57] = imputer.transform(X[:, 32:57])
print (np.count_nonzero(X[:, 32:57]==-1))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoder_X = LabelEncoder()

#X[:,1:14] = labelencoder_X.fit_transform(X[:, 1:14])

onehotencoder = OneHotEncoder(categorical_features = [i for i in range(1,15)])

X = onehotencoder.fit_transform(X).toarray()
temp=X
X=temp
print (X.shape)
X=X[:,1:219]

print (X.shape)
print (X[0:10,])
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train=X

X_train[:,192:] = sc_X.fit_transform(X_train[:,192:])



from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier(n_estimators=150,max_depth=20,random_state=0)

clf.fit(X_train,y)
from sklearn.externals import joblib

joblib.dump(clf,"model.pkl")
dataset=pd.read_csv("../input/test.csv")
dataset.convert_objects(convert_numeric=True)
print (dataset.shape)
cols=list(dataset)

lenAttr=len(cols)

print (lenAttr)

print (cols)
cat=[]

binary=[]

con_ord=[]

for i in range(1,lenAttr):

    if str(cols[i][-3:len(cols[i])])=="bin":

        binary.append(cols[i])

    elif str(cols[i][-3:len(cols[i])])=="cat":

        cat.append(cols[i])

    else:

        con_ord.append(cols[i])

print (cat)

print (binary)

print (con_ord)
cols=['id']+cat+binary+con_ord

print (cols)
dataset=dataset.loc[:,cols]

print (dataset.head())
X = dataset.iloc[:, :].values
print (X.shape)
dataID=list(dataset['id'])

print (dataID)
print (np.count_nonzero(dataset.iloc[:, 1:14].values==-1))

print (np.sum(dataset.iloc[:, 15:31].values==-1))

print (np.sum(dataset.iloc[:, 32:57].values==-1))

n_cat=len(cat)

n_bin=len(binary)

n_con_ord=len(con_ord)

print (n_cat,n_bin,n_con_ord)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 1:n_cat])

X[:, 1:n_cat] = imputer.transform(X[:, 1:n_cat])
print (np.count_nonzero(X[:, 1:14]==-1))
imputer = Imputer(missing_values = -1, strategy = 'median', axis = 0)

imputer = imputer.fit(X[:, 32:57])

X[:, 32:57] = imputer.transform(X[:, 32:57])
print (np.count_nonzero(X[:, 32:57]==-1))
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoder_X = LabelEncoder()

#X[:,1:14] = labelencoder_X.fit_transform(X[:, 1:14])

onehotencoder = OneHotEncoder(categorical_features = [i for i in range(1,15)])

X = onehotencoder.fit_transform(X).toarray()
temp=X
X=temp
print (X.shape)
print (X)
X=X[:,1:219]

print (X.shape)
print (X[0:10,])
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train=X

X_train[:,192:] = sc_X.fit_transform(X_train[:,192:])



y_prob=(clf.predict_proba(X_train[:,:]))
print (len(y_prob))

print (y_prob[0:50])
print (y_prob[0][1])

y_pred=clf.predict(X_train)

print (X_train.shape)
import csv

with open("output.csv","w+") as f:

    writer=csv.writer(f,delimiter=",")

    writer.writerow(['id','target'])

    for i in range(0,len(y_pred)):

        writer.writerow([int(dataID[i]),float(y_prob[i][1])])

f.close()

                         