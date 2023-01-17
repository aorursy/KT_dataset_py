import pandas as pd
from pandas import DataFrame
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
df = pd.read_csv('../input/dmassign1/data.csv')
orig_class = df['Class']
df  = df.drop('Class', axis=1)
import numpy as np
df = df.replace('?', np.nan)
pd.options.display.max_rows = 100

pd.options.display.max_columns = 200
for col in df.columns:

    if df[col].dtype!=int:

        try:

            df[col] = df[col].astype(float)

        except:

            df[col] = df[col].astype('category')

            df[col] = df[col].fillna(df[col].value_counts().idxmax())

for col in df.columns:

    if df[col].isnull().any():

         df[col] = df[col].fillna(df[col].mean())
df['Col189'] = np.where(df['Col189'] == 'yes', 1, 0)
for col in ['Col190', 'Col191', 'Col192', 'Col193', 'Col194', 'Col195', 'Col196', 'Col197']:

    one_hot = pd.get_dummies(df[col],prefix=col,prefix_sep='_')

    df = df.drop(col, axis = 1)

    df = df.join(one_hot)
import pandas as pd

from sklearn import preprocessing

ids = df['ID']

x = df.drop('ID', axis=1).values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled).join(ids)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.feature_selection import RFECV

import sklearn
from sklearn.decomposition import PCA



def get_PCA_model(X):

    model=PCA(n_components=5)

    model.fit(X)

    return model
X, y = df.drop('ID', axis=1), orig_class
pca_model =  get_PCA_model(X)
X = X.head(1300)

y = y.head(1300)
X_new = pca_model.transform(X)
X_new = X.join(pd.DataFrame(X_new, columns = ['pca{}'.format(i) for i in range(X_new.shape[1])]))
dtree = DecisionTreeClassifier(max_depth=10, min_samples_leaf = 50)



rfecv = RFECV(estimator=dtree, step=1, cv=KFold(2), min_features_to_select=20, scoring='accuracy')

rfecv.fit(X_new, y)

selected = rfecv.support_
X_new = X_new[X_new.columns[selected]]
X_train, X_test, Y_train, Y_test = train_test_split(X_new, y, test_size=0.2, random_state=0)
import sklearn



clf = sklearn.neighbors.KNeighborsClassifier(45, weights='uniform')



clf.fit(X_train, Y_train)



(clf.predict(X_train) ==  Y_train).sum() * 1.0 / len(X_train)

(clf.predict(X_test) ==  Y_test).sum() * 1.0 / len(X_test)
# import sklearn



# clf = sklearn.neighbors.KNeighborsClassifier(45, weights='uniform')



# clf.fit(X_train, Y_train)



# (clf.predict(X_train) ==  Y_train).sum() * 1.0 / len(X_train)

# (clf.predict(X_test) ==  Y_test).sum() * 1.0 / len(X_test)





# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

#                      metric_params=None, n_jobs=None, n_neighbors=45, p=2,

#                      weights='uniform')

# 0.4125

# 0.40384615384615385
X = df.drop('ID', axis=1)
X_new = pca_model.transform(X)

X_new = X.join(pd.DataFrame(X_new, columns = ['pca{}'.format(i) for i in range(X_new.shape[1])]))
X_new = X_new[X_new.columns[selected]]
import sklearn



clf = sklearn.neighbors.KNeighborsClassifier(45, weights='uniform')



clf.fit(X_new.head(1300), orig_class.head(1300))



(clf.predict(X_train) ==  Y_train).sum() * 1.0 / len(X_train)

(clf.predict(X_test) ==  Y_test).sum() * 1.0 / len(X_test)


df['Predictions'] = clf.predict(X_new).astype(int)
results = df.copy()



results.tail(13000-1300)[['ID', 'Predictions']].rename(columns={'Predictions':'Class'}).to_csv('submission.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(results.tail(13000-1300)[['ID', 'Predictions']].rename(columns={'Predictions':'Class'}))