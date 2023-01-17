import pandas as pd
data = pd.read_csv("../input/train.csv",header = 0)

data_test = pd.read_csv("../input/test.csv",header = 0)

df = pd.DataFrame(data)

df_test = pd.DataFrame(data_test)
null_count = {}

for i in range(0,100000):

    for j in df.columns:

        if df[j][i] == "?":

            if j not in null_count:

                null_count[j] = 1

            else:

                null_count[j] = null_count[j] + 1
for j in null_count:

    if null_count[j] > 50000:

        df = df.drop(j,axis=1)

        df_test = df_test.drop(j,axis=1)
null_cols = []

null_cols.append('COB FATHER')

null_cols.append('COB MOTHER')

null_cols.append('COB SELF')

null_cols.append('Hispanic')

rows_to_drop = []

for i in range(0,100000):

    for j in null_cols:

        if df[j][i] == '?':

            rows_to_drop.append(i)

            break
df = df.drop(rows_to_drop)
import numpy as np

col_list = df.columns

for col in col_list:

    if df[col].dtype != np.int64 and col != 'Weight' and col != 'Class':

        if len(df[col].unique()) >= 30:

            df = df.drop(col,axis = 1)

    if col != 'Class' and df_test[col].dtype != np.int64 and col != 'Weight':

        if len(df_test[col].unique()) >= 30:

            df_test = df_test.drop(col,axis = 1)

df.info()

df_test.info()
col_list = df.columns

rep_col = []

for col in col_list:

    if df[col].dtype != np.int64 and col != 'Weight' and col != 'Class':

        rep_col.append(col)

rep_col
one_hot = []

one_hot_test = []

for column in rep_col:

    one_hot.append(pd.get_dummies(df[column], prefix = column))

    one_hot_test.append(pd.get_dummies(df_test[column], prefix = column))
df = df.drop(rep_col,axis=1)

df_test = df_test.drop(rep_col,axis=1)
for i in range(0,len(one_hot)):

    df = df.join(one_hot[i])

    df_test = df_test.join(one_hot_test[i])
import sklearn
from sklearn.model_selection import train_test_split
X = df.drop(['ID','Class'],axis=1)
Y = df['Class']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
from sklearn import preprocessing

#Performing Min_Max Normalization

min_max_scaler2 = preprocessing.MinMaxScaler()

np_scaled_xtrain = min_max_scaler2.fit_transform(X_train)

X_train_n = pd.DataFrame(np_scaled_xtrain)

np_scaled_xtest = min_max_scaler2.fit_transform(X_test)

X_test_n = pd.DataFrame(np_scaled_xtest)
from sklearn.neighbors import KNeighborsClassifier

train_acc = []

test_acc = []

for i in range(1,15):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train_n,Y_train)

    acc_train = knn.score(X_train_n,Y_train)

    train_acc.append(acc_train)

    acc_test = knn.score(X_test_n,Y_test)

    test_acc.append(acc_test)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Test Accuracy"])

plt.title('Accuracy vs K neighbors')

plt.xlabel('K neighbors')

plt.ylabel('Accuracy')
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train_n,Y_train)

knn.score(X_test_n,Y_test)
from sklearn.naive_bayes import GaussianNB as NB
nb = NB()

nb.fit(X_train,Y_train)

nb.score(X_test,Y_test)
from sklearn.linear_model import LogisticRegression

train_acc = []

test_acc = []

for i in range(1,50):

    lg = LogisticRegression(solver = 'liblinear', C = i, multi_class = 'ovr')

    lg.fit(X_train,Y_train)

    train_acc.append(lg.score(X_train,Y_train))

    test_acc.append(lg.score(X_test,Y_test))
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []

score_test_RF = []



for i in range(1,50,1):

    rf = RandomForestClassifier(n_estimators=i)

    rf.fit(X_train, Y_train)

    sc_train = rf.score(X_train,Y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,Y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,50,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,50,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score



rf_temp = RandomForestClassifier(n_estimators = 16)        #Initialize the classifier object



parameters = {'max_depth':[3, 5, 8, 10, 12],'min_samples_split':[2, 3, 4, 5, 6, 7]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, Y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf = RandomForestClassifier(n_estimators = 16, max_depth = 12, min_samples_split = 3)

rf.fit(X_train,Y_train)

rf.score(X_test,Y_test)
df_test_new = df_test.drop('ID',axis=1)
df_test_new = df_test_new.drop('Hispanic_?',axis=1)

np_scaled_dftestn = min_max_scaler2.fit_transform(df_test_new)

dftestn = pd.DataFrame(np_scaled_dftestn)
preds_submit = nb.predict(df_test_new)
dict = {"ID" : df_test["ID"], "Class" : preds_submit}

final = pd.DataFrame(dict, columns = ["ID","Class"])
from IPython.display import HTML 

import pandas as pd 

import numpy as np 

import base64 

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)     

    b64 = base64.b64encode(csv.encode())     

    payload = b64.decode()     

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'    

    html = html.format(payload=payload,title=title,filename=filename)     

    return HTML(html) 

create_download_link(final)