import pandas as pd

import numpy  as np

import sklearn.preprocessing as prep

import matplotlib.pyplot as plt

from   sklearn.model_selection import train_test_split 
df_tr = pd.read_csv("../input/train.csv")

y = df_tr["Class"]

df_tr = df_tr.drop(["Class"],axis=1)

df_tr.head()
df_te = pd.read_csv("../input/test.csv")

df_te.head()
df = pd.concat([df_tr,df_te])

df.head()
df.info()
df.describe()
column_object = []

for col in df.columns :

    if df[col].dtype == "object" :

        print(col)

        column_object.append(col)
for col in column_object :

    print("{}\n{}\n".format(col,df[col].value_counts()))
dropped = ["Worker Class","MOC","MIC","Enrolled","MLU","Reason","Area","State","MSA","REG","MOVE","Live","PREV","Teen","Fill"]

df = df.drop(dropped,axis=1)

for col in dropped :

    column_object.remove(col)

for col in column_object :

    print("{}\n{}\n".format(col,df[col].value_counts()))
df = df.replace("?",np.NaN)

df["COB MOTHER"] = df["COB MOTHER"].fillna("c24")

df["COB SELF"] = df["COB SELF"].fillna("c24")

df["Hispanic"] = df["Hispanic"].fillna("HA")

df["COB FATHER"] = df["COB FATHER"].fillna("c24")

for col in column_object :

    print("{}\n{}\n".format(col,df[col].unique()))
df.info()
encoded = prep.LabelEncoder()

for col in column_object :

    df[col] = encoded.fit_transform(df[col])
df.head()
df.info(verbose=True)


df_sample = df.copy()

df_sample = df_sample[:100000]

df_sample["Class"] = y

df_sample_0 = df_sample[df_sample["Class"]==0]

df_sample_1 = df_sample[df_sample["Class"]==1]



sample_1 = [df_sample_1 for i in range(14)]

sample_1.append(df_sample_0)

df_over_sample_1 = pd.concat(sample_1)

y = df_over_sample_1["Class"]

df_over_sample_1 = df_over_sample_1.drop(["Class"],axis=1)

df_over_sample_1
df_over_sample_1 = df_over_sample_1.drop(["ID"],axis=1)

x = df_over_sample_1.values

x = prep.MinMaxScaler().fit_transform(x)

idd = df["ID"][100000:]

df = df.drop(["ID"],axis=1)

x_pred = df[100000:].values

x_pred = prep.MinMaxScaler().fit_transform(x_pred)

x_pred
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.06,shuffle=True)

x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.06,shuffle=True)
#DECISIONTREE

from sklearn.tree import DecisionTreeClassifier

train_acc = []

test_acc = []

for i in range(1,15):

    dTree = DecisionTreeClassifier(max_depth=i)

    dTree.fit(x_tr,y_tr)

    acc_train = dTree.score(x_tr,y_tr)

    train_acc.append(acc_train)

    acc_test = dTree.score(x_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score = plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed',markerfacecolor='green', markersize=5)

test_score = plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])

plt.title('Accuracy vs Max Depth')

plt.xlabel('Max Depth')

plt.ylabel('Accuracy')
from sklearn.tree import DecisionTreeClassifier

train_acc = []

test_acc = []

for i in range(2,30):

    dTree = DecisionTreeClassifier(max_depth = 14, min_samples_split=i, random_state = 42)

    dTree.fit(x_tr,y_tr)

    acc_train = dTree.score(x_tr,y_tr)

    train_acc.append(acc_train)

    acc_test = dTree.score(x_val,y_val)

    test_acc.append(acc_test)
plt.figure(figsize=(10,6))

train_score = plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed',markerfacecolor='green', markersize=5)

test_score = plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',markerfacecolor='blue', markersize=5)

plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])

plt.title('Accuracy vs min_samples_split')

plt.xlabel('min_samples_split')

plt.ylabel('Accuracy')
dTree = DecisionTreeClassifier(max_depth=6,min_samples_split=10,random_state = 42)

dTree.fit(x_tr,y_tr)

dTree.score(x_val,y_val)
y_pred_dt = dTree.predict(x_te)
from sklearn.metrics import classification_report

print(classification_report(y_te, y_pred_dt))
#GAUSSIANNAIVEBAYES

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_tr,y_tr)

print(nb.score(x_val,y_val))
y_pred_nb = nb.predict(x_te)
from sklearn.metrics import classification_report

print(classification_report(y_te, y_pred_nb))
y_pred = nb.predict(x_pred)
#LOGISTICREGRESSION

from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial')

lg.fit(x_tr,y_tr)

lg.score(x_val,y_val)
y_pred_LR = lg.predict(x_te)
from sklearn.metrics import classification_report

print(classification_report(y_te, y_pred_LR))
y_pred = lg.predict(x_pred)

answer = pd.DataFrame(columns=["ID","Class"])

answer["ID"] = idd

answer["Class"] = y_pred

answer.to_csv("answer.csv",index=False)
#RANDOMFOREST

from sklearn.ensemble import RandomForestClassifier

score_train_RF = []

score_test_RF = []

for i in range(1,25,1):

    rf = RandomForestClassifier(n_estimators=i,class_weight="balanced")

    rf.fit(x_tr, y_tr)

    sc_train = rf.score(x_tr,y_tr)

    score_train_RF.append(sc_train)

    sc_test = rf.score(x_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(1,25,1),score_train_RF,color='blue', linestyle='dashed',markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,25,1),score_test_RF,color='red',linestyle='dashed',markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=12,class_weight="balanced")

rf.fit(x_tr, y_tr)

rf.score(x_val,y_val)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer,f1_score

rf_temp = RandomForestClassifier(n_estimators = 12,class_weight="balanced") 

parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]} 

scorer = make_scorer(f1_score, average = 'micro')

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) 

grid_fit = grid_obj.fit(x_tr, y_tr)

best_rf = grid_fit.best_estimator_

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 12, max_depth = 10, min_samples_split = 3,class_weight="balanced")

rf_best.fit(x_tr, y_tr)

rf_best.score(x_val,y_val)
y_pred_RF_best = rf_best.predict(x_te)
from sklearn.metrics import classification_report

print(classification_report(y_te, y_pred_RF_best))
y_pred = rf_best.predict(x_pred)
y_pred
answer = pd.DataFrame(columns=["ID","Class"])

answer["ID"] = idd

answer["Class"] = y_pred
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = df.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link(answer)