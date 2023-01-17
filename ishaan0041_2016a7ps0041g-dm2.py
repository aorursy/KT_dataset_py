import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn.preprocessing as prep

from sklearn.model_selection import train_test_split
data_train = pd.read_csv("../input/train.csv")
data_train
y = data_train["Class"]

data_train = data_train.drop(["Class"], axis=1)

data_train
data_train.head()
data_train.info()
data_test = pd.read_csv("../input/test.csv")
data_test
data = pd.concat([data_train,data_test])

data
data.info()
object_cols = []



for col in data.columns:

    if data[col].dtype == "object":

        print(col)

        object_cols.append(col)

    
for col in object_cols :

    print("{}\n{}\n".format(col,data[col].value_counts()))
drop_cols = ["Worker Class","MOC","MIC","Enrolled","MLU","Reason","Area","State","MSA","REG","MOVE","Live","PREV","Teen","Fill"]

data = data.drop(drop_cols,axis=1)

for col in drop_cols :

    object_cols.remove(col)

data.info()
for col in object_cols :

    print("{}\n{}\n".format(col,data[col].value_counts()))
data = data.replace("?",np.NaN)
for col in object_cols :

    print("{}\n{}\n".format(col,data[col].value_counts()))


data["COB SELF"] = data["COB SELF"].fillna("c24")

data["COB MOTHER"] = data["COB MOTHER"].fillna("c24")

data["COB FATHER"] = data["COB FATHER"].fillna("c24")

data["Hispanic"] = data["Hispanic"].fillna("HA")



for col in object_cols :

    print("{}\n{}\n".format(col,data[col].unique()))
data.info()
## Label Encoding



encode = prep.LabelEncoder()

for col in object_cols :

    data[col] = encode.fit_transform(data[col])

data
data.info()
### SAMPLING CASE ###



data_sample = data.copy()

data_sample = data_sample[:100000]

data_sample["Class"] = y

data_sample_1 = data_sample[data_sample["Class"]==1]

data_sample_0 = data_sample[data_sample["Class"]==0]

sampling_1 = [data_sample_1 for i in range(14)]

sampling_1.append(data_sample_0)

data_over_sampling_1 = pd.concat(sampling_1)

y = data_over_sampling_1["Class"]

data_over_sampling_1 = data_over_sampling_1.drop(["Class"],axis=1)

data_over_sampling_1
### SAMPLING CASE ###



data_over_sampling_1 = data_over_sampling_1.drop(["ID"],axis=1)

X = data_over_sampling_1.values

X = prep.MinMaxScaler().fit_transform(X)

idd = data["ID"][100000:]

data = data.drop(["ID"],axis=1)

X_predicted = data[100000:].values

X_predicted = prep.MinMaxScaler().fit_transform(X_predicted)

X_predicted
## SAMPLING CASE ###



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05,shuffle=True)
## Naive Bayes



from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

print(nb.score(X_val,y_val))
y_pred_nb = nb.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_nb))
## Random Forest Classifier



from sklearn.ensemble import RandomForestClassifier

score_train_RF = []

score_test_RF = []

for i in range(1,25,1):

    rf = RandomForestClassifier(n_estimators=i,class_weight="balanced")

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_val,y_val)

    score_test_RF.append(sc_test)
plt.figure(figsize=(12,6))

train_score,=plt.plot(range(1,25,1),score_train_RF,color='blue', linestyle='dashed',markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(1,25,1),score_test_RF,color='red',linestyle='dashed',markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=2,class_weight="balanced")

rf.fit(X_train, y_train)

rf.score(X_val,y_val)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer,f1_score

rf_temp = RandomForestClassifier(n_estimators = 2,class_weight="balanced") 

parameters = {'max_depth':[3, 5, 8, 10],'min_samples_split':[2, 3, 4, 5]} 

scorer = make_scorer(f1_score, average = 'micro')

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer) 

grid_fit = grid_obj.fit(X_train, y_train)

best_rf = grid_fit.best_estimator_

print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators = 2, max_depth = 10, min_samples_split = 5,class_weight="balanced")

rf_best.fit(X_train, y_train)

rf_best.score(X_val,y_val)
y_pred_RF_best = rf_best.predict(X_test)
y_pred = rf_best.predict(X_predicted)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_RF_best))
predictions = pd.DataFrame(columns=["ID","Class"])

predictions["ID"] = id

predictions["Class"] = y_pred
## AdaBoost Classifier



from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

adc = AdaBoostClassifier(n_estimators=14,base_estimator=nb,learning_rate=1)

adc.fit(X_train,y_train)
y_pred_adb = adc.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_adb))
## Logistic Regression Classifier



from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(solver = 'lbfgs', C=8, multi_class = 'multinomial')

lg.fit(X_train,y_train)

lg.score(X_val,y_val)
y_pred_lr = lg.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_lr))
from IPython.display import HTML

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

     csv = df.to_csv(index=False)

     b64 = base64.b64encode(csv.encode())

     payload = b64.decode()

     html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

     html = html.format(payload=payload,title=title,filename=filename)

     return HTML(html)

create_download_link(predictions)