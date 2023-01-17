import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie
%matplotlib inline
import time
import pandas_profiling
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from hyperopt import hp,Trials,fmin,tpe
df = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
df.head()
df.shape
print(df.isnull().sum())
pandas_profiling.ProfileReport(df)
# Dropping rows with more than 75% missing values
row = df.isnull().sum(axis=1)
junk_25 = int(df.shape[1] * 0.75)
row = row[row > junk_25]
index_to_delete = row.index.tolist()
df = df.drop(index_to_delete, axis=0)



# Finding columns with more than 25% missing
null_count_dict = df.isnull().sum().to_dict()
forty_percent = int(df.shape[0] * .25)
column_to_delete = []
for key, value in null_count_dict.items():
    if (value > forty_percent):
        column_to_delete.append(key)

# delete the column in column 40% missing
df = df.drop(column_to_delete, axis=1)

print(df.shape)



#Dump csv 
# df.to_csv("../input/new.csv",index=False)

# df = pd.read_csv("/home/user/Downloads/gtd/new.csv")
y = df['gname']
X = df.drop(['gname'], axis=1)
print(type(y))
y.value_counts()
class_count = df['gname'].value_counts()
keep_class = []
for i, v in class_count.items():
    if (v > 500):
        keep_class.append(i)
frame = pd.DataFrame(columns=df.columns)
for i in keep_class:
    frame = frame.append(df[df.gname == i])

y = frame['gname']
X = frame.drop(['gname'], axis=1)
for i in X.columns:
    if X[i].dtype == object:
        X[i] = X[i].fillna(X[i].mode()[0])

    else:
        X[i] = X[i].fillna(X[i].mean())


label = []
for i in X.columns:
    if X[i].dtype == object:
        l = LabelEncoder()
        X[i] = l.fit_transform(X[i])
        label.append(l)
X.boxplot(figsize=(30,10))
def detect_n_treat_outliers(data_frame, y,space, treat=True):
    print("==================================================")
    print("Outlier detection and treatment started ...")
    print("Space:", space)
    #     print("Label column name:", data_frame.columns[-1])
    # y = data_frame[data_frame.columns[-1]]
    # data_frame.drop([data_frame.columns[-1]], axis=1, inplace=True)
    X = data_frame

    dtypesss = X.dtypes

    #     print("Dtypes:;;", dtypesss)

    y_predicted = None
    params = space['params']

    if space['model'] == "DBSCAN":

        db = DBSCAN(**params)
        y_predicted = db.fit_predict(X)
        y_predicted = list(map(lambda x: 1 if x < 0 else 0, y_predicted))

    elif space['model'] == "EllipticEnvelope":
        elliptic = EllipticEnvelope(**params)
        for i in range(0, X.shape[0], 10000):
            elliptic.fit(X[i:10000 + i], y[i:10000 + i])
        y_predicted = elliptic.predict(X)
        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))

    elif space['model'] == "IsolationForest":
        iso = IsolationForest(**params)
        for i in range(0, X.shape[0], 10000):
            iso.fit(X[i:10000 + i], y[i:10000 + i])
        y_predicted = iso.predict(X)

        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))

    elif space['model'] == "OneClassSVM":
        ocv = OneClassSVM(**params)
        for i in range(0, X.shape[0], 10000):
            ocv.fit(X[i:10000 + i], y[i:10000 + i])
        y_predicted = ocv.predict(X)
        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))

    elif space['model'] == "LocalOutlierFactor":
        lof = LocalOutlierFactor(**params)
        for i in range(0, X.shape[0], 10000):
            lof.fit(X[i:10000 + i], y[i:10000 + i])
        y_predicted = lof._predict(X)
        y_predicted = list(map(lambda x: 1 if x == -1 else 0, y_predicted))

    elif space['model'] == "zscore":
        threshold = params['threshold']
        print("thold", threshold)
        score_frame = pd.DataFrame()
        for i in X.columns:
            score = zscore(X[i], axis=0, ddof=1)
            score_frame[i] = score
        score_frame = score_frame.abs()
        predicted_outliers = []
        for i in range(len(score_frame)):
            if any(score_frame.iloc[i] > threshold):
                predicted_outliers.append(1)
            else:
                predicted_outliers.append(0)
        y_predicted = predicted_outliers

    return y_predicted


def auto_detect_n_treat_outliers(df,y,hyperopt_trained_space, voting_percentage, treat=True):
    # Voting starts
    all_votes = [0] * df.shape[0]
    all_votes = np.array(all_votes)

    # iterating over each space and getting y_predicted
    for i in hyperopt_trained_space:
        y_predicted = detect_n_treat_outliers(df.copy(), y,i, treat=treat)
        y_predicted = np.array(y_predicted)
        all_votes += y_predicted

    voting_criteria = voting_percentage * (len(hyperopt_trained_space) ) * 1.0
    remove_rows = []
    for i in range(len(all_votes)):
        if all_votes[i] >= voting_criteria:
            remove_rows.append(i)

    final_remove_index = sorted(np.unique(remove_rows))
    return final_remove_index
hyperspace = [
             {"model": "IsolationForest", "params": {"n_estimators": 100,"contamination":0.05,"n_jobs": -1}},
             {"model": "LocalOutlierFactor", "params": {'n_neighbors': 50,"contamination":0.05,'novelty':True}},
             {"model": "DBSCAN", "params": {'eps': 0.7}},
             {"model": "OneClassSVM", "params": {'max_iter': 10}},
             {"model": "EllipticEnvelope", "params": {'contamination':0.05}},
             {"model": "zscore", "params": {'threshold':3 }}
            ]

final_remove_index = auto_detect_n_treat_outliers(X,y,hyperopt_trained_space=hyperspace,voting_percentage=0.83)
X.drop(X.index[final_remove_index], axis=0, inplace=True)
y.drop(y.index[final_remove_index], axis=0, inplace=True)
# Feature selection
f = f_classif(X, y)

class_f = dict(zip(X.columns, f[0]))
sorted_idx_class_f = sorted(class_f, key=(lambda key: class_f[key]), reverse=True)

#Taking top 25 features contributing
j = 25
X = X[sorted_idx_class_f[0:j]]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf = RandomForestClassifier()
for i in range(0, X_train.shape[0], 10000):
    clf.fit(X_train[i:10000 + i], y_train[i:10000 + i])
    print(i)
predicted = clf.predict(X_test)    
def model_evalueation(y_test, y_pred):
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1score = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    cf = confusion_matrix(y_true=y_test, y_pred=y_pred)
    return {"accuracy": accuracy, "f1_score": f1score, "confusion_matrix":cf}
score = model_evalueation(y_test,predicted)
print(score)
space = hp.choice("algo", [

    {"model": "RandomForestClassifier",
     "params": {
         "min_samples_split": hp.choice("min_samples_split", range(2, 10, 1)),
         "min_samples_leaf": hp.choice("min_samples_leaf", range(1, 5, 1)),
         "n_jobs": hp.choice("n_jobs", [-1]),

     }},

])


def optimization(space):
    params = space['params']
    model = space['model']
    print(params)
    clf = RandomForestClassifier(**params)

    for i in range(0, X_train.shape[0], 10000):
        clf.fit(X_train[i:10000 + i], y_train[i:10000 + i])

    predicted = clf.predict(X_test)
    f1 = f1_score(y_test, predicted, average='macro')
    print(f1)
    loss = (1 - f1)
    # print(loss)

    print("==============================================================")
    return loss


trials = Trials()

best = fmin(optimization, space, algo=tpe.suggest, max_evals=100, trials=trials, verbose=False)
min_loss = min(trials.losses())
print(best)
print('Minimum loss:', min_loss)
print("Best f1",1- min_loss)
