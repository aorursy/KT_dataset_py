import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate,GridSearchCV,cross_val_score,train_test_split,StratifiedKFold

from sklearn.preprocessing import StandardScaler,LabelEncoder,RobustScaler,MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

from sklearn.metrics import f1_score,pairwise_distances

from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans

from sklearn.metrics import make_scorer

%matplotlib inline

from IPython.display import HTML

import base64

import warnings

import operator

import seaborn as sns

warnings.filterwarnings("ignore")

from collections import defaultdict,Counter



def create_download_link(df, title = "Download CSV file",count=[0]):

    count[0] = count[0]+1

    filename = "data"+str(count[0])+".csv"

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

def generate_submission(model,train_x,train_y,test_x):

    model.fit(train_x,train_y)

    test_y=model.predict(test_x)

    test["Class"]=test_y

    return create_download_link(test[["ID","Class"]])
train_path = '../input/data-mining-assignment-2/train.csv'

test_path = '../input/data-mining-assignment-2/test.csv'

train = pd.read_csv(train_path)

test = pd.read_csv(test_path)
d1 = {"Silver":0,"Gold":1,"Diamond":2,"Platinum":3}

d3 = {"Male":0,"Female":1}

d2 = {"Yes":0,"No":1}

d4 = {"Low":0,"Medium":1,"High":2}

train["col2"] = train["col2"].apply(lambda x:d1[x])

test["col2"] = test["col2"].apply(lambda x:d1[x])

train["col11"] = train["col11"].apply(lambda x:d2[x])

test["col11"] = test["col11"].apply(lambda x:d2[x])

train["col37"] = train["col37"].apply(lambda x:d3[x])

test["col37"] = test["col37"].apply(lambda x:d3[x])

train["col44"] = train["col44"].apply(lambda x:d2[x])

test["col44"] = test["col44"].apply(lambda x:d2[x])

train["col56"] = train["col56"].apply(lambda x:d4[x])

test["col56"] = test["col56"].apply(lambda x:d4[x])
correlation = train.drop(columns=["ID","Class"]).corr().abs()

drop=[]

for i in range(len(correlation.columns)):

        for j in range(i):

            if (correlation.iloc[i, j] >= 0.9) :

                colname1 = correlation.columns[i] 

                colname2 = correlation.columns[j] 

                drop.append([correlation.iloc[i, j],sorted([colname1,colname2])])
to_drop=set()

for i in drop:

    if i[0]>0.9:to_drop.add(i[1][0])

len(to_drop)
train_1=train.drop(columns=list(to_drop)+["ID","Class"])

test_1=test.drop(columns=list(to_drop)+["ID"])
scaler = RobustScaler()

#scaler = MinMaxScaler()

columns = train_1.columns

train_1 = pd.DataFrame(scaler.fit_transform(train_1))

test_1 = pd.DataFrame(scaler.transform(test_1))

train_1.columns = columns

test_1.columns = columns

train_y = train["Class"]
train_shift = train_1.copy(deep=True)

test_shift = test_1.copy(deep=True)

train_shift["Class"]=0

test_shift["Class"]=1

df = test_shift.append(train_shift.sample(300,random_state=42))

x = df.drop(columns = ["Class"])

y = df["Class"]

scv = StratifiedKFold(n_splits=5)

model =  RandomForestClassifier(n_estimators =50 , max_depth = 5,random_state=42)

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=42,stratify=y)

importances = []

for col in x.columns:

    model.fit(np.array(X_train[col]).reshape(-1,1), y_train)

    col_score = roc_auc_score(y_test,model.predict(np.array(X_test[col]).reshape(-1,1)))

    importances.append((col_score,col))
importances = [(np.mean(i[0]),i[1]) for i in importances]

sorted(importances,reverse=True)
to_drop=[i[1] for i in importances if i[0]>0.8]

to_drop
train_2=train_1.drop(columns=list(to_drop))

test_2=test_1.drop(columns=list(to_drop))
best_model = RandomForestClassifier(max_depth=8,max_features=4,min_samples_leaf=2,min_samples_split=8,

                                    n_estimators=200,class_weight='balanced',bootstrap=True,random_state=20)

scorer_f1 = make_scorer(f1_score, average = 'micro')

cv_results = cross_validate(best_model, train_2, train_y, cv=scv, scoring=(scorer_f1), return_train_score=True)

print("Train Score for 5 folds= ",np.mean(cv_results['train_score']))

print("Test Score for 5 folds = ",np.mean(cv_results['test_score']))
generate_submission(best_model,train_2,train_y,test_2)