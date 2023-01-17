import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score
test=pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')

train=pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
test.head()
train.head()
nc = train.columns[train.isnull().any()]

nc
y=train['Class']

data = test.drop(['ID'],axis=1)

tr=train.drop(['ID','Class'],axis=1)
tr.head()
col = data.columns[data.dtypes == np.object]

col
# data['col11'].replace({

#     'Yes':1,

#     'No':0

#     },inplace=True)

# tr['col11'].replace({

#     'Yes':1,

#     'No':0

#     },inplace=True)

# data['col44'].replace({

#     'Yes':1,

#     'No':0

#     },inplace=True)

# tr['col44'].replace({

#     'Yes':1,

#     'No':0

#     },inplace=True)
ob=[]

for i in col:

    try:

        data[i] = data[i].apply(pd.to_numeric)

    except:

        ob.append(i)

ob
data = pd.get_dummies(data, columns=[i for i in ob])

tr = pd.get_dummies(tr, columns=[i for i in ob])

data.head()
from sklearn import preprocessing

#Performing Min_Max Normalization

# min_max_scaler = preprocessing.MinMaxScaler()

# np_scaled = min_max_scaler.fit_transform(tr)

# X_N = pd.DataFrame(np_scaled)

# np_scaled1 = min_max_scaler.fit_transform(data)

# X_t = pd.DataFrame(np_scaled1)

# data.head()



# StandardScaler

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaled_data=scaler.fit(tr).transform(tr)

X_N=pd.DataFrame(scaled_data,columns=tr.columns)

scaled_data=scaler.fit(data).transform(data)

X_t=pd.DataFrame(scaled_data,columns=data.columns)

X_N.tail()
# from sklearn.manifold import TSNE

# from sklearn.decomposition import PCA

# X_N = TSNE(n_components=2).fit_transform(X_N)

# X_t = TSNE(n_components=2).fit_transform(X_t)

# model=PCA(n_components=50)

# X_N=model.fit(X_N).transform(X_N)

# X_t=model.fit(X_t).transform(X_t)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_N, y, test_size=0.3, random_state=100)
from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,20,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)

    rf.fit(X_train, y_train)

    sc_train = rf.score(X_train,y_train)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_test,y_test)

    score_test_RF.append(sc_test)
plt.figure(figsize=(10,6))

train_score,=plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='green', markersize=5)

test_score,=plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed',  marker='o',

         markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])

plt.title('Fig4. Score vs. No. of Trees')

plt.xlabel('No. of Trees')

plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=100, max_depth = 10)

rf.fit(X_train, y_train)

rf.score(X_test,y_test)
# rf = RandomForestClassifier(n_estimators=2000, max_depth = 7)

# rf.fit(X_train, y_train)

# rf.score(X_test,y_test)
y_pred_RF = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred_RF))

print(classification_report(y_test, y_pred_RF))
from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 100)        #Initialize the classifier object



parameters = {'max_depth':[7, 10],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=100, max_depth = 10, min_samples_split =4)

rf_best.fit(X_train, y_train)

rf_best.score(X_test,y_test)
y_pred_RF_best = rf_best.predict(X_test)

confusion_matrix(y_test, y_pred_RF_best)
f=rf_best.predict(X_t)

f

res=[]

for i in f:

    res.append(i)

res1 = pd.DataFrame(res)

res1.reset_index(drop=True, inplace=True)

test.reset_index(drop=True, inplace=True)

final = pd.concat([test["ID"], res1], axis=1).reindex()

final.columns = ['ID','Class']

final
final.to_csv('rf4.csv',index=False)
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