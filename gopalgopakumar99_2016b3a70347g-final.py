!pip install --user numpy

!pip install --user pandas

!pip install --user matplotlib

!pip install --user seaborn

!pip install --user sklearn



import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

data_orig = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')

train = data_orig

data_orig = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')

test = data_orig
train.head()
train.info()
train.duplicated().sum()
train = pd.get_dummies(train, columns=['col2','col11','col37','col44','col56'])
train.head()
test.info()
test.duplicated().sum()
test = pd.get_dummies(test, columns=['col2','col11','col37','col44','col56'])
test.head()
f, ax = plt.subplots(figsize=(20, 20))

corr = train.drop(columns=['ID','Class']).corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
train = train.drop(train[to_drop], axis=1)
test = test.drop(test[to_drop], axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
xtrain = train.drop(columns=['ID', 'Class'])

ytrain = train['Class']

xtest = test.drop(columns=['ID'])

xtrain = pd.DataFrame(scaler.fit_transform(xtrain))

xtest = pd.DataFrame(scaler.fit_transform(xtest))
ytrain.head()
xtest.head()
xtrain.head()
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(xtrain, ytrain, test_size=.2,random_state=42 )
np.random.seed(42)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,20,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i)

    rf.fit(X_tr, y_tr)

    sc_train = rf.score(X_tr,y_tr)

    score_train_RF.append(sc_train)

    sc_test = rf.score(X_te,y_te)

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
rf = RandomForestClassifier(n_estimators=100, max_depth = 8)

rf.fit(X_tr, y_tr)

rf.score(X_te,y_te)
rf = RandomForestClassifier(n_estimators=2000, max_depth = 8)

rf.fit(X_tr, y_tr)

rf.score(X_te,y_te)
y_pred_RF = rf.predict(X_te)

confusion_matrix(y_te, y_pred_RF)
print(classification_report(y_te, y_pred_RF))
from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 100)        #Initialize the classifier object



parameters = {'max_depth':[6, 9],'min_samples_split':[2, 3, 4, 5]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(xtrain, ytrain)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=100, max_depth = 9, min_samples_split = 4)

rf_best.fit(X_tr, y_tr)

rf_best.score(X_te,y_te)
y_pred_RF_best = rf_best.predict(X_te)

confusion_matrix(y_te, y_pred_RF_best)
print(classification_report(y_te, y_pred_RF_best))
rf_best.fit(xtrain, ytrain)

y_test=rf_best.predict(xtest)
y_test
test['Class']=y_test

test[['ID', 'Class']].to_csv('sub3.csv', index=False)
test.head()
df = pd.DataFrame(test, columns = ['ID', 'Class']) 
df
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

create_download_link(df)