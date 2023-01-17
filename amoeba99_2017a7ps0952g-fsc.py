import numpy as np

np.random.seed(42)

import pandas as pd

import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
df = pd.read_csv("../input/data-mining-assignment-2/train.csv", sep=',')
def dpp(df):

    #one hot encoding

    for col in df.columns:

        if(df[col].dtype==object and col!='ID' and col!='Class'):

            one_hot = pd.get_dummies(df[col],prefix=col,prefix_sep='_')

            df = df.drop(col,axis = 1)

            # Join the encoded df

            df = df.join(one_hot)

        

    df = df.drop(['ID'], axis = 1)

    

    return df
df = dpp(df)
y = df['Class']

X = df.drop(['Class'], axis = 1)
from sklearn import preprocessing



def scaler_transform(min_max_scaler, X):

    np_scaled = min_max_scaler.transform(X)

    X_N = pd.DataFrame(np_scaled)

    return X_N



def scaler_fit(X):

    #Performing Min_Max Normalization

    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler = min_max_scaler.fit(X)

    return min_max_scaler, scaler_transform(min_max_scaler, X)
min_max_scaler, X = scaler_fit(X)
X.head()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=8)

dtree.fit(X, y)

dtree.feature_importances_
selected = (dtree.feature_importances_ > 0)

selected
res = X[X.columns[selected]] 

X = X[X.columns[selected]]

X.head()
res.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
X_train.head()
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier



score_train_RF = []

score_test_RF = []



for i in range(5,20,1):

    rf = RandomForestClassifier(n_estimators = 100, max_depth=i, random_state=42)

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

plt.title('Fig4. Score vs. Max Depth')

plt.xlabel('Max Depth')

plt.ylabel('Score')
from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 1000)        #Initialize the classifier object



parameters = {'max_depth':[5, 6, 7],'min_samples_split':[6, 7, 8, 9, 10, 11, 12], 'random_state':[42]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
from sklearn.model_selection import GridSearchCV



rf_temp = RandomForestClassifier(n_estimators = 1000)        #Initialize the classifier object



parameters = {'max_depth':[8, 10],'min_samples_split':[6, 7, 8, 9, 10, 11, 12], 'random_state':[42]}    #Dictionary of parameters



scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier



grid_fit = grid_obj.fit(X_train, y_train)        #Fit the gridsearch object with X_train,y_train



best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object



print(grid_fit.best_params_)
rf_best = RandomForestClassifier(n_estimators=2000, max_depth = 8, min_samples_split = 6, random_state = 42)

rf_best.fit(X_train, y_train)

rf_best.score(X_train, y_train)

rf_best.score(X_test,y_test)
from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score



scorer_f1 = make_scorer(f1_score, average = 'micro')



cv_results = cross_validate(rf_best, X, y, cv=10, scoring=(scorer_f1), return_train_score=True)

print(cv_results.keys())

print("Train Accuracy for 3 folds= ",np.mean(cv_results['train_score']))

print("Test Accuracy for 3 folds = ",np.mean(cv_results['test_score']))
otest_df = pd.read_csv("../input/data-mining-assignment-2/test.csv", sep=',')

test_df = dpp(otest_df)



X_train = X

y_train = y

X_test = scaler_transform(min_max_scaler, test_df)
X_test.info()
res.head()
X_test.head()
AB = X_test[list(res.columns.values)].copy()
AB.head()
X_test = AB
X_test.head()
rf_best_f = RandomForestClassifier(n_estimators=2000, max_depth = 8, min_samples_split = 6)

rf_best_f.fit(X_train, y_train)
rf_best_f.predict(X_test)
frame1 = pd.DataFrame(rf_best_f.predict(X_test))

frame1.columns = ['Class']



frame2 = otest_df['ID']



result = pd.concat([frame2, frame1], axis=1)

result.to_csv('submission.csv', index = False)

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

create_download_link(result)