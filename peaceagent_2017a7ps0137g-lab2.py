import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

%matplotlib inline

import xgboost as xgb

from IPython.display import HTML

import base64

import warnings

import operator

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.model_selection import cross_val_score, train_test_split,StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
def create_download_link(df, title = "Download CSV file",count=[0]):

    count[0] = count[0]+1

    filename = "data"+str(count[0])+".csv"

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
def evaluate(model,x,y,retrain=True):

    if retrain:model.fit(x,y)

    test_df=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

    test_id = test_df["id"].values

    test_df.drop(["id"],axis=1,inplace=True)

    tdf_values = test_df.values

    pred = model.predict(tdf_values)

    d={0:1,1:2,2:3,3:5,4:6,5:7}

    pred = [d[i] for i in pred]

    result = [[test_id[i],pred[i]] for i in range(len(pred))]

    result = pd.DataFrame(result,columns=["id","class"])

    return create_download_link(result)
train_df = pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

train_df["class"] = train_df["class"]-1

train_df = train_df.drop(["id"],axis=1)
train_df.describe()
train_df.drop_duplicates(inplace=True)
for i in range(0,7):

    print(i,len(train_df[train_df["class"] == i ] ) )
d={0:0,1:1,2:2,4:3,5:4,6:5}

train_df["class"]=train_df["class"].apply( lambda x:d[x])

for i in range(0,6):

    print(i,len(train_df[train_df["class"] == i ] ) )
plt.figure(figsize=(10,10))

sns.heatmap(data=train_df.corr(),annot=True)
x = train_df.drop(["class"],axis=1)

y=train_df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
pl_random_forest = Pipeline(steps=[('random_forest', RandomForestClassifier(n_estimators=20,random_state=42))])

scores = cross_val_score(pl_random_forest, x,y, cv=5,scoring='accuracy')

print('Accuracy : ', scores.mean())

pl_random_forest.fit(x_train,y_train)

pred=pl_random_forest.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_random_forest,x.values,y.values)
from rgf.sklearn import RGFClassifier

rgf = RGFClassifier(algorithm="RGF_Sib", test_interval=10)

n_folds = 4

rgf_scores = cross_val_score(rgf,x,y,cv=StratifiedKFold(n_folds))

print(sum(rgf_scores)/n_folds)

rgf.fit(x_train,y_train)

pred=rgf.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(rgf,x.values,y.values)
pl = xgb.XGBClassifier()

scores = cross_val_score(pl, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl.fit(x_train,y_train)

pred=pl.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl,x.values,y.values)
pl_r = xgb.XGBRFClassifier(n_estimators=20)

scores = cross_val_score(pl_r, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_r.fit(x_train,y_train)

pred=pl_r.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_r,x.values,y.values)
pl_ex = ExtraTreesClassifier(n_estimators=20)

scores = cross_val_score(pl_ex, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_ex.fit(x_train,y_train)

pred=pl_ex.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_ex,x.values,y.values)
pl_knn = KNeighborsClassifier(n_neighbors=5)

scores = cross_val_score(pl_knn, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_knn.fit(x_train,y_train)

pred=pl_knn.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_knn,x.values,y.values)
pl_bag = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=20))

scores = cross_val_score(pl_bag, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_bag.fit(x_train,y_train)

pred=pl_bag.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_bag,x.values,y.values)
pl_bag1 = BaggingClassifier()

scores = cross_val_score(pl_bag1, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_bag1.fit(x_train,y_train)

pred=pl_bag1.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_bag1,x.values,y.values)
pl_ada = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=20))

scores = cross_val_score(pl_ada, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_ada.fit(x_train,y_train)

pred=pl_ada.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_ada,x.values,y.values)
pl_adar = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=20,random_state=42))

scores = cross_val_score(pl_adar, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_adar.fit(x_train,y_train)

pred=pl_adar.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_adar,x.values,y.values)
pl_gbc = GradientBoostingClassifier()

scores = cross_val_score(pl_gbc, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_gbc.fit(x_train,y_train)

pred=pl_gbc.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_gbc,x.values,y.values)
#stacking 6:adaboost on extratree,extratree,randomforest,xgboostclassfier,baggingonrandom,rgfc

#1-voting:soft &grid-search-weights

#xgboost on result

#xgboost on probabilities

#1-7137 2-7318,3-7323,4-7647,5-701 6-698
predictors=[('rgf',RGFClassifier(algorithm="RGF_Sib", test_interval=10)),('extrat',ExtraTreesClassifier(n_estimators=20)),('adaex',AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=20))),('rf',RandomForestClassifier(n_estimators=20,random_state=42)),('xgb',xgb.XGBClassifier()),('bag',BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=20)))]

pl_vote = VotingClassifier(estimators=predictors, voting='soft')

scores = cross_val_score(pl_vote, x,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_vote.fit(x_train,y_train)

pred=pl_vote.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

evaluate(pl_vote,x.values,y.values)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import accuracy_score



parameters = {'weights':[[1,1,1,1,1,1],[2,1,1,1,1,1],[1,2,1,1,1,1],[1,1,2,1,1,1],[1,1,1,2,1,1],[2,2,2,2,1,1],[2,2,1,2,1,1],[2,1,1,2,1,1],[1,2,2,2,1,1],[2,1,2,2,1,1],[1,1,2,2,1,1]]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(VotingClassifier(estimators=predictors, voting='soft'),parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(x_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf_sv = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object
pred= best_clf_sv.predict(x_test)

print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)
test_df=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

test_id = test_df["id"].values

test_df.drop(["id"],axis=1,inplace=True)

pred = best_clf_sv.predict(test_df)

d={0:1,1:2,2:3,3:5,4:6,5:7}

pred = [d[i] for i in pred]

result = [[test_id[i],pred[i]] for i in range(len(pred))]

result = pd.DataFrame(result,columns=["id","class"])

create_download_link(result)
#stacking attempt

predictors=[('rgf',RGFClassifier(algorithm="RGF_Sib", test_interval=10)),('adaex',AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=20))),('rf',RandomForestClassifier(n_estimators=20,random_state=42)),('xgb',xgb.XGBClassifier()),('bag',BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=20)))]

res_dict={}

trained_models={}

for clf_name,clf in predictors:

    clf.fit(x,y)

    trained_models[clf_name]=clf

    res_dict[clf_name]=clf.predict_proba(x)

x_stack = np.concatenate(tuple([res_dict[i] for i in res_dict]),axis=1)

pl_stack = RandomForestClassifier(n_estimators=50,random_state=16)

scores = cross_val_score(pl_stack, x_stack,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

x_t_s, x_v_s, y_t_s, y_v_s = train_test_split(x_stack, y,test_size=0.2)

pl_stack.fit(x_t_s,y_t_s)

pred=pl_stack.predict(x_v_s)

print(len([ i for i in range(len(pred))if pred[i]==y_v_s.values[i]])/len(pred)*100)
test_df=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

test_id = test_df["id"].values

test_df.drop(["id"],axis=1,inplace=True)

t_dict={}

for clf_name,clf in predictors:

    t_dict[clf_name]=trained_models[clf_name].predict_proba(test_df)

    t_stack = np.concatenate(tuple([t_dict[i] for i in t_dict]),axis=1)

pred = pl_stack.predict(t_stack)

d={0:1,1:2,2:3,3:5,4:6,5:7}

pred = [d[i] for i in pred]

result = [[test_id[i],pred[i]] for i in range(len(pred))]

result = pd.DataFrame(result,columns=["id","class"])

create_download_link(result)
y_labels={}

for i in range(6):

    y_labels[i]=[1 if j==i else 0 for j in list(y.values)]

models={}

for i in range(6):

    clf =  xgb.XGBClassifier()

    scores = cross_val_score(clf, x,y_labels[i], cv=4,scoring='accuracy')

    print('Accuracy  ',i," : ",scores.mean())

    clf.fit(x,y_labels[i])

    models[i]=clf

results={}

for i in range(6):

    temp = models[i].predict_proba(x)

    results[i] = [j[1] for j in temp]

t={}

for i in range(len(results[0])):

   t[i]=[results[j][i] for j in range(6)] 

r={}

for i,j in t.items():

    r[i] = j.index(max(j))

res=[]

for i in range(len(r)):

    res.append(r[i])
test_df=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

test_id = test_df["id"].values

test_df.drop(["id"],axis=1,inplace=True)

results={}

for i in range(6):

    temp = models[i].predict_proba(test_df)

    results[i] = [j[1] for j in temp]

t={}

for i in range(len(results[0])):

   t[i]=[results[j][i] for j in range(6)] 

r={}

for i,j in t.items():

    r[i] = j.index(max(j))

res=[]

for i in range(len(r)):

    res.append(r[i])

pred=res

d={0:1,1:2,2:3,3:5,4:6,5:7}

pred = [d[i] for i in pred]

result = [[test_id[i],pred[i]] for i in range(len(pred))]

result = pd.DataFrame(result,columns=["id","class"])

create_download_link(result)
y_labels={}

for i in range(6):

    y_labels[i]=[1 if j==i else 0 for j in list(y.values)]

models={}

for i in range(6):

    clf =  xgb.XGBClassifier()

    scores = cross_val_score(clf, x,y_labels[i], cv=4,scoring='accuracy')

    print('Accuracy i: ',scores.mean())

    clf.fit(x,y_labels[i])

    models[i]=clf

results={}

for i in range(6):

    temp = models[i].predict_proba(x)

    results[i] = [j[1] for j in temp]

t={}

for i in range(len(results[0])):

   t[i]=[results[j][i] for j in range(6)] 

x_t=[]

for i in range(119):

    x_t.append(t[i])

pl_stack = RandomForestClassifier(n_estimators=50,random_state=16)

scores = cross_val_score(pl_stack, x_t,y, cv=4,scoring='accuracy')

print('Accuracy: ',scores.mean())

pl_stack.fit(x_t,y)
test_df=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

test_id = test_df["id"].values

test_df.drop(["id"],axis=1,inplace=True)

results={}

for i in range(6):

    temp = models[i].predict_proba(test_df)

    results[i] = [j[1] for j in temp]

t={}

for i in range(len(results[0])):

   t[i]=[results[j][i] for j in range(6)] 

y_t=[]

for j in range(80):

    y_t.append(t[j])

pred = pl_stack.predict(y_t)

d={0:1,1:2,2:3,3:5,4:6,5:7}

pred = [d[i] for i in pred]

result = [[test_id[i],pred[i]] for i in range(len(pred))]

result = pd.DataFrame(result,columns=["id","class"])

create_download_link(result)
import xgboost as xgb

dtrain = xgb.DMatrix( x_train, label = y_train)

dtest = xgb.DMatrix( x_test, label = y_test)

dall = xgb.DMatrix(x,label=y)

param = {

    'max_depth': 6,  # the maximum depth of each tree

    'eta': 0.2,  # the training step for each iteration

    'objective': 'multi:softmax',  # error evaluation for multiclass training

    'num_class': 6,

    "eval_metric":"mlogloss"}  # the number of classes that exist in this datset

num_round = 400 # the number of training iterations
bst = xgb.train(param, dall, num_round)

pred= bst.predict(dtest)

#print(len([ i for i in range(len(pred))if pred[i]==y_test.values[i]])/len(pred)*100)

xgb.to_graphviz(bst, num_trees=2)
xgb.plot_importance(bst)