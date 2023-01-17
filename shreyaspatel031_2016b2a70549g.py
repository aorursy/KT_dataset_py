import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
inp = pd.read_csv("../input/training/train.csv").dropna()
inp['agent'] = (inp['id']-1)%7
inp.drop(columns=['id','a0','a1','a2','a3','a4','a5','a6'],inplace=True)
display(inp)
data =[]
for i in range(7):
#     display(inp[inp.agent==i])
    data.append(inp[inp.agent==i])
    data[i].drop(columns=['agent'],inplace=True)
display(data)

corr = data[0].corr() 
fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(inp.corr(), mask=np.zeros_like(inp.corr(), dtype=np.bool), square=True, ax=ax, annot = False)
print(rfc_random.best_params_)

scores = cross_validate(rfc_random.best_estimator_, inp, op, cv=3,return_train_score=True,n_jobs=-1
                        ,scoring='neg_root_mean_squared_error'
                       )
print("agent "+str(i)+" : "+str(-scores['test_score'].mean())+" "+str(-scores['train_score'].mean()))
scores
%%time
total={'test':[],'train':[]}
models=[
    RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_depth=2000,
    ),
    RandomForestRegressor( 
        n_estimators=50,
        min_samples_leaf=2,
        max_depth=5000,
    ),
    RandomForestRegressor( 
        n_estimators=50,
        min_samples_leaf=4,
        max_depth=4000,
    ),
    RandomForestRegressor( 
        n_estimators=100,
        min_samples_leaf=1
    ),
    RandomForestRegressor(),
    RandomForestRegressor( 
        n_estimators=10,
        min_samples_leaf=1,
        max_depth=3000,
    ),
    RandomForestRegressor( 
        n_estimators=10,
        min_samples_leaf=4,
        max_depth=2000,
    ),

]
for i in range(7):
    print(i)
    op = data[i].label
    inp = data[i].drop(columns=['label'])    
    scores = cross_validate(models[i], inp, op, cv=3,return_train_score=True,n_jobs=-1
                            ,scoring='neg_root_mean_squared_error'
                           )
    print("agent "+str(i)+" : "+str(-scores['test_score'].mean())+" "+str(-scores['train_score'].mean()))
    total['test'].append(-scores['test_score'].mean())
    total['train'].append(-scores['train_score'].mean())
    
print(total)
total={'test':[2.6128766726667263,2.395973832665174,2.985475947132634,2.1907463181372333,3.8860396394434855,2.8632204831324444,3.4490153124245277],
       'train':[0.7995500657852489,0.642968178133839,0.5678964262941829,0.6256465534740411,0.6675156055103281,0.5990221337948212,0.5690796306094336]
      }
display(np.mean(total['test']),np.mean(total['train']) )
# forest vanilla
total={'test': [2.581057069165214, 2.4333940365637043, 2.951062990130309, 2.2050804836384477, 3.7454605592271633, 2.7527176066497745, 2.974621165297814], 
       'train': [0.956152301761899, 0.7914662880919233, 0.9490401638012055, 0.6300178241021209, 0.6642892842715221, 0.6989687594906107, 0.9992019453180377]}
display(np.mean(total['test']),np.mean(total['train']) )
%%time

models=[
    RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_depth=2000,
    ),
    RandomForestRegressor( 
        n_estimators=50,
        min_samples_leaf=2,
        max_depth=5000,
    ),
    RandomForestRegressor( 
        n_estimators=50,
        min_samples_leaf=4,
        max_depth=4000,
    ),
    RandomForestRegressor( 
        n_estimators=100,
        min_samples_leaf=1
    ),
    RandomForestRegressor( 
        n_estimators=50,
        min_samples_leaf=8,
        max_depth=None,
    ),
    RandomForestRegressor( 
        n_estimators=10,
        min_samples_leaf=1,
        max_depth=3000,
    ),
    RandomForestRegressor( 
        n_estimators=10,
        min_samples_leaf=4,
        max_depth=2000,
    ),

]
for i in range(7):
    op = data[i].label
    inp = data[i].drop(columns=['label']) 
    models[i].fit(inp,op)
def pred( x , agent ):
#     print(x)
    return models[agent].predict(x)[0]
    
test = pd.read_csv('../input/bits-f464-l1/test.csv')
test['agent'] = (test['id']-1)%7
test.drop(columns=['id','a0','a1','a2','a3','a4','a5','a6'],inplace=True)
h=[]
display(test)
# print( pred(data[0].loc[14][:'time'].values.reshape(1,95),0) )
for i in test.index:        
    inp = test.loc[i][:"time"]
    op = test.loc[i].agent
#     print(inp,op )
    h.append( pred(inp.values.reshape(1,len(inp)) , int(op) ) )
print(h)

sub = pd.DataFrame(columns=['id','label'])
test = pd.read_csv('../input/bits-f464-l1/test.csv')
sub.id = test.id
sub.label = h
sub.to_csv('ML_Lab1_sub0.csv',index=False)
test
import os
cwd = os.getcwd()
sub.to_csv('ML_Lab1_sub0.csv',index=False)