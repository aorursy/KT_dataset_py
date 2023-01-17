import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt
people = pd.read_csv('../input/people.csv',

                       dtype={'people_id': np.str,

                              'activity_id': np.str,

                              'char_38': np.int32},

                       parse_dates=['date'])

act_train = pd.read_csv('../input/act_train.csv',

                        dtype={'people_id': np.str,

                               'activity_id': np.str,

                               'otcome': np.int8},

                        parse_dates=['date'])

act_test = pd.read_csv('../input/act_test.csv',

                        dtype={'people_id': np.str,

                               'activity_id': np.str,

                               'otcome': np.int8},

                        parse_dates=['date'])
act_train['date'].groupby(act_train.date.dt.date).count().plot(figsize=(10,5), label='Train')

act_test['date'].groupby(act_test.date.dt.date).count().plot(figsize=(10,5), label='Test')

plt.legend()

plt.show()
goods=act_train[act_train['outcome']==1]

bads=act_train[act_train['outcome']==0]



goods['date'].groupby(goods.date.dt.date).count().plot(figsize=(10,5),label='Good')

bads['date'].groupby(bads.date.dt.date).count().plot(figsize=(10,5),c='r',label='Bad')

plt.legend()

plt.show()
positive_counts=pd.DataFrame({'positive_counts' : act_train[act_train['outcome']==1].groupby('people_id',as_index=True).size()}).reset_index()

negative_counts=pd.DataFrame({'negative_counts' : act_train[act_train['outcome']==0].groupby('people_id',as_index=True).size()}).reset_index()

hstry=positive_counts.merge(negative_counts, on='people_id',how='outer')

hstry['positive_counts']=hstry['positive_counts'].fillna('0').astype(np.int64)

hstry['negative_counts']=hstry['negative_counts'].fillna('0').astype(np.int64)

hstry['profit']=hstry['positive_counts']-hstry['negative_counts']
hstry.sort_values(by='positive_counts',ascending=False).head(10)
hstry.sort_values(by='negative_counts',ascending=False).head(10)
hstry['profit'].describe()
hstry['prof_label']=((pd.to_numeric(hstry['profit']<-5).astype(int) * 4 )+

                     (pd.to_numeric(hstry['profit'].isin(range(-5,1))).astype(int) * 3)+                     

                     (pd.to_numeric(hstry['profit'].isin(range(1,6))).astype(int) * 2)+

                     (pd.to_numeric(hstry['profit']>5).astype(int) * 1 )

                    )

                     
plt.figure()

plt.hist(hstry['prof_label'],4,range=(1,5))

plt.show()
people2 = pd.merge(people, hstry, on='people_id', how='inner')

people2['positive_counts']=people2['positive_counts'].fillna('0').astype(np.int64)

people2['negative_counts']=people2['negative_counts'].fillna('0').astype(np.int64)

people2['profit']=people2['profit'].fillna('0').astype(np.int64)
# Turn all of the categorical data into integers



obs=['group_1']

for i in range (1,10):

    obs.append('char_'+str(i))



for x in obs:

    people2[x]=people2[x].fillna('type 0')

    people2[x]=people2[x].str.split(' ').str[1]



bools=[]

for i in range(10,38):

    bools.append('char_'+str(i))



for x in list(set(obs).union(set(bools))):

    people2[x]=pd.to_numeric(people2[x]).astype(int)

people2['date']=pd.to_numeric(people2['date']).astype(int)
#for x in bools:

#    plt.figure()

#    fig, ax= plt.subplots()

#    ax.set_xticks([1.5,2.5,3.5,4.5])

#    ax.set_xticklabels(('Very\nGood','Good','Bad','Very\nBad'))

#    fig.suptitle(x, fontsize=15)

#    neg=people2[people2[x]==0]

#    pos=people2[people2[x]==1]

#    plt.hist([pos['prof_label'],neg['prof_label']], 4,range=(1,5), 

#             normed=True, stacked=True, label=['Has Trait','No Trait'])

#    plt.legend()

#    plt.show()
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor

from sklearn.metrics import auc, mean_squared_error

from sklearn.cross_validation import train_test_split, cross_val_score
xfeats = list(people2.columns)

xfeats.remove('people_id')

xfeats.remove('profit')

xfeats.remove('prof_label')

xfeats.remove('positive_counts')

xfeats.remove('negative_counts')

print(xfeats)



X, Y = people2[xfeats],people2['prof_label']
X_train, X_test, y_train, y_test = train_test_split(

    X, Y, test_size=0.2, random_state=42)



clf = RandomForestRegressor(n_estimators=50)

clf.fit(X_train,y_train)



print(clf.feature_importances_)
sortedfeats=sorted(zip(xfeats,clf.feature_importances_), key=lambda x:x[1])

newfeats=[]

for i in range(1,6):

    newfeats.append(sortedfeats[len(sortedfeats) -i])

newfeats = [x[0] for x in newfeats]

print(newfeats)
X, Y = people2[newfeats],people2['prof_label']



X_train2, X_test2, y_train2, y_test2 = train_test_split(

    X, Y, test_size=0.2, random_state=42)



clf2 = RandomForestRegressor(n_estimators=100)

clf2.fit(X_train2,y_train2)



print(clf2.feature_importances_)
print(clf.score(X_test,y_test), clf2.score(X_test2,y_test2))

print(mean_squared_error(clf.predict(X_test),y_test),mean_squared_error(clf2.predict(X_test2),y_test2))
people2['pred']=clf.predict(people2[xfeats])

people2['pred2']=clf2.predict(people2[newfeats])
people2[['prof_label','pred']].sample(20)