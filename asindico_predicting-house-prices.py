import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing



train_df = pd.read_csv("../input/train.csv")

macro = pd.read_csv("../input/macro.csv")



train_df.shape
def _revrt(X,m=None):

    """

    Inverse of forrt. Equivalent to Munro (1976) REVRT routine.

    """

    if m is None:

        m = len(X)

    i = int(m // 2+1)

    y = X[:i] + np.r_[0,X[i:],0]*1j

    return np.fft.irfft(y)*m



from statsmodels.nonparametric import kdetools



# replace the implementation with new method.

kdetools.revrt = _revrt
train_df['price_doc'].describe()

prices  =train_df['price_doc']
f,axarray = plt.subplots(1,2)

f.set_size_inches(13,10)

axarray[0].set_title("price")

sns.distplot(train_df.price_doc.values, bins=50, kde=False,ax =axarray[0] )

lprices = np.log(prices)

axarray[1].set_title("price log")

sns.distplot(lprices, bins=50, kde=False,ax = axarray[1])

plt.show()
print (len(train_df[train_df['num_room']>0])/float(len(train_df)))

print (train_df[train_df['num_room']>0]['num_room'].describe())
nrsq = train_df[train_df['num_room']>0][['full_sq','num_room']]

nrsq.corr()
f,axarray = plt.subplots(1,2)

f.set_size_inches(13,8)

sns.distplot(train_df['full_sq'], bins=50, kde=False,ax =axarray[0])

lsq = np.log(train_df['full_sq']+0.0001)

train_df['full_sq']=lsq

sns.distplot(lsq, bins=50, kde=False,ax=axarray[1])

plt.show()
print (len(train_df[train_df['full_sq']>=0])/float(len(train_df)))

print (train_df[train_df['full_sq']>=0]['full_sq'].describe())
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

import math



nr = train_df[train_df['num_room']>0]['full_sq']

nrp = train_df[train_df['num_room']>0]['num_room']

X_train, X_test, y_train, y_test = train_test_split(nr, nrp, test_size=0.3, random_state=42)



room_pred = DecisionTreeRegressor()

room_pred.fit(X_train[:, None],y_train)

pred = room_pred.predict(X_test[:, None])

print (r2_score(y_test.astype(int),pred.astype(int)))



fsqnr = np.array(train_df[['full_sq','num_room']])

nr = [int(np.round(room_pred.predict(np.array([el[0]]).reshape(1, -1)))) if math.isnan(el[1]) else el[1] for el in fsqnr]
train_df['num_room'] = nr

nf = pd.DataFrame(pd.Series(nr),columns=["num_room"])

nf = pd.concat([nf,pd.DataFrame(train_df['full_sq'],columns=["full_sq"])],axis=1)

nf.head()
sns.pairplot(pd.concat([nf,prices],axis=1),size = 3.5)

plt.show()
print (len(train_df[train_df['floor']>=-10])/float(len(train_df)))

print (train_df[train_df['floor']>=-10]['floor'].describe())
fl = [5 if math.isnan(fl) else fl for fl in train_df['floor']]

train_df['floor']=fl

print (len(train_df[train_df['floor']>=0])/float(len(train_df)))

nf = pd.concat([nf,pd.DataFrame(fl,columns=["floor"])],axis=1)

nf.head()
f,axarray = plt.subplots(1,2)

f.set_size_inches(13,8)

sns.distplot(train_df['floor'], bins=50, kde=False,ax=axarray[0])

plt.xlabel('floor', fontsize=12)



train_df['floor'] = np.log(train_df['floor']+0.1)

nf['floor'] = train_df['floor']

sns.distplot(train_df['floor'], bins=50, kde=False,ax= axarray[1])

plt.xlabel('floor log', fontsize=12)
sns.pairplot(pd.concat([nf,prices],axis=1),size = 2.7)

plt.show()
print (len(train_df[train_df['max_floor']>=0])/float(len(train_df)))

print (train_df[train_df['max_floor']>=0]['max_floor'].describe())
maxff = train_df[train_df['max_floor']>=0][['floor','max_floor']]

maxff.corr()
fl = train_df[train_df['max_floor']>=0]['floor']

mf = train_df[train_df['max_floor']>=0]['max_floor']

X_train, X_test, y_train, y_test = train_test_split(fl, mf, test_size=0.3, random_state=42)



max_floor_pred= DecisionTreeRegressor()

max_floor_pred.fit(X_train[:, None],y_train)

pred = max_floor_pred.predict(X_test[:, None])

print (r2_score(y_test,pred))



flmf = np.array(train_df[['floor','max_floor']])

mf = [int(np.round(max_floor_pred.predict(np.array([el[0]]).reshape(1, -1)))) if math.isnan(el[1]) else el[1] for el in flmf]
nf = pd.concat([nf,pd.DataFrame(mf,columns=["max_floor"])],axis=1)

nf.head()
tf = nf[['full_sq','floor','max_floor']]

sns.pairplot(pd.concat([tf,prices],axis=1),size = 2.5)

plt.show()
print (len(train_df[train_df['life_sq']>=0])/float(len(train_df)))

print (train_df[train_df['life_sq']>=0]['life_sq'].describe())
maxff = train_df[train_df['num_room']>=0][['life_sq','num_room']]

maxff.corr()
nr = train_df[(train_df.num_room>=0) & (train_df.life_sq>=0)]['num_room']

ls = train_df[(train_df.num_room>=0) & (train_df.life_sq>=0)]['life_sq']

X_train, X_test, y_train, y_test = train_test_split(nr, ls, test_size=0.3, random_state=42)



life_sq_pred= DecisionTreeRegressor()

life_sq_pred.fit(X_train[:, None],y_train)

pred = life_sq_pred.predict(X_test[:, None])

print (r2_score(y_test,pred))



nrls = np.array(train_df[['num_room','life_sq']])

lsq = [int(np.round(life_sq_pred.predict(np.array(   [el[0]]   ).reshape(1, -1)))) if math.isnan(el[1]) else el[1] for el in nrls]
nf = pd.concat([nf,pd.DataFrame(lsq,columns=["life_sq"])],axis=1)

lsq = np.log(nf['life_sq']+0.0001)

nf['life_sq'] = lsq

nf.head()


psm = train_df['price_doc']/(train_df['full_sq']+1).astype(float)

psm = psm / np.max(psm)

train_df['psm'] = psm



sa = train_df['sub_area']

sap = train_df[['sub_area','psm']]



area_psm =  sap.groupby('sub_area').mean()



print (pd.DataFrame(area_psm['psm']).head())

nf['sap'] = train_df['sub_area'].apply(lambda x: area_psm['psm'][x])

print (nf['sap'].isnull().sum())



sa = pd.Series(sa)

#print sa.unique()

sad = pd.get_dummies(sa)

sad.shape



pt = train_df['product_type']

pt = pd.Series(pt)

ptd = pd.get_dummies(pt)
import re

years = [int(re.split(r'^([0-9]*)-',ts)[1]) for ts in train_df['timestamp']]

years = pd.Series(years).astype(int)

print (years.value_counts())

print (years.unique())
yp = pd.concat([years,prices],axis=1)

yp.columns = ['years','prices']

grouped = yp.groupby('years')

grouped.mean()
nf = pd.concat([nf,pd.get_dummies(years)],axis=1)
nf = pd.concat([nf,pd.DataFrame(train_df['timestamp'])],axis=1)
tpt = train_df.filter(regex='^(?=.*metro)(?!.*ID_).*|^(?=.*road)(?!.*ID_).*')

tpt = tpt.select_dtypes(include=['number'])

tpt = np.log(tpt)

from sklearn.decomposition import PCA

tf = tpt

print (tf.head())

tf = tf.select_dtypes(include=['number']).dropna(axis=1, how='any')

tf[np.isinf(tf)] = 0

#print(np.isnan(tf).sum())

print(np.isinf(tf).sum())

pca = PCA(n_components=3)

pca.fit(tf)

reduced_data = pca.transform(tf)

tpt_reduced_data = pd.DataFrame(reduced_data,columns=['tpt_pca1','tpt_pca2','tpt_pca3'])

print (np.isnan(tpt_reduced_data).sum())

tmp = pd.concat([tpt_reduced_data,prices],axis=1)

print (tmp.corr())
sns.pairplot(pd.concat([tpt_reduced_data,prices],axis=1),size = 2.5)

plt.show()
tpt = train_df.filter(regex='^(?=.*school)(?!.*ID_).*')

tpt = tpt.select_dtypes(include=['number'])

#tpt = np.log(tpt)

tpt = tpt.drop('preschool_quota',axis=1)

from sklearn.decomposition import PCA

tf = tpt

print (tf.head())

tf = tf.select_dtypes(include=['number']).dropna(axis=1, how='any')

pca = PCA(n_components=3)

pca.fit(tf)

reduced_data = pca.transform(tf)

sch_reduced_data = pd.DataFrame(reduced_data,columns=['sch_pca1','sch_pca2','sch_pca3'])

tmp = pd.concat([sch_reduced_data,prices],axis=1)

print (tmp.corr())
sns.pairplot(pd.concat([sch_reduced_data,prices],axis=1),size = 2.5)

plt.show()
tpt = train_df.filter(regex='^(?=.*green)(?!.*ID_).*')

tpt = tpt.select_dtypes(include=['number'])

from sklearn.decomposition import PCA

tf = tpt

print (tf.head())

tf = tf.select_dtypes(include=['number']).dropna(axis=1, how='any')

tf[np.isinf(tf)] = 0

pca = PCA(n_components=3)

pca.fit(tf)

reduced_data = pca.transform(tf)

green_reduced_data = pd.DataFrame(reduced_data,columns=['gr_pca1','gr_pca2','gr_pca3'])

tmp = pd.concat([tpt_reduced_data,prices],axis=1)

print (tmp.corr())
nnf = pd.concat([nf,ptd],axis=1)

nnf = pd.concat([nnf,tpt_reduced_data],axis=1)

nnf = pd.concat([nnf,sch_reduced_data],axis=1)

nnf = pd.concat([nnf,green_reduced_data],axis=1)

nnf = pd.concat([nnf,train_df['raion_popul']],axis=1)

nnf = nnf.drop('timestamp',axis=1)

nnf['num_room'] = nnf['num_room'].astype('int')

nnf['floor'] = nnf['floor'].astype('int')

nnf['max_floor'] = nnf['max_floor'].astype('int')

nnf['full_sq']=nf['full_sq']/np.max(nf['full_sq'])

nnf['life_sq']=nf['full_sq']/np.max(nf['life_sq'])

nnf.head()
X_train, X_test, y_train, y_test = train_test_split(nnf,prices,test_size=0.2,random_state = 42)
import math

def rmsle(a,p):

    tm =0

    for idx,el in enumerate(a):

        tm = tm+(math.log(p[idx]+1)-math.log(el+1))**2

    tm = tm /len(a)

    return math.sqrt(tm)
from sklearn.metrics import r2_score



def performance_metric(y_true, y_predict):

    score = r2_score(y_true,y_predict)

    return score
from sklearn.metrics import make_scorer

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import SGDRegressor

from sklearn import datasets, linear_model

from sklearn import grid_search

from sklearn.cross_validation import ShuffleSplit



def fit_model(X, y):

    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    params = {'max_depth': (1,20),'n_estimators':(1,50)}

    scoring_fnc = make_scorer(score_func= performance_metric)

    regressor = RandomForestRegressor()

    grid =  grid_search.GridSearchCV(regressor,params,cv=cv_sets,scoring = scoring_fnc);

    grid = grid.fit(X, y)

    return grid.best_estimator_
#model =  regressor = linear_model.LinearRegression().fit(X_train,y_train)

from sklearn import preprocessing

from sklearn.pipeline import Pipeline



#model = Pipeline([("scale", preprocessing.StandardScaler()),

#               ("rf", RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=2))])

#model.fit(X_train,y_train)

model = fit_model(X_train,y_train)
pred = model.predict(X_test)

#pred = pred - pred*0.3
print (model)
from sklearn.metrics import mean_squared_error

#e_y_test = np.exp(y_test)

#e_pred = np.exp(pred)

print (rmsle(y_test,pred))

#print math.sqrt(mean_squared_error(y_test,pred))

#print prices.describe()
importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(nnf.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
nnf.columns[indices-1]
topf = nnf.columns[indices-1][0:10]

top_df = nnf[topf]

top_df.head()


X_train, X_test, y_train, y_test = train_test_split(top_df,prices,test_size=0.2,random_state = 42)
model = fit_model(X_train,y_train)
pred = model.predict(X_test)

print (rmsle(y_test,pred))