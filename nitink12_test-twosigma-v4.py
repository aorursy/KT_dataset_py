import json  

import zipfile  

import numpy as np

import pandas as pd





path= "../input/"

d = None  

data = None  



with open(path+"train.json") as f:

    #data = f.read()

    d=json.load(f)

    f.close()



type(d)

##Test



d_test = None

#data_test=None

with open(path+"test.json") as f1:

    #data = f.read()

    d_test=json.load(f1)

    f1.close()



type(d_test)

df_test=pd.DataFrame(d_test)

len(df_test)
df=pd.DataFrame(d)

len(df)

df.head(10)
len(df[df['building_id']=="0"])
len(df[df['description']==""])
df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

df.head(10)
len(df[df['features']==""])
df1=df[df['features']==""]



df1.groupby('interest_level').size()
df2=df[df['description']==""]



df2.groupby('interest_level').size()
df2=df[df['description']==""]



df2.groupby('interest_level').size()
df3= df[df['building_id']=="0"]

df3.groupby('interest_level').size()
df['created'] = df['created'].astype('datetime64[ns]')
a=max(df['created'])-df['created']
df['daysFromCreated']=a/np.timedelta64(1, 'D')
df.head()
xl=df.longitude

yl=df.latitude

xl=(xl-np.mean(xl))/np.std(xl)

yl=(yl-np.mean(yl))/np.std(yl)



df.longitude=xl

df.latitude=yl
import matplotlib.pyplot as plt
plt.plot(xl,yl,'ro')

plt.show()
from sklearn.cluster import KMeans
a= np.array(xl)

b=np.array(yl)
c=np.column_stack((a,b))
kmeans = KMeans(n_clusters=20)

kmeans.fit_predict(c)
centers = kmeans.cluster_centers_
prediction = kmeans.predict(c)
prediction
plt.scatter(centers[:, 0], centers[:, 1], marker='x')



plt.show()
df["KMeans_Clusters"]= prediction
df.head(50)
df.describe()
df.info()
df.groupby("KMeans_Clusters").count()
c=[1,2,3,4,5,8,13,14,16,17,18,19]
df['KMeans_Clusters'].replace(1, 10,inplace=True)

df['KMeans_Clusters'].replace(2, 10,inplace=True)

df['KMeans_Clusters'].replace(3, 10,inplace=True)

df['KMeans_Clusters'].replace(4, 10,inplace=True)

df['KMeans_Clusters'].replace(9, 10,inplace=True)

df['KMeans_Clusters'].replace(8, 10,inplace=True)

df['KMeans_Clusters'].replace(13, 10,inplace=True)

df['KMeans_Clusters'].replace(14, 10,inplace=True)

df['KMeans_Clusters'].replace(11, 10,inplace=True)

df['KMeans_Clusters'].replace(15, 10,inplace=True)

##Test



df_test['created'] = df_test['created'].astype('datetime64[ns]')

#a=max(df_test['created'])-df_test['created']

df_test['daysFromCreated']=(max(df_test['created'])-df_test['created'])/np.timedelta64(1, 'D')

xl_test=df_test.longitude

yl_test=df_test.latitude

xl_test=(xl_test-np.mean(xl))/np.std(xl)

yl_test=(yl_test-np.mean(yl))/np.std(yl)

a_test= np.array(xl_test)

b_test=np.array(yl_test)

c_test=np.column_stack((a_test,b_test))

kmeans.fit_predict(c_test)

prediction_test = kmeans.predict(c_test)

df_test.longitude=xl_test

df_test.latitude=yl_test



df_test["KMeans_Clusters"]= prediction_test



df_test['KMeans_Clusters'].replace(1, 10,inplace=True)

df_test['KMeans_Clusters'].replace(2, 10,inplace=True)

df_test['KMeans_Clusters'].replace(3, 10,inplace=True)

df_test['KMeans_Clusters'].replace(4, 10,inplace=True)

df_test['KMeans_Clusters'].replace(9, 10,inplace=True)

df_test['KMeans_Clusters'].replace(8, 10,inplace=True)

df_test['KMeans_Clusters'].replace(13, 10,inplace=True)

df_test['KMeans_Clusters'].replace(14, 10,inplace=True)

df_test['KMeans_Clusters'].replace(15, 10,inplace=True)

df_test['KMeans_Clusters'].replace(11, 10,inplace=True)

#df_test['KMeans_Clusters'].replace(18, 10,inplace=True)

#df_test['KMeans_Clusters'].replace(19, 10,inplace=True)

df_test['KMeans_Clusters']=df_test['KMeans_Clusters'].astype(object)
df['KMeans_Clusters']=df['KMeans_Clusters'].astype(object)
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer



df['features'] = df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
tfidf = CountVectorizer(stop_words='english', max_features=200)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .80
df.head()
train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of observations in the training data:', len(train))

print('Number of observations in the test data:',len(test))
x=[0,1,8,10,13,15,16]

features = df.columns[x]



features
y = pd.factorize(train['interest_level'])[0]

y
from sklearn.ensemble import RandomForestClassifier
train.shape
from scipy import sparse

tr_sparse = tfidf.fit_transform(train["features"])

te_sparse = tfidf.fit_transform(test["features"])
tr_sparse.shape
categorical = ["KMeans_Clusters"]

for f in categorical:

        if train[f].dtype=='object':

            lbl = LabelEncoder()

            lbl.fit(list(train[f].values) + list(test[f].values))

            train[f] = lbl.transform(list(train[f].values))

            test[f] = lbl.transform(list(test[f].values))

#            features_to_use.append(f)
train_x = sparse.hstack([train[features], tr_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train['interest_level'].apply(lambda x: target_num_map[x]))
#preds, model = runXGB(train_X, train_y, test_X, num_rounds=2000)

param = {}

param['objective'] = 'multi:softprob'

param['eta'] = 0.02

param['max_depth'] = 8

param['silent'] = 1

param['num_class'] = 3

param['eval_metric'] = "mlogloss"

param['min_child_weight'] = 1

param['subsample'] = 0.7

param['colsample_bytree'] = 0.7

param['seed'] = 321

num_rounds = 2000



plst = list(param.items())

xgtrain = xgb.DMatrix(train_x, label=train_y)
test_x = sparse.hstack([test[features], te_sparse]).tocsr()

test_y = np.array(test['interest_level'].apply(lambda x: target_num_map[x]))
#xgtest = xgb.DMatrix(test_x, label=test_y)

#watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

#model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
xgtest = xgb.DMatrix(test_x, label=test_y)

watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
#xgtest = xgb.DMatrix(test_X)

#model = xgb.train(plst, xgtrain, num_rounds)
pred_test_y = model.predict(xgtest)
clf = RandomForestClassifier(n_estimators=450,max_depth=35)



# Train the classifier to take the training features and learn how they relate

# to the training y (the species)

clf.fit(train_x, train['interest_level'])
preds=clf.predict(test_x)
clf.predict_proba(test_x)[0:10]
preds[0:5]
test['interest_level'][0:5]
pd.crosstab(test['interest_level'], preds, rownames=['Actual interest_level'], colnames=['Predicted interest_level'])
import sklearn
sklearn.metrics.log_loss(test['interest_level'], clf.predict_proba(test_x))
##Test



pred_actualTest = clf.predict_proba(df_test[features])

pred_actualTest
your_permutation = [0,2,1]

i = np.argsort(your_permutation)

i

pred_actualTest1=pred_actualTest[:,i]
out_df = pd.DataFrame(pred_actualTest1)

out_df.columns = ["high", "medium", "low"]

out_df["listing_id"] = df_test.listing_id.values
out_df.head()
out_df.to_csv("first1.csv", index=False)
print(check_output(["ls"]).decode("utf8"))
out_df