import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling

from sklearn.metrics import accuracy_score
data_train = pd.read_csv('train.csv')

data_test  = pd.read_csv('test.csv')

data_test
# data_train=np.array(data_train)

# data_train.shape

# data_train
# data_notv=data_train where TVConnection=="No"
# for i in data_train:

#     #print (i.shape)

#     if (i[5]=="No"):

#         print (i[20])
print(data_train_x.dtypes)
data_train_x=data_train.drop(["custId","Satisfied"],axis=1)

data_train_y=data_train['Satisfied']

data_train_x
# no_tv=[]

# for i in range(data_train_x.size):

#     if (data_train_x[i]["TVConnection"]=="No"):

#         no_tv.append(data_train[i]["Satisfied"])

# no_tv.value_counts()
corr = data_train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
numerical_val=["tenure","MonthlyCharges","TotalCharges"]
from sklearn.cluster import KMeans
X_encoded = pd.get_dummies(data_train_x.drop(numerical_val,axis=1))

X_encoded.head()
X_num=data_train_x[numerical_val]

X_num.head()
frames=[X_encoded,X_num]

X = pd.concat(frames,axis=1, join='inner')

X.head()
print(X.dtypes)
"Subscription_Monthly","Subscription_Biannually","Subscription_Annually"
notv=["TotalCharges","Channel6_No tv connection","Channel5_No tv connection","Channel4_No tv connection","Channel3_No tv connection","Channel2_No tv connection","Channel1_No tv connection"]
kmeans = KMeans(n_clusters=2, init="random", random_state=0,n_init=15).fit(X.drop(notv,axis=1))

pred_y=kmeans.labels_
pred_y
accuracy_score(data_train_y,pred_y)
data_test_x=data_test.drop(["custId"],axis=1)

data_test_x
X_test_encoded = pd.get_dummies(data_test_x.drop(numerical_val,axis=1))

X_test_encoded.head()
X_test_num=data_test_x[numerical_val]

X_test_num.head()
frames_test=[X_test_encoded,X_test_num]

X_test = pd.concat(frames_test,axis=1, join='inner')

X_test.head()
y_pred = kmeans.fit_predict(X_test.drop(notv,axis=1))

y_pred
ans=[]

for i in y_pred:

    if (i==1):

        ans.append(0)

    else:

        ans.append(1)

ans
submission = pd.DataFrame({'custId':data_test['custId'],'Satisfied':ans})

submission.to_csv('initrand.csv',index=False)
data_train = pd.read_csv('train.csv')

data_test  = pd.read_csv('test.csv')

data_test.shape
train=data_train[data_train.TVConnection!="No"]

data_train_x=train.drop(["custId","Satisfied"],axis=1)

data_train_y=train['Satisfied']

data_train_x
data_train_x=data_train_x[data_train_x.TVConnection!="No"]
data_train_x
X_encoded = pd.get_dummies(data_train_x.drop(numerical_val,axis=1))

X_encoded.head()
X_num=data_train_x[numerical_val]

X_num.head()
frames=[X_encoded,X_num]

X = pd.concat(frames,axis=1, join='inner')

X.head()
kmeans = KMeans(n_clusters=2, init="random", random_state=0,n_init=15).fit(X.drop("TotalCharges",axis=1))

pred_y=kmeans.labels_
accuracy_score(data_train_y,pred_y)
test=data_test[data_test.TVConnection!="No"]

data_test_x=test.drop(["custId"],axis=1)

# data_train_y=train['Satisfied']

data_test_x
X_test_encoded = pd.get_dummies(data_test_x.drop(numerical_val,axis=1))

X_test_encoded.head()
X_test_num=data_test_x[numerical_val]

X_test_num
frames_test=[X_test_encoded,X_test_num]

X_test = pd.concat(frames_test,axis=1, join='inner')

X_test
y_pred = kmeans.fit_predict(X_test.drop("TotalCharges",axis=1))

y_pred.shape
data_test
i=1656

print(test.index[test["custId"] == data_test.loc[i,"custId"]])
# ans = []

# for i in range(len(data_test)):

#     if data_test.loc[i,"TVConnection"]=="No":

#         ans=[{"custId":data_test.loc[i,"custId"],"Satisfied":1}]

#     else:

#         ans=[{"custId":data_test.loc[i,"custId"],"Satisfied":y_pred[test.index[test.custId == data_test.loc[i,"custId"]]]}]
# submission = pd.DataFrame({'custId':data_test['custId'],'Satisfied':})
# submission.to_csv('initrand.csv',index=False)