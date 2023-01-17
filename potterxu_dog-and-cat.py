# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
submisson=pd.read_csv("../input/sample_submission.csv")

submisson.head()
train.info()
##Breed和Color类别比较多。可以考虑hash,但是各颜色不是完全无关系的。比如同是黄，就分很多种

###Name字段特征化（纬度比较多）

train.Color.unique()
train.groupby("Breed").AnimalID.count()

train.head()
train_x=train[["Name","DateTime","AnimalType","SexuponOutcome","AgeuponOutcome","Breed","Color"]]

test_x=test[["Name","DateTime","AnimalType","SexuponOutcome","AgeuponOutcome","Breed","Color"]]

train_x.shape,test_x.shape
data=pd.concat([train_x,test_x],axis=0)

data.DateTime=pd.to_datetime(data.DateTime)

data.head()
data.info()


from sklearn.feature_extraction import FeatureHasher

h = FeatureHasher(n_features=100,input_type="string")

data["Name"]=data["Name"].fillna("None")

#h.transform(data["Name"])



#pd.DataFrame(h.transform(data["Name"]).toarray(),columns=["name"+str(i) for i in range(100)])

for i in data["Name"].value_counts().index[0:400]:

    print(i)
import re 

fre_name=data["Name"].value_counts().index[0:400]

def etl(data):

    

    data["year"]=data.DateTime.dt.year

    data["month"]=data.DateTime.dt.month

    data["day"]=data.DateTime.dt.day

    data["dayofweek"]=data.DateTime.dt.dayofweek

    data["hour"]=data.DateTime.dt.hour

    

    data["AgeuponOutcome"]= data.AgeuponOutcome.fillna("-1")

    data["SexuponOutcome"]=data.SexuponOutcome.fillna("Unknown")

    data["color"]=data.Color.apply(lambda e: e.split("/")[0].split(" ")[0])

    data["color1"]=data.Color.apply(lambda e: re.split("/",e)[1].split(" ")[0]

                                    if len(re.split("/",e))>1 else "None")

    data["breed"]=data.Breed.apply(lambda e : e.split("/")[0])

    data["breed1"]=data.Breed.apply(lambda e: re.split("/",e)[1]

                                    if len(re.split("/",e))>1 else "None")

    data["is_mix"]=data.Breed.apply(lambda e : '1' if "Mix" in e else '0' )

    cols=["year","month","day","dayofweek","hour","AgeuponOutcome","SexuponOutcome","color","color1",

                 "breed","breed1","is_mix"]

    d=pd.get_dummies(data.AnimalType,prefix="AnimalType")

    #print(d.shape)

    for col in cols:

       

        d=pd.concat([d,pd.get_dummies(data[col],prefix=col)],axis=1)

    for name in fre_name:

        d[name]=data.Name.apply(lambda e :1 if e ==name else 0)

    d.drop(["Name"])

    return d 

d=etl(data)

#d.shape
data.head()
train1=d.iloc[0:train_x.shape[0],:]

test1=d.iloc[train_x.shape[0]:,:]

train1.shape
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

enc.fit(train.OutcomeType)

train_y=enc.transform(train.OutcomeType)

train_y
###添加几个变量后准确率提高，LR的效果为啥一直很差（70多的准确率）,

from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

clf = linear_model.LogisticRegression(C=0.1,max_iter=2000,random_state=234)

clf = RandomForestClassifier(n_estimators=1000,max_depth=50, min_samples_split=1, random_state=234)

#clf = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=250,objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)

#clf = GradientBoostingClassifier(n_estimators=800, learning_rate=1.0)

clf.fit(train1, train_y)

y_pred = clf.predict(train1)

print("Number of mislabeled points out of a total %d points : %d"  % (train1.shape[0],(train_y != y_pred).sum()))
###xgboost的树deep越大精准度越高，但是训练时间一般都比较长。是否有简单的办法减少时间

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

T_train_xgb = xgb.DMatrix(train1, label=train_y)

watchlist = [(T_train_xgb, 'train'),(T_train_xgb, 'eval')]

params = {"objective": "multi:softprob",

          "booster" : "gbtree",

          "num_class" : 5,

          "eta": 0.3,          

          "subsample": 0.9,

          "colsample_bytree": 0.7,

          "seed": 1301,

          "max_depth":80,

          "eval_metric":"mlogloss"

          }

gbm = xgb.train(dtrain=T_train_xgb,params=params, evals=watchlist,num_boost_round = 35,verbose_eval=True)

#y_pred = gbm.predict(xgb.DMatrix(train1))

#print("Number of mislabeled points out of a total %d points : %d"  % (train1.shape[0],(train_y != y_pred).sum()))
y_pred
#importance=pd.DataFrame({"f":train1.columns,"import":clf.feature_importances_})

#importance.sort_values("import",0,False)
id=test.ID

#pred = clf.predict_proba(test1)

pred = gbm.predict(xgb.DMatrix(test1))

id.shape,pred.shape

submission=pd.DataFrame({"ID":id})

submission['Adoption']=pred[:,0]

submission['Died']=pred[:,1]

submission['Euthanasia']=pred[:,2]

submission['Return_to_owner']=pred[:,3]

submission['Transfer']=pred[:,4]



id=test.ID

pred = clf.predict_proba(test1)

#pred = gbm.predict(xgb.DMatrix(test1))

id.shape,pred.shape

submission=pd.DataFrame({"ID":id})

submission['Adoption']=pred[:,0]

submission['Died']=pred[:,1]

submission['Euthanasia']=pred[:,2]

submission['Return_to_owner']=pred[:,3]

submission['Transfer']=pred[:,4]

submission.to_csv("submission.csv",index=False)
submission.to_csv("submission.csv",index=False)

submission.head()
submission.describe()