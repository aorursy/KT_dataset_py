import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

import numpy as np 

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

from math import cos, sin, asin, sqrt, pi

from tqdm import tqdm 

import time

tqdm.pandas()

!pip install swifter 2>/dev/null 1>/dev/null

import swifter 

import xgboost as xgb



#pandas 

import pandas as pd 



#Sklearn models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import Lasso

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN

from sklearn.ensemble import RandomForestClassifier





#Rapids 

import cudf

from cuml import LogisticRegression as cLogisticRegression

from cuml.neighbors import KNeighborsClassifier as cKNeighborsClassifier

from cuml import SVC as cSVC

from cuml.linear_model import Lasso as cLasso

from cuml.manifold import TSNE as cTSNE

from cuml import DBSCAN as cDBSCAN

from cuml.decomposition import PCA as cPCA

from cuml.ensemble import RandomForestClassifier as cRandomForestClassifier





import warnings 

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/titanic/train.csv')
# Little data preprocessing for the models 

features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked',"Survived"]

x = data[features]

x['Age'] = x['Age'].fillna(x['Age'].median())

x['Embarked']= x['Embarked'].fillna(x['Embarked'].value_counts().index[0])

LE = LabelEncoder()

x['Sex'] = LE.fit_transform(x['Sex'])

x['Embarked'] = LE.fit_transform(x['Embarked'])

for i in range(13):

    x = pd.concat([x,x])
len(x)
x.reset_index(inplace=True)
x_cudf = cudf.from_pandas(x)
def test_function(Fare):

    out=0

    for i in range(10):

        out +=(sin(Fare/2)**2 + cos(Fare) * cos(Fare) * sin(Fare/2)**2)*i

    return out 
start_time = time.time()

x["test"]=x.Fare.progress_apply(test_function)

print("%s seconds " % round((time.time() - start_time),2))
start_time = time.time()

x["test"]=x.Fare.swifter.apply(test_function)

print("%s seconds " % round((time.time() - start_time),2))
def test_function(Fare,test):

    for i,x in enumerate(Fare):

        for j in range(10):

            test[i]  += (sin(x/2)**2 + cos(x) * cos(x) * sin(x/2)**2)*j
start_time = time.time()

x_cudf = x_cudf.apply_rows(test_function,

                   incols=['Fare'],

                   outcols=dict(test=np.float64),

                   kwargs=dict())

print("%s seconds " % round((time.time() - start_time),2))
dtrain = xgb.DMatrix(x.drop(["Survived"],axis=1),label=x["Survived"])
num_round = 100

print("Training with CPU ...")

param = {}

param['tree_method'] = 'hist'

tmp = time.time()

xgb.train(param, dtrain, num_round)

cpu_time = time.time() - tmp

print("CPU Training Time: %s seconds" % (str(cpu_time)))
print("Training with Single GPU ...")

param = {}

param['tree_method'] = 'gpu_hist'

tmp = time.time()



xgb.train(param, dtrain, num_round)

gpu_time = time.time() - tmp

print("GPU Training Time: %s seconds" % (str(gpu_time)))
dtrain = xgb.DMatrix(x_cudf.drop(["Survived"],axis=1),x_cudf["Survived"])
print("Training with Single GPU ...")

param = {}

param['tree_method'] = 'gpu_hist'

tmp = time.time()



xgb.train(param, dtrain, num_round)

gpu_time = time.time() - tmp

print("GPU Training Time: %s seconds" % (str(gpu_time)))
tmp = time.time()

LogisticRegression().fit(X=x.drop(["Survived"],axis=1),y=x["Survived"])

cpu_time = time.time() - tmp

print("LogisticRegression Time: %s seconds" % (str(round(cpu_time,3))))



tmp = time.time()

KNeighborsClassifier().fit(X=x.drop(["Survived"],axis=1)[:1000000],y=x["Survived"][:1000000])

cpu_time = time.time() - tmp

print("KNeighbors Time: %s seconds" % (str(round(cpu_time,3))))





tmp = time.time()

SVC().fit(X=x.drop(["Survived"],axis=1)[:50000],y=x["Survived"][:50000])

cpu_time = time.time() - tmp

print("SVM Training Time: %s seconds" % (str(round(cpu_time,3))))





tmp = time.time()

Lasso().fit(X=x.drop(["Survived"],axis=1),y=x["Survived"])

cpu_time = time.time() - tmp

print("Lasso Training Time: %s seconds" % (str(round(cpu_time,3))))





tmp = time.time()

TSNE(n_components=2).fit(x.drop(["Survived"],axis=1)[:10000])

cpu_time = time.time() - tmp

print("TSNE Training Time: %s seconds" % (str(round(cpu_time,3))))





tmp = time.time()

DBSCAN(eps=0.6, min_samples=2).fit(x.drop(["Survived"],axis=1)[:100000])

cpu_time = time.time() - tmp

print("DBScan Training Time: %s seconds" % (str(round(cpu_time,3))))





tmp = time.time()

PCA(n_components=2).fit(x.drop(["Survived"],axis=1)[:100000])

cpu_time = time.time() - tmp

print("PCA Training Time: %s seconds" % (str(round(cpu_time,3))))
import gc

gc.collect()
x_cudf["Survived"] = x_cudf["Survived"].astype(np.float64)



tmp = time.time()

cLogisticRegression().fit(X=x_cudf.drop(["Survived"],axis=1),y=x_cudf["Survived"])

gpu_time = time.time() - tmp

print("LogisticRegression Time: %s seconds" % (str(round(gpu_time,3))))







tmp = time.time()

cKNeighborsClassifier().fit(X=x_cudf.drop(["Survived"],axis=1)[:1000000],y=x_cudf["Survived"][:1000000])

gpu_time = time.time() - tmp

print("KNeighbors Time: %s seconds" % (str(round(gpu_time,3))))







tmp = time.time()

cSVC().fit(X=x_cudf.drop(["Survived"],axis=1)[:50000],y=x_cudf["Survived"][:50000])

gpu_time = time.time() - tmp

print("SVM Training Time: %s seconds" % (str(round(gpu_time,3))))





tmp = time.time()

cLasso().fit(X=x_cudf.drop(["Survived"],axis=1),y=x_cudf["Survived"])

gpu_time = time.time() - tmp

print("Lasso Training Time: %s seconds" % (str(round(gpu_time,3))))

gc.collect()



tmp = time.time()

cTSNE(n_components=2).fit(X=x_cudf.drop(["Survived"],axis=1)[:10000])

gpu_time = time.time() - tmp

print("TSNE Training Time: %s seconds" % (str(round(gpu_time,3))))







tmp = time.time()

cDBSCAN(eps=0.6, min_samples=2).fit(X=x_cudf.drop(["Survived"],axis=1)[:100000])

gpu_time = time.time() - tmp

print("DbScan Training Time: %s seconds" % (str(round(gpu_time,3))))





tmp = time.time()

cPCA(n_components=2).fit(X=x_cudf.drop(["Survived"],axis=1)[:100000])

gpu_time = time.time() - tmp

print("PCA Training Time: %s seconds" % (str(round(gpu_time,3))))
tmp = time.time()

RandomForestClassifier(n_estimators = 150, max_depth=13).fit(X=x.drop(["Survived"],axis=1)[:1000000],y=x["Survived"][:1000000])

cpu_time = time.time() - tmp

print("Random Forest Time: %s seconds" % (str(round(cpu_time,3))))
tmp = time.time()

model = cRandomForestClassifier(n_estimators = 150, max_depth=13)

model.fit(X=x_cudf.drop(["Survived"],axis=1)[:1000000],y=x_cudf["Survived"].astype("int32")[:1000000])

gpu_time = time.time() - tmp

print("Random Forest Training Time: %s seconds" % (str(round(gpu_time,3))))