from warnings import filterwarnings

filterwarnings('ignore')
import seaborn as sns

diamonds = sns.load_dataset('diamonds') 

df = diamonds.copy()

df = df.select_dtypes(include = ['float64', 'int64']) 

df.head()
df_table = df["table"].copy()
sns.boxplot(x = df_table)
Q1 = df_table.quantile(0.25)

Q3 = df_table.quantile(0.75)

IQR = Q3 - Q1



alt_sinir = Q1- 1.5*IQR

alt_sinir



ust_sinir = Q3 + 1.5*IQR

ust_sinir
(df_table < (alt_sinir)) | (df_table > (ust_sinir))
df_table < (alt_sinir)
aykiri_tf = df_table < (alt_sinir)
aykiri_tf[0:10]
aykirilar = df_table[aykiri_tf]

aykirilar.index
import pandas as pd
df_table.head()

type(df_table)

df_table.shape

df_table=df_table.to_frame()

temiz_df_table = df_table[~((df_table < (alt_sinir)) | (df_table > (ust_sinir))).any(axis = 1)]

temiz_df_table.shape
df_table = df["table"].copy()
sns.boxplot(x = df_table)
df_table[aykiri_tf]
df_table.mean()
df_table[aykiri_tf] = df_table.mean()
df_table[aykiri_tf]
aykiri_tf = (df_table < (alt_sinir)) | (df_table > (ust_sinir))
df_table[aykiri_tf].head()
df_table.describe()
df_table[aykiri_tf] = df_table.mean()
df_table.describe()
df_table = df["table"].copy()
aykiri_tf = df_table < (alt_sinir)
df_table[aykiri_tf]
df_table[aykiri_tf] = alt_sinir 
df_table[aykiri_tf]
from IPython.display import Image

Image(filename =  "lof_intuition.png" , width=400, height=400)
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor



np.random.seed(42)

X_inliers = np.random.normal(70, 3, (100, 2))



X_inliers = np.r_[X_inliers + 10, X_inliers - 10] 



print(X_inliers.shape)

print(X_inliers[:3,:2])
X_outliers = np.random.uniform(low=15, high=130, size=(20, 2))
X_outliers
X = np.r_[X_inliers, X_outliers]
X[0:3,:]
LOF = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
LOF.fit_predict(X)
X_score = LOF.negative_outlier_factor_
X_score[0:3]
X_score.mean()
X_score.std()
np.sort(X_score)[0:10]
plt.hist(X_score, bins = "auto", density = True)

plt.show
plt.scatter(X[:,0], X[:,1], color = "k", s = 3, label = "Gözlem Birimleri");
radius = radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())
plt.scatter(X[:,0], X[:,1], color = "k", s = 3, label = "Gözlem Birimleri");



plt.scatter(X[:, 0], X[:, 1], s = 1000 * radius, edgecolors='r', 

            facecolors='none',label='LOF Skorları')



plt.xlim((10,100))

plt.ylim((10,100))



legend = plt.legend(loc = "upper left")



legend.legendHandles[0]._sizes = [10]

legend.legendHandles[1]._sizes = [30]
X[0:3]
np.sort(X_score)[0:9]
esik_deger = np.sort(X_score)[9]

esik_deger
(X_score > esik_deger)[200:220]
tf_vektor = (X_score > esik_deger)
X[X_score < esik_deger]
X[~tf_vektor]
X[X_score < esik_deger]
X[200:220]
df = X[X_score > esik_deger]
df[0:10]
df_X = X.copy()
np.mean(df_X[0])

np.mean(df_X[1])
df_X[~tf_vektor]
aykirilar = df_X[~tf_vektor]
aykirilar[:,:1]
aykirilar[:,:1] = np.mean(df_X[0])
aykirilar[:,1:2] = np.mean(df_X[1])
aykirilar
df_X[~tf_vektor] = aykirilar
df_X[~tf_vektor]
df_X = X.copy()
df_X[~tf_vektor]
df_X[X_score == esik_deger]
df_X[~tf_vektor] = df_X[X_score == esik_deger]
df_X[~tf_vektor]
import numpy as np

import pandas as pd



V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])



df = pd.DataFrame(

        {"V1" : V1,

         "V2" : V2,

         "V3" : V3}        

)



df
df.isnull().sum()
df.dropna()
df
dff = df.dropna()
dff.isnull().sum()
df["V1"].mean()
df["V1"].fillna(df["V1"].mean())
df["V1"].fillna(0)
df.apply(lambda x: x.fillna(x.mean()), axis = 0)
import numpy as np

import pandas as pd



V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])



df = pd.DataFrame(

        {"V1" : V1,

         "V2" : V2,

         "V3" : V3}        

)



df
df.shape
df.describe()
df.dtypes
df.notnull().sum()
df.isnull().sum()
df.isnull().sum().sum()
df.isnull()
df[df.isnull().any(axis = 1)]
df[df.notnull().all(axis = 1)]
df[df["V1"].notnull() & df["V2"].notnull() & df["V3"].notnull()]
!pip install missingno
import missingno as msno
df.head()
msno.bar(df);
df.isnull().sum()
df
import seaborn as sns

sns.heatmap(df.isnull(), cbar = False);
msno.matrix(df)
df = sns.load_dataset("planets").copy()

df.head()
import seaborn as sns

sns.heatmap(df.isnull(), cbar = False);
msno.matrix(df);
msno.heatmap(df);
null_pattern = (np.random.random(1000).reshape((50, 20)) > 0.5).astype(bool)



null_pattern = pd.DataFrame(null_pattern).replace({False: None})



msno.matrix(null_pattern.set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M')) , freq='BQ');
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])



df = pd.DataFrame(

        {"V1" : V1,

         "V2" : V2,

         "V3" : V3}        

)



df
df.dropna()
df.dropna(how = "all")
df.dropna(axis = 1)
df["V1"][[3,6]] = 99
df.dropna(axis = 1)
df.dropna(axis = 1, how = "all")
df["sil_beni"] = np.nan
df
df.dropna(axis = 1, how = "all", inplace = True)
df
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])



df = pd.DataFrame(

        {"V1" : V1,

         "V2" : V2,

         "V3" : V3}        

)



df
df["V1"].fillna(0)
df["V1"].fillna(df["V1"].mean())
df.apply(lambda x: x.fillna(x.mean()), axis = 0 )
df.fillna(df.mean()[:])
df.fillna(df.mean()["V1":"V2"])

df.fillna(df.median()["V3"])
df.where(pd.notna(df), df.mean(), axis = "columns")
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])



df = pd.DataFrame(

        {"maas" : V1,

         "V2" : V2,

         "V3" : V3,

        "departman" : V4}        

)



df
df.groupby("departman")["maas"].mean()
df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

V4 = np.array(["IT",np.NaN,"IK","IK","IK","IK","IK","IT","IT"])



df = pd.DataFrame(

        {"maas" : V1,

         "V2" : V2,

         "V3" : V3,

        "departman" : V4}        

)



df
df.isnull()
df.groupby("departman")["departman"].count()
df.departman.loc[df.departman == "nan"] = "IK"
df
df.departman[0] = df.V3[0]
df
df.groupby("departman")["departman"].count()
df.departman.fillna(df["departman"].mode())
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])

V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])

V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])



df = pd.DataFrame(

        {"maas" : V1,

         "V2" : V2,

         "V3" : V3,

        "departman" : V4}        

)



df
df["maas"].interpolate()
df["maas"].fillna(method = "bfill")
import seaborn as sns

df = sns.load_dataset('planets').copy()

df = df.select_dtypes(include = ['float64', 'int64'])

print(df.isnull().sum())

msno.matrix(df);
#!pip install fancyimpute
from fancyimpute import KNN
import pandas as pd
var_names = list(df)
knn_imp = KNN(k = 5).fit_transform(df);
knn_imp[0:1]
dff = pd.DataFrame(knn_imp)
dff.head()
dff.columns = var_names
dff.head()
dff.isnull().sum()
!pip install ycimpute
from ycimpute.imputer import knnimput
var_names = list(df)
n_df = np.array(df)
n_df.shape
dff = knnimput.KNN(k=4).complete(n_df)
dff = pd.DataFrame(dff, columns = var_names)
dff.head()
dff.isnull().sum()
import seaborn as sns

df = sns.load_dataset('planets').copy()

df = df.select_dtypes(include = ['float64', 'int64'])

print(df.isnull().sum())

msno.matrix(df);
from ycimpute.imputer import iterforest
var_names = list(df)
n_df = np.array(df)
dff = iterforest.IterImput().complete(n_df)
dff = pd.DataFrame(dff, columns = var_names)
dff.isnull().sum()
df.head()
from ycimpute.imputer import EM
var_names = list(df)
n_df = np.array(df)
dff = EM().complete(n_df)
dff = pd.DataFrame(dff, columns = var_names)
dff.isnull().sum()
import numpy as np

import pandas as pd



V1 = np.array([1,3,6,5,7])

V2 = np.array([7,7,5,8,12])

V3 = np.array([6,12,5,6,14])



df = pd.DataFrame(

        {"V1" : V1,

         "V2" : V2,

         "V3" : V3}        

)







df = df.astype(float)

df
from sklearn import preprocessing
preprocessing.scale(df)
preprocessing.normalize(df)
scaler = preprocessing.MinMaxScaler(feature_range = (10,20))
scaler.fit_transform(df)
binarizer = preprocessing.Binarizer(threshold = 5).fit(df)
binarizer.transform(df)
import seaborn as sns

tips = sns.load_dataset('tips')

df = tips.copy()

df_l = df.copy()
df_l.head()
df_l["yeni_sex"] = df_l["sex"].cat.codes
df_l.head()
lbe = preprocessing.LabelEncoder()
df_l["daha_yeni_sex"] = lbe.fit_transform(df_l["sex"])
df_l.head()
df.head()
df_l.head()
df_l["yen_gun"] = np.where(df_l["day"].str.contains("Sun"),1,0)
df_l.head(20)
lbe = preprocessing.LabelEncoder()
df_l["daha_yeni_gun"] = lbe.fit_transform(df_l["day"])
df_l
df_one_hot = df.copy()
pd.get_dummies(df_one_hot, columns = ["sex"], prefix = ["sex"]).head()
pd.get_dummies(df_one_hot, columns = ["day"], prefix = ["day"]).head()
df.head()
dff = df.select_dtypes(include = ["float64", "int64"])
est = preprocessing.KBinsDiscretizer(n_bins = [3,2,2], encode = "ordinal", strategy = "quantile").fit(dff)
est.transform(dff)[0:10]
df.head()
df["yeni_degisken"]  = df.index
df["yeni_degisken"] = df["yeni_degisken"] + 10
df.head()
df.index = df["yeni_degisken"]
df.index