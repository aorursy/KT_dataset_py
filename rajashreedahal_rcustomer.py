# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import silhouette_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/sa-customer-segmentation/flight_train.csv')
test_data=pd.read_csv('/kaggle/input/sa-customer-segmentation/flight_test.csv')
sample_data=pd.read_csv('/kaggle/input/sa-customer-segmentation/sample.csv')
train_data.head()
train_data.info()
train_data.isnull().sum()
plt.subplots(figsize=(12,12))
sns.heatmap(train_data.corr(),annot=True)
train_data=train_data.append(test_data)
train_data.head()
train_data.info()
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
imputer=SimpleImputer(missing_values=np.nan,strategy='median')
train_data=train_data[train_data['WORK_CITY'].notnull() & train_data['WORK_PROVINCE']]
age=imputer.fit(train_data[['AGE']])
train_data['AGE']=imputer.transform(train_data[['AGE']])
test_data['AGE']=imputer.transform(test_data[['AGE']])
train_data.isnull().sum()
mis_columns=['SUM_YR_1','SUM_YR_2']
def linear_regression_imputer(df,column):
    from sklearn.preprocessing import StandardScaler
    scalar=StandardScaler()
    df['BP_SUM']=scalar.fit_transform(df[['BP_SUM']])
    parameter='BP_SUM'
    lr.fit(df[df[column].notnull()][[parameter]],df[df[column].notnull()][[column]])
    return(lr.predict(df[df[column].isnull()][[parameter]]))
        
df=train_data.copy()
for column in mis_columns:
    lr1=(linear_regression_imputer(df,column))
    l=len(lr1)
    train_data.loc[train_data[column].isnull(),column]=lr1.reshape((l,1))
    
train_data.isnull().sum()
train_data.nunique().sort_values(ascending=False )
train_data['GENDER'].unique()
print(train_data['GENDER'].value_counts())
f,ax=plt.subplots(1,1)
f.set_size_inches(5,3)
sns.countplot(train_data["GENDER"], ax=ax)
train_data["GENDER"].fillna("Male",inplace=True)
train_data.info()
train_data['WORK_COUNTRY'].unique()
customer_countrytrain=train_data[['WORK_COUNTRY','MEMBER_NO']]
customer_countrytrain.groupby(['WORK_COUNTRY']).agg('count').reset_index().sort_values('MEMBER_NO', ascending=False)

train_data.info()
train_data.WORK_PROVINCE.nunique()
def unique_counts(train_data):
    for i in train_data.columns:
        count = train_data[i].nunique()
        print(i, ": ", count)
unique_counts(train_data)
train_data.info()
train_data.describe()
age0_30=train_data.AGE[(train_data.AGE>=0) & (train_data.AGE<30)]
age30_50=train_data.AGE[(train_data.AGE>=30) & (train_data.AGE<50)]
age50_70=train_data.AGE[(train_data.AGE>=50) & (train_data.AGE<70)]
age70_100=train_data.AGE[(train_data.AGE>=70) & (train_data.AGE<100)]

y=[len(age0_30.values),len(age30_50.values),len(age50_70.values),len(age70_100.values)]
x=['0-30','30-50','50-70','70-100']
sns.barplot(x=x,y=y)
plt.xlabel('AGE BAND')
plt.ylabel('count')
train_data['FIRST_FLIGHT_DATE']=pd.to_datetime(train_data['FIRST_FLIGHT_DATE'],errors='coerce')
train_data['LAST_FLIGHT_DATE']=pd.to_datetime(train_data['LAST_FLIGHT_DATE'],errors='coerce')
train_data['LOAD_TIME']=pd.to_datetime(train_data['LOAD_TIME'],errors='coerce')
train_data['FFP_DATE']=pd.to_datetime(train_data['FFP_DATE'],errors='coerce')

train_data=train_data.loc[train_data['LAST_FLIGHT_DATE'].notnull()]
train_data['Recurency']=(train_data['LOAD_TIME']-train_data['FIRST_FLIGHT_DATE']).dt.days
train_data['Days_old']=(train_data['LOAD_TIME']-train_data['LAST_FLIGHT_DATE']).dt.days
train_data['FFP_days']=(train_data['LOAD_TIME']-train_data['FFP_DATE']).dt.days
train_data.info()
train_data.isnull().sum()
train_data.count()
train_data.drop(columns=['FFP_DATE','WORK_COUNTRY','FIRST_FLIGHT_DATE','WORK_CITY','LOAD_TIME','LAST_FLIGHT_DATE'],inplace=True)

train_data['GENDER'].replace(['Male','Female'],[1,0],inplace=True)

train_data.head()

train_data=train_data.reset_index(drop=True)

train_data.tail()
train_data=pd.get_dummies(train_data,columns=['FFP_TIER'])
train_data.head()

train_data=pd.get_dummies(train_data,columns=['WORK_PROVINCE'])

train_data.head()
train_id=train_data['MEMBER_NO'].copy()
train_data.drop(['MEMBER_NO'],axis=1,inplace=True)
train_data.head()
train_id
def normalization(df):
    from sklearn.preprocessing import StandardScaler
    columns_name=df.columns
    scaler=StandardScaler()
    X=scaler.fit_transform(df)
    df=pd.DataFrame(X)
    df.columns=columns_name
    return df

def pca_decomposition(df,num):
    pca_data=df.copy()
    from sklearn.decomposition import PCA
    decompose=PCA(n_components=num,copy=True, 
          whiten=False, 
          svd_solver='auto', 
          tol=0.0, 
          iterated_power='auto', 
          random_state=0)
    decompose.fit(pca_data)
    trans_pca = decompose.transform(pca_data)
    df_pca = pd.DataFrame(trans_pca)
    variance_ratio = decompose.explained_variance_ratio_
    print('pca_components ',num,'   describe ',sum(variance_ratio))
    return df_pca,sum(variance_ratio)

def train_kmeans(x, k):
    KMModel = KMeans(n_clusters=k, 
                     init='k-means++', 
                     n_init=10, 
                     max_iter=600, 
                     tol=0.0001, 
                     precompute_distances='auto', 
                     verbose=0, 
                     random_state=1, 
                     copy_x=True, 
                     n_jobs=1, 
                     algorithm='auto')
    KMModel.fit(x) 
    print("kmeans,", k, "clusters:   ", "silhouette_score", silhouette_score(x, KMModel.labels_),'\n' )
    return KMModel
from sklearn.cluster import KMeans

train_normalize=normalization(train_data)


from sklearn.decomposition import PCA

alist=[]
dim=[]
vari=[]
for l in range(150,600,10):
    pca_result,var=pca_decomposition(train_normalize,l)
    alist.append(pca_result)
    vari.append(var)
    dim.append(l)
    
    
plt.figure(figsize=(20,8))
plt.plot(dim,vari,marker='o',linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative explained variance")
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score
df=train_normalize.copy()
for pca in [300,450,500,400]:
    df=train_normalize.copy()
    pca_df,vari=pca_decomposition(df,pca)
    for cluster in [2,3,4,5,6,7]:
        train_kmeans(pca_df,cluster)
wcss=[]
pca_df,vari=pca_decomposition(df,400)
for i in [2,3,4,5,6,7]:
    kmeans=train_kmeans(pca_df, i)
    wcss.append(kmeans.inertia_)

plt.plot([2,3,4,5,6,7], wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()
k_means_data=pca_df
list1=[]

kmeans=KMeans(n_clusters=2,init='k-means++',random_state=42)
labels=kmeans.fit_predict(k_means_data.iloc[:,:])


from sklearn.metrics import silhouette_score
print(silhouette_score(k_means_data,labels))
pca_final,vari=pca_decomposition(pca_df,3)
pca_final.head()
pca_final.info()
pca_final['first_pca']=pca_final[0]
pca_final['second_pca']=pca_final[1]
pca_final['third_pca']=pca_final[2]

pca_final=pca_final.drop(columns=[0,1,2])
pca_final.head()
pca_final['clusters']=labels
pca_final.head()
pca_final.tail()
pca_final.info()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
x=pca_final['first_pca']
y=pca_final['second_pca']
z=pca_final['third_pca']
labels=pca_final['clusters']
print("DONE")
ax.scatter3D(x, y, z,c=labels,marker='o')
plt.show()
ax.set_xlabel('first')
ax.set_ylabel('second')
ax.set_zlabel('third')
 
import seaborn as sns
sns.scatterplot(x=x,y=y,hue=labels,palette=['green','orange'])
sns.scatterplot(x=y,y=z,hue=labels,palette=['green','orange'])
sns.scatterplot(x=x,y=z,hue=labels,palette=['green','orange'])
train_data['cluster']=labels
train_data.info()
train_data.info()
sns.scatterplot(x=train_data['AGE'],y=train_data['FLIGHT_COUNT'],hue=labels,palette=['green','orange'])
sns.scatterplot(x=train_data['AGE'],y=train_data['BP_SUM'],hue=labels,palette=['green','orange'])
sns.scatterplot(x=train_data['FLIGHT_COUNT'],y=train_data['BP_SUM'],hue=labels,palette=['green','orange'])
train_data.info()
tidy_train=pd.concat([train_data,train_id],axis=1)
tidy_train.info()
tidy_train.to_csv("Tidy_Train.csv",index=False)
Train_tidy=pd.read_csv("Tidy_Train.csv")
Train_tidy.info()
train_data["MEMBER_NO"]=train_id
train_data.info()
Test_tidy=train_data.loc[49818:]
Test_tidy.info()
Test_tidy.head()
Test_tidy.info()
result=Test_tidy[["MEMBER_NO","cluster"]]
result.head()
result.info()
result['cluster'].unique()
result.to_csv("Final_result.csv",index=False)
test_check=pd.read_csv("Final_result.csv")
test_check.info()
Test_tidy.head()
sns.scatterplot(x=Test_tidy['AGE'],y=Test_tidy['FLIGHT_COUNT'],hue=labels,palette=['green','orange'])

sns.scatterplot(x=Test_tidy['AGE'],y=Test_tidy['BP_SUM'],hue=labels,palette=['green','orange'])

sns.scatterplot(x=Test_tidy['FLIGHT_COUNT'],y=Test_tidy['BP_SUM'],hue=labels,palette=['green','orange'])
