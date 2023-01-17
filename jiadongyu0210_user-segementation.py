import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
mb=pd.read_csv('../input/user-segementation-mobike-clustering/week4_mobike.csv')
mb.info()
mb.head(10)
mb.drop('Unnamed: 0',axis=1,inplace=True)
mb.describe()
mb.gender.unique()
#观察数据缺失比例。发现有gender和birthyear确实的数据，占比不是特别大。
mb.isnull().sum()/mb.count()
sum(mb.duplicated())
#start_time, end_time转换成时间变量。
mb_clean=mb.copy()
mb_clean.info()
#对gender进行unkown填充
mb_clean['gender'].fillna('Unknown',inplace=True)
mb_clean.info()
#dropna
#age与birthyear属于同一类型，可以drop掉birthyear.以及无用行userid,bikeid
mb_clean.drop(['birthyear','user_id','bikeid','to_station_name','from_station_name'],axis=1,inplace=True)
#start-time,end-time转换成日期格式，age转化成数字Int格式。
mb_clean['start_time']=pd.to_datetime(mb_clean['start_time'])
mb_clean['end_time']=pd.to_datetime(mb_clean['end_time'])
print(mb_clean.dtypes)
mb_clean.head(10)
#年龄中有未知年龄填充为0
mb_clean['age']=mb_clean['age'].replace(" ", "0")
mb_clean['age']=mb_clean['age'].astype(int)
#创建一个位置年龄的分类
mb_clean['age_known']=mb_clean['age'].apply(lambda x:1 if x>0 else 0 )
mb_clean.info()
mb_clean.age.describe()
# 对age进行分布分析
sns.distplot(mb_clean['age'],10)
mb_clean['age'].plot(kind='box')
#年龄异常值处理drop
mb_clean=mb_clean[mb_clean['age']<=80]
sns.countplot(x='age_known',data=mb_clean)
#位置年龄的人数占比
mb_clean['age_known'][mb_clean['age_known']==0].count()/mb_clean.age_known.count()
mb_clean.timeduration.describe()
# 对age进行分布分析
bins=[0,5,10,15,20,25,30,35,40,45,50,55,60]
mb_clean['timeduration_range']=pd.cut(mb_clean['timeduration'],bins,right=False)
mb_clean.groupby(['timeduration_range']).timeduration.count()
plt.figure(figsize=(15,5))
sns.countplot(x='timeduration_range',data=mb_clean)
mb_clean.tripduration.describe()
# 对age进行分布分析
bins=[0,350,700,1050,1400,2000,5000,10000,50000,100000,250000]
mb_clean['tripduration_range']=pd.cut(mb_clean['tripduration'],bins,right=False)
mb_clean.groupby(['tripduration_range']).tripduration.count()
plt.figure(figsize=(15,5))
sns.countplot(x='tripduration_range',data=mb_clean)
mb_clean=mb_clean[mb_clean['tripduration']<10000]
sns.countplot(x='gender',data=mb_clean)
#观察用户填写信息
sns.countplot(x='age_known',hue='gender',data=mb_clean)
sns.countplot(x='usertype',data=mb_clean)
sns.countplot(x='usertype',hue='gender',data=mb_clean)
#from datetime import date
#mb_clean['weekday']=mb_clean['start_time'].weekday()
#Return the day of the week as an integer, where Monday is 0 and Sunday is 6.
mb_clean['weekday']= mb_clean['start_time'].apply(lambda time: time.dayofweek)
sns.countplot(x='weekday',data=mb_clean)
mb_clean['weekdayornot']=mb_clean['weekday'].apply(lambda x:1 if x<5 else 0)
mb_clean['hour'] = mb_clean['start_time'].apply(lambda time: time.hour)
sns.countplot(x='hour',data=mb_clean)

def peak_hour(x):
    if x==7 or x==8:
       return 'early'
    elif x==16 or x==17 or x==18 or x==19:
       return 'late'
    else:
       return 'normal'

mb_clean['peakornot']=mb_clean['hour'].apply(peak_hour)
mb_clean.from_station_id.unique()
mb_clean.groupby(['from_station_id'])['start_time'].count()
mb_clean.to_station_id.unique()
mb_clean.groupby(['to_station_id'])['start_time'].count()
mb_clean.head()
#去掉未知年龄的用户，以便后续分析。
mb_model=mb_clean.copy()
mb_model=mb_model[mb_model['age']>0]
mb_model=mb_model.drop(['start_time','end_time','from_station_id','to_station_id','timeduration_range','tripduration_range'],axis=1)
mb_model.head()
mb_model=pd.get_dummies(mb_model)
mb_model.head()
mb_model=mb_model.drop(['usertype_Customer','gender_Female','gender_Unknown','age_known'],axis=1)
mb_model.head()
plt.figure(figsize=(10,10))
sns.heatmap(mb_model.corr())
#数据标准化
mb_model_x=mb_model[['timeduration','age','gender_Male','usertype_Subscriber','peakornot_early','peakornot_late','peakornot_normal','weekdayornot']]
from sklearn.preprocessing import scale
from sklearn import cluster
from sklearn import metrics
x=pd.DataFrame(scale(mb_model_x))

#评估模型
for i in np.arange(2,12,1):
    model=cluster.KMeans(n_clusters=i,random_state=10)
    model.fit(x)
    x_cluster=model.fit_predict(x)
    print(f'聚类类别:{i},评分:{metrics.silhouette_score(x,x_cluster)}')


model=cluster.KMeans(n_clusters=5,random_state=10)
model.fit(x)
mb_model['cluster']=model.labels_

centers=pd.DataFrame(model.cluster_centers_ , columns=['timeduration','age','gender_Male','usertype_Subscriber','peakornot_early','peakornot_late','peakornot_normal','weekdayornot'])
print(centers)