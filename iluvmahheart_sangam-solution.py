import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df_train=pd.read_csv("../input/Train.csv")

df_test=pd.read_csv("../input/Test.csv")

df_sample=pd.read_csv("../input/sample_submission.csv")
df_train.head()
df_train.shape
df_train.dtypes
df_test.head()
df_test.shape
df_sample.head()
df_sample.shape
df_train.isnull().sum()
df_test.isnull().sum()
df_train.head()
del df_train['date_time']
submit = pd.DataFrame(columns=['date_time','traffic_volume'])
submit_time=df_test['date_time']
submit['date_time']=submit_time
submit.head()
del df_test['date_time']
df_train.head()
df_test.head()
y=df_train['traffic_volume']
del df_train['traffic_volume']
df_train.dtypes
df_train['is_holiday'].unique()
df_train['weather_type'].unique()
df_train['weather_description'].unique()
plt.matshow(df_train.corr())

plt.show()
corr = df_train.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
is_holiday=pd.factorize(df_train['is_holiday'])[0]

is_holiday1=pd.factorize(df_test['is_holiday'])[0]

weather_type=pd.factorize(df_train['weather_type'])[0]

weather_type1=pd.factorize(df_test['weather_type'])[0]

weather_description=pd.factorize(df_train['weather_description'])[0]

weather_description1=pd.factorize(df_test['weather_description'])[0]
temp_train = pd.DataFrame({'is_holiday':is_holiday , 'weather_type':weather_type , 'weather_description':weather_description})
temp_test = pd.DataFrame({'is_holiday1':is_holiday1 , 'weather_type1':weather_type1 , 'weather_description1':weather_description1})
df_train=pd.concat([df_train,temp_train],axis=1)

df_test=pd.concat([df_test,temp_test],axis=1)
df_train.drop(['is_holiday','weather_type','weather_description'],axis=1,inplace=True)
df_test.drop(['is_holiday1','weather_type1','weather_description1','is_holiday','weather_type','weather_description'],axis=1,inplace=True)
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_jobs=2, random_state=0)
clf.fit(df_train, y)
preds=clf.predict(df_test)
submit['traffic_volume']=preds
submit.to_csv("output.csv",index=False)
submit.head()
df_train.columns
df_test.columns