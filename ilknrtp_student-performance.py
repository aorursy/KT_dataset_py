# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
sns.set_palette("coolwarm", 4)
mat = pd.read_csv('../input/ozyegin-dataset/mat.csv', sep=';')
por = pd.read_csv('../input/ozyegin-dataset/por.csv', sep=';')
#Mat ile Por merge
data = pd.merge(mat, por, on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"], suffixes=["_mat", "_por"])
data.info()
data.head()
sns.set_palette("coolwarm", 7)
sns.countplot(x='age', data=data)
sns.countplot(x='Medu', data=data)
sns.countplot(x='Fedu', data=data)
sns.countplot(x='Mjob', data=data)
sns.countplot(x='Fjob', data=data)
sns.countplot(x='reason', data=data)
sns.countplot(x='guardian_mat', data=data)
sns.countplot(x='traveltime_mat', data=data)
sns.countplot(x='studytime_mat', data=data)
sns.countplot(x='sex', data=data)
sns.countplot(x='Pstatus', data=data)
sns.countplot(x='famsize', data=data)
sns.set_palette("coolwarm", 40)
fig, ax = plt.subplots(figsize=(10,7))
sns.countplot(x='absences_mat', data=data)
ax.set_xlabel('Devamsızlık', fontsize=12)
ax.set_ylabel('Öğrenci Sayısı', fontsize=12)
ax.set_title('Devamsızlık Yapan Öğrencilerin Dağılımı', fontsize=20)
data.plot(kind='scatter',x='absences_mat',y='G3_mat')
df = data.copy()
#Kategorik sütunlar
list(data.select_dtypes(['object']).columns)
from sklearn.preprocessing import LabelEncoder

kategorik_sutunlar = ['school',
 'sex',
 'address',
 'famsize',
 'Pstatus',
 'Mjob',
 'Fjob',
 'reason',
 'guardian_mat',
 'schoolsup_mat',
 'famsup_mat',
 'paid_mat',
 'activities_mat',
 'nursery',
 'higher_mat',
 'internet',
 'romantic_mat',
 'guardian_por',
 'schoolsup_por',
 'famsup_por',
 'paid_por',
 'activities_por',
 'higher_por',
 'romantic_por']

for i in kategorik_sutunlar:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
df.head()
df.columns
y = df.G3_mat
X = df[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian_mat', 'traveltime_mat',
       'studytime_mat', 'failures_mat', 'schoolsup_mat', 'famsup_mat',
       'paid_mat', 'activities_mat', 'nursery', 'higher_mat', 'internet',
       'romantic_mat', 'famrel_mat', 'freetime_mat', 'goout_mat', 'Dalc_mat',
       'Walc_mat', 'health_mat', 'absences_mat','G1_mat', 'G2_mat',
       'guardian_por', 'traveltime_por', 'studytime_por', 'failures_por',
       'schoolsup_por', 'famsup_por', 'paid_por', 'activities_por',
       'higher_por', 'romantic_por', 'famrel_por', 'freetime_por', 'goout_por',
       'Dalc_por', 'Walc_por', 'health_por', 'absences_por', 'G1_por',
       'G2_por', 'G3_por']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lgb_train = lgb.Dataset(data=X_train, label=y_train,  free_raw_data=False)


params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'regression_l2',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'verbose': -1
}

gbm = lgb.train(params,
                lgb_train)
y_pred = gbm.predict(X_test)
ax = lgb.plot_importance(gbm, max_num_features=33, figsize=(10,8))
ax.set_title('')
ax.set_xlabel('Özniteliklerin Önemi')
ax.set_ylabel('Öznitelikler')
plt.show()
from sklearn.metrics import mean_squared_error 
mean_squared_error(y_test,y_pred)
from math import sqrt
sqrt(mean_squared_error(y_test, y_pred))
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), annot=True)
sns.set_palette("coolwarm", 10)
data_graph = data.groupby(['school'])['G3_mat'].mean()
ax = data_graph.plot(kind='bar', figsize=(10,6), fontsize=13);
ax.set_title("Liselere Göre Matematik Başarısı", fontsize=22)
plt.show()
sns.set_palette("coolwarm", 10)
data_graph = data.groupby(['studytime_mat'])['G3_mat'].mean()
ax = data_graph.plot(kind='bar', figsize=(10,6), fontsize=13)
ax.set_xlabel('Ders Çalışma Süresi', fontsize=12)
ax.set_ylabel('Matematik Notu', fontsize=12)
ax.set_title("Ders Çalışma Süresine Göre Matematik Başarısı", fontsize=22)
plt.show()
data.plot(kind='scatter',x='studytime_mat',y='G3_mat')
sns.set_palette("coolwarm", 10)
data_graph = data.groupby(['activities_mat'])['G3_mat'].mean()
ax = data_graph.plot(kind='bar', figsize=(10,6), fontsize=13)
ax.set_xlabel('Müfredat Dışı Faaliyetler', fontsize=12)
ax.set_ylabel('Matematik Notu', fontsize=12)
ax.set_title("Müfredat Dışı Faaliyetlere Göre Matematik Başarısı", fontsize=22)
plt.show()
data.plot(kind='scatter',x='activities_mat',y='G3_mat')
sns.set_palette("coolwarm", 10)
data_graph = data.groupby(['paid_mat'])['G3_mat'].mean()
ax = data_graph.plot(kind='bar', figsize=(10,6), fontsize=13)
ax.set_xlabel('Ücretli Dersler', fontsize=12)
ax.set_ylabel('Matematik Notu', fontsize=12)
ax.set_title("Ücretli Derslere Katılımın Matematik Başarısı Üzerindeki Etkisi", fontsize=22)
plt.show()
sns.set_palette("coolwarm", 10)
data_graph = data.groupby(['health_mat'])['G3_mat'].mean()
ax = data_graph.plot(kind='bar', figsize=(10,6), fontsize=13)
ax.set_xlabel('Sağlık Durumu', fontsize=12)
ax.set_ylabel('Matematik Notu', fontsize=12)
ax.set_title("Sağlık Durumuna Göre Matematik Başarısı", fontsize=22)
plt.show()
data.plot(kind='scatter',x='school',y='age')
data.plot(kind='scatter',x='sex', y='absences_mat')
sns.set_palette("coolwarm", 21)
fig, ax = plt.subplots(figsize=(10,7))
sns.barplot(x=df['G3_mat'].value_counts().index, y=df['G3_mat'].value_counts())
ax.set_xlabel('Matemetik Notu', fontsize=12)
ax.set_ylabel('Öğrenci Sayısı', fontsize=12)
ax.set_title('Matematik Notu Dağılımı', fontsize=20)
fig, ax = plt.subplots(figsize=(10,7))
plt.scatter(df['age'], df['G3_mat'])
plt.show()
sns.lmplot(x='absences_mat', y='G3_mat', data=df, height=7, aspect=1.6, legend_out=False)
sns.set_palette("coolwarm", 2),
sns.lmplot(x='age', y='G3_mat',hue="sex", data=data, height=7, aspect=1.6, legend_out=False)
sns.set_palette("coolwarm", 2)
sns.lmplot(x='absences_mat', y='G3_mat',hue="sex", data=data, height=7, aspect=1.6, legend_out=False)
data[data['sex']=='F']['G3_mat'].mean()
data[data['sex']=='M']['G3_mat'].mean()
data[data['address']=='U']['G3_mat'].mean()
data[data['address']=='R']['G3_mat'].mean()

