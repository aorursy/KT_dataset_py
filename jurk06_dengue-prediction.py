import pandas as pd
import numpy as np
import seaborn as sns
train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv')
train.sample(5)
test=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv')
test.shape
feat_train=pd.read_csv('../input/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv')
feat_train.shape
train.fillna(train.mean(), inplace=True)
test.fillna(train.mean(), inplace=True)
test.isnull().sum()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
train['ndvi_ne'].plot()
train['ndvi_nw'].plot()
plt.legend()
df=pd.merge(train, feat_train)
df.head()
c1=df[df['city']=='sj']
c1.shape
c2=df[df['city']=='iq']
c2.shape

c1['week_start_date']=pd.to_datetime(c1['week_start_date'])
c1.dtypes
%timeit pd.to_datetime(c1['week_start_date'], infer_datetime_format=True)
c1.dtypes
c1.set_index(c1['week_start_date'], inplace=True)
c1.shape
plt.figure(figsize=(10,10))
c1['ndvi_ne'].plot()
c1['ndvi_nw'].plot()
c1['ndvi_se'].plot()
c1['ndvi_sw'].plot()
plt.legend()
plt.show()


%timeit pd.to_datetime(c2['week_start_date'], infer_datetime_format=True)
c2.set_index(c2['week_start_date'], inplace=True)
plt.figure(figsize=(10,10))
c2['ndvi_ne'].plot()
c2['ndvi_nw'].plot()
c2['ndvi_se'].plot()
c2['ndvi_sw'].plot()
plt.legend()
plt.show()


plt.subplot(1,2,1)
c1['ndvi_ne'].plot()
c1['ndvi_nw'].plot()
c1['ndvi_se'].plot()
c1['ndvi_sw'].plot()
plt.legend()
plt.subplot(1,2,2)
c2['ndvi_ne'].plot()
c2['ndvi_nw'].plot()
c2['ndvi_se'].plot()
c2['ndvi_sw'].plot()
plt.legend()


%timeit pd.to_datetime(df['week_start_date'], infer_datetime_format=True)

c2['week_start_date']=pd.to_datetime(c2['week_start_date'])

c2.set_index(c2['week_start_date'], inplace=True)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
c1['ndvi_ne'].plot()
c1['ndvi_nw'].plot()
c1['ndvi_se'].plot()
c1['ndvi_sw'].plot()
plt.legend()
plt.subplot(1,2,2)
c2['ndvi_ne'].plot()
c2['ndvi_nw'].plot()
c2['ndvi_se'].plot()
c2['ndvi_sw'].plot()
plt.legend()

plt.figure(figsize=(15,15))
c1['reanalysis_air_temp_k'].plot()
c1['reanalysis_avg_temp_k'].plot()
c1['reanalysis_dew_point_temp_k'].plot()
c1['reanalysis_max_air_temp_k'].plot()
c1['reanalysis_min_air_temp_k'].plot()
plt.legend()

plt.figure(figsize=(15,15))
c2['reanalysis_air_temp_k'].plot()
c2['reanalysis_avg_temp_k'].plot()
c2['reanalysis_dew_point_temp_k'].plot()
c2['reanalysis_max_air_temp_k'].plot()
c2['reanalysis_min_air_temp_k'].plot()
plt.legend()

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
c1['reanalysis_air_temp_k'].plot()
c1['reanalysis_avg_temp_k'].plot()
c1['reanalysis_dew_point_temp_k'].plot()
c1['reanalysis_max_air_temp_k'].plot()
c1['reanalysis_min_air_temp_k'].plot()
plt.legend()
plt.subplot(1,2,2)
c2['reanalysis_air_temp_k'].plot()
c2['reanalysis_avg_temp_k'].plot()
c2['reanalysis_dew_point_temp_k'].plot()
c2['reanalysis_max_air_temp_k'].plot()
c2['reanalysis_min_air_temp_k'].plot()
plt.legend()
df.dtypes
g=sns.lmplot(x='total_cases', y='ndvi_ne', data=c1, markers='o')
g=sns.lmplot(x='total_cases', y='ndvi_nw', data=c1, markers='v')
g=sns.lmplot(x='total_cases', y='ndvi_se', data=c1, markers='^')
g=sns.lmplot(x='total_cases', y='ndvi_sw', data=c1, markers='s')

g=sns.lmplot(x='total_cases', y='ndvi_ne', data=df, markers='o', col='city', hue='city')
g=sns.lmplot(x='total_cases', y='ndvi_nw', data=df, markers='o', col='city', hue='city')
g=sns.lmplot(x='total_cases', y='ndvi_se', data=df, markers='o', col='city', hue='city')
g=sns.lmplot(x='total_cases', y='ndvi_sw', data=df, markers='o', col='city', hue='city')


g=sns.lmplot(x='year', y='ndvi_ne', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='ndvi_nw', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='ndvi_se', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='ndvi_sw', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)



g=sns.lmplot(x='year', y='reanalysis_air_temp_k', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_avg_temp_k', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_dew_point_temp_k', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_max_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_min_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)

g=sns.lmplot(x='total_cases', y='ndvi_ne', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='ndvi_nw', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='ndvi_se', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='ndvi_sw', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)

g=sns.lmplot(x='total_cases', y='reanalysis_air_temp_k', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_avg_temp_k', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_dew_point_temp_k', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_max_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_min_air_temp_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)


g=sns.lmplot(x='year', y='reanalysis_precip_amt_kg_per_m2', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_relative_humidity_percent', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_sat_precip_amt_mm', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_specific_humidity_g_per_kg', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='reanalysis_tdtr_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)

g=sns.lmplot(x='total_cases', y='reanalysis_precip_amt_kg_per_m2', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_relative_humidity_percent', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_sat_precip_amt_mm', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_specific_humidity_g_per_kg', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='reanalysis_tdtr_k', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)

g=sns.lmplot(x='year', y='station_avg_temp_c', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='station_diur_temp_rng_c', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='station_max_temp_c', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='year', y='station_min_temp_c', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)

g=sns.lmplot(x='total_cases', y='station_avg_temp_c', data=df, markers='o', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='station_diur_temp_rng_c', data=df, markers='v', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='station_max_temp_c', data=df, markers='s', col='city', hue='city',aspect=1.5, x_jitter=.1)
g=sns.lmplot(x='total_cases', y='station_min_temp_c', data=df, markers='^', col='city', hue='city',aspect=1.5, x_jitter=.1)

feature_col=['city','week_start_date', 'total_cases','weekofyear']
X=df.drop(feature_col, axis=1)


X.shape

y=df.iloc[:,24]

y.shape
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
np.mean(score)*100

clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
np.mean(score)*100

clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
np.mean(score)*100

clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
np.mean(score)*100

clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
np.mean(score)*100

test.head()
feature_col=['city','week_start_date', 'weekofyear']
test=test.drop(feature_col, axis=1)
clf=SVC()
clf.fit(X, y)

pred=clf.predict(test)
submission = pd.DataFrame({
        "total": pred
})

submission.to_csv('submission_format.csv', index=False)
submission = pd.read_csv('submission_format.csv')
submission.head()