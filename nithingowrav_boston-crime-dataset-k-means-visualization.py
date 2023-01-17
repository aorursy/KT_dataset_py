#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline

from mpl_toolkits.basemap import Basemap
import folium
from folium import plugins
os.chdir("/kaggle/input/crimes-in-boston/")
os.listdir()
df=pd.read_csv(r'crime.csv', encoding='unicode_escape',low_memory=False)
df.head()
def pie_plot(list_number, list_unique):
    plt.figure(figsize=(20,10))
    plt.pie(list_unique, 
        labels=list_number,
        autopct='%1.1f%%', 
        shadow=True, 
        startangle=140)
 
    plt.axis('equal')
    plt.show()
    return 0
def bar_chart(list_number, list_unique,xlabel,ylabel):
    objects = list_unique
    y_pos = np.arange(len(objects))
    performance = list_number
 
    plt.figure(figsize=(20,10))    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(ylabel) 
    plt.xlabel(xlabel)
    plt.show()
    
    return 0
#df_dy_crim=pd.DataFrame(df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.transform(max)).reset_index()
df_dy_crim=df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.reset_index(name='Number_of_incidents')
dy_mst_crm=df_dy_crim.sort_values(by=['Number_of_incidents'],ascending=False).head(1)
dy_mst_crm
df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.reset_index(name='Number_of_incidents').sort_values(by='Number_of_incidents', ascending=False).head(1)
crm_typ_yr=df.groupby(['OFFENSE_CODE_GROUP','YEAR']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
crm_typ_yr.head()
loc_crm_typ=df.loc[0:,['OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION','STREET']].groupby('STREET').agg(' ,'.join).reset_index()
loc_crm_typ['Lat']= df.loc[0:,['Lat']]
loc_crm_typ['Long']=df.loc[0:,['Long']]
loc_crm_typ.head()
df_inc_by_yr=df.groupby('YEAR').count().INCIDENT_NUMBER.reset_index(name="Number of Incidents")
df_inc_by_yr_lbl=df['YEAR'].unique()
df_inc_by_yr_lbl
def create_list_number_crime(name_column, list_unique):
    # list_unique = df[name_column].unique()
    
    i = 0
    
    list_number = list()
    
    while i < len(list_unique):
        list_number.append(len(df.loc[df[name_column] == list_unique[i]]))
        i += 1
    
    return list_unique, list_number
create_list_number_crime('YEAR',df['YEAR'].unique())
pie_plot(list(df['YEAR'].unique()),df_inc_by_yr['Number of Incidents'])
bar_chart(df_inc_by_yr['Number of Incidents'],df_inc_by_yr['YEAR'],'Year','Number of Incidents')
def drop_NaN_two_var(x, y):

    df1 = df[[x, y]].dropna()
    print(df1.shape)

    x_value = df1[x]
    y_value = df1[y]

    del df1
        
    print(x + ': ' + str(x_value.shape))
    print(y + ': ' + str(y_value.shape))
        
    return x_value, y_value
df_dst_yr_crm=df.groupby(by=['DISTRICT','YEAR']).count().INCIDENT_NUMBER.reset_index(name='Number of Incidents')
sns.barplot(x='DISTRICT',y='Number of Incidents',hue='YEAR',data=df_dst_yr_crm)
plt.tight_layout()
plt.show()
df_mnth_crm = df.groupby('MONTH').count().INCIDENT_NUMBER.reset_index(name='Number of Incidents')
bar_chart(df_mnth_crm['Number of Incidents'],df_mnth_crm['MONTH'],'Month','Number of Incidents')
df_dy_crim=df.groupby(['DAY_OF_WEEK']).count().INCIDENT_NUMBER.reset_index(name='num_of_incidents')
df_dy_crim.head()
day_of_week=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
bar_chart(df_dy_crim['num_of_incidents'],day_of_week,'Day of the week','Number of Incidents')
hr_num_crm=df.groupby('HOUR').count().INCIDENT_NUMBER.reset_index(name='num_of_inc')
hr_num_crm
bar_chart(hr_num_crm['num_of_inc'],hr_num_crm['HOUR'],'Hour of the day','Number of Incidents')
hr_yr_crm=df.groupby(['HOUR','YEAR']).count().INCIDENT_NUMBER.reset_index(name='Number of Incidents')

sns.barplot(x='HOUR',y='Number of Incidents',hue='YEAR',data=hr_yr_crm)
crm_off_grp_df=df.groupby(['OFFENSE_CODE_GROUP']).count().INCIDENT_NUMBER.reset_index(name='Number of Incidents').sort_values(by='Number of Incidents', ascending=False)
fig, ax = plt.subplots()
fig.set_size_inches(10, 20)
sns.barplot('Number of Incidents','OFFENSE_CODE_GROUP',data=crm_off_grp_df,ax=ax)
df['SHOOTING'].fillna(0,inplace=True)

df['SHOOTING'] = df['SHOOTING'].map({
    0: 0,
    'Y':1
})

df['SHOOTING'].unique()
Shoot_True=len(df.loc[df['SHOOTING'] == 1])
Shoot_False=len(df.loc[df['SHOOTING'] == 0])

print('With shooting(num): ' + str(Shoot_True))
print('With shooting(%):   ' + str(round(Shoot_True*100/len(df),2))+'%')
print()
print('Without shooting(num): ' + str(Shoot_False))
print('Without shooting(%):   ' + str(round(Shoot_False*100/len(df),2))+'%')
shoot_by_yr=df[df['SHOOTING']==1].groupby('YEAR').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_yr
bar_chart(shoot_by_yr['Number of Shootings'],shoot_by_yr['YEAR'],'Year','Number of Shootings')
shoot_by_mnth=df[df['SHOOTING']==1].groupby('MONTH').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_mnth

bar_chart(shoot_by_mnth['Number of Shootings'],shoot_by_mnth['MONTH'],'Month','Number of Shootings')
shoot_by_day=df[df['SHOOTING']==1].groupby('DAY_OF_WEEK').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_day

bar_chart(shoot_by_day['Number of Shootings'],day_of_week,'Day of Week','Number of Shootings')
shoot_by_hour=df[df['SHOOTING']==1].groupby('HOUR').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_hour

bar_chart(shoot_by_hour['Number of Shootings'],shoot_by_hour['HOUR'],'Hour of the day','Number of Shootings')
shoot_by_district=df[df['SHOOTING']==1].groupby('DISTRICT').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')
shoot_by_district

bar_chart(shoot_by_district['Number of Shootings'],shoot_by_district['DISTRICT'],'Hour of the day','Number of Shootings')
plt.figure(figsize=[10,5])
color_dis=plt.cm.Spectral(np.linspace(0, 1, len(shoot_by_district['DISTRICT'])))
plt.bar(shoot_by_district['DISTRICT'],shoot_by_district['Number of Shootings'], color=color_dis)
plt.show()
plt.figure(figsize=(10,5))
df_shoot=df[df['SHOOTING']==1]
df_shoot['DISTRICT'].value_counts().plot.bar(color=color_dis)
plt.show()
shoot_location = df_shoot[['Lat','Long']]
shoot_location = shoot_location.dropna()

shoot_location.head()
shoot_location=shoot_location.loc[(shoot_location['Lat']>40) & (shoot_location['Long'] < -60)]  

x_shoot = shoot_location['Long']
y_shoot = shoot_location['Lat']

sns.jointplot(x_shoot, y_shoot, kind='scatter')
sns.jointplot(x_shoot, y_shoot, kind='hex')
sns.jointplot(x_shoot, y_shoot, kind='kde')
sns.jointplot(x_shoot,y_shoot,kind='reg')
plt.show()
shoot_by_UCR=df[df['SHOOTING']==1].groupby('UCR_PART').count().INCIDENT_NUMBER.reset_index(name='Number of Shootings')

plt.figure(figsize=(10,5))
color_ucr=plt.cm.Spectral(np.linspace(0, 1, len(shoot_by_UCR['UCR_PART'])))
df_shoot['UCR_PART'].value_counts().plot.bar(color=color_ucr)
plt.show()
df[['Lat','Long']].describe()

location = df[['Lat','Long']]
location = location.dropna()

location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]
x = location['Long']
y = location['Lat']
rand_colors = np.random.rand(len(x))
plt.figure(figsize=(20,20))
plt.scatter(x, y,c=rand_colors, alpha=0.5)
plt.show()
m = folium.Map([42.348624, -71.062492], zoom_start=11)
#generate various join plots to see if there is a pattern or trend
sns.jointplot(x, y, kind='scatter')
sns.jointplot(x, y, kind='hex')
sns.jointplot(x, y, kind='kde')
df.isnull().sum()
df['Day']=0
df['Night']=0
# Day time for 1st month
df['Day'].loc[(df['MONTH'] == 1) & (df['HOUR'] >= 6) & (df['HOUR'] <= 18)] = 1

# Day time for 2st month
df['Day'].loc[(df['MONTH'] == 2) & (df['HOUR'] >= 6) & (df['HOUR'] <= 19)] = 1

# Day time for 3rd month
df['Day'].loc[(df['MONTH'] == 3) & (df['HOUR'] >= 6) & (df['HOUR'] <= 20)] = 1

# Day time for 4st month
df['Day'].loc[(df['MONTH'] == 4) & (df['HOUR'] >= 5) & (df['HOUR'] <= 20)] = 1

# Day time for 5th month
df['Day'].loc[(df['MONTH'] == 5) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1

# Day time for 6th month
df['Day'].loc[(df['MONTH'] == 6) & (df['HOUR'] >= 4) & (df['HOUR'] <= 21)] = 1

# Day time for 7th month
df['Day'].loc[(df['MONTH'] == 7) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1

# Day time for 8th month
df['Day'].loc[(df['MONTH'] == 8) & (df['HOUR'] >= 5) & (df['HOUR'] <= 21)] = 1

# Day time for 9th month
df['Day'].loc[(df['MONTH'] == 9) & (df['HOUR'] >= 6) & (df['HOUR'] <= 20)] = 1

# Day time for 10th month
df['Day'].loc[(df['MONTH'] == 10) & (df['HOUR'] >= 6) & (df['HOUR'] <= 19)] = 1

# Day time for 11th month
df['Day'].loc[(df['MONTH'] == 11) & (df['HOUR'] >= 6) & (df['HOUR'] <= 17)] = 1

# Day time for 12th month
df['Day'].loc[(df['MONTH'] == 12) & (df['HOUR'] >= 7) & (df['HOUR'] <= 17)] = 1


#Update Night as 1 where Day is 0
df['Night'].loc[df['Day']==0]=1
plt.figure(figsize=(16,8))
color_DN=plt.cm.Spectral(np.linspace(0, 1, 2))
df['Night'].value_counts().plot.bar(color=color_DN)
plt.show()
df['OFFENSE_CODE_GROUP'].value_counts().head(15)
list_offense_code_group=('Motor Vehicle Accident Response',
                           'Larceny',
                           'Medical Assistance',
                           'Investigate Person',
                           'Other',
                           'Drug Violation',
                           'Simple Assault',
                           'Vandalism',
                           'Verbal Disputes',
                           'Towed',
                           'Investigate Property',
                           'Larceny From Motor Vehicle')
list_offense_code_group
df_model = pd.DataFrame()
i = 0

while i < len(list_offense_code_group):

    df_model = df_model.append(df.loc[df['OFFENSE_CODE_GROUP'] == list_offense_code_group[i]])
    
    i+=1
df_model.columns
list_column = ['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK',
               'HOUR','Lat','Long', 'OFFENSE_CODE_GROUP','Day','Night']
df_model=df_model[list_column]
df_model['DISTRICT'].unique()
df_model['DISTRICT'] = df_model['DISTRICT'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})

df_model['DISTRICT'].unique()
df_model['REPORTING_AREA'] = pd.to_numeric(df_model['REPORTING_AREA'], errors='coerce')
df_model['MONTH'].unique()
df_model['DAY_OF_WEEK'] = df_model['DAY_OF_WEEK'].map({
    'Monday':1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6, 
    'Sunday':7    
})

df_model['DAY_OF_WEEK'].unique()
df_model['HOUR'].unique()
df_model[['Lat', 'Long']].head()
df_model.fillna(0, inplace = True)
x = df_model[['DISTRICT','REPORTING_AREA','MONTH','DAY_OF_WEEK','HOUR','Lat','Long','Day','Night']]
y = df_model['OFFENSE_CODE_GROUP']
y.unique()
y=y.map({
    'Motor Vehicle Accident Response':1, 
    'Larceny':2, 
    'Medical Assistance':3,
    'Investigate Person':4, 
    'Other':5, 
    'Drug Violation':6, 
    'Simple Assault':7,
    'Vandalism':8, 
    'Verbal Disputes':9, 
    'Towed':10, 
    'Investigate Property':11,
    'Larceny From Motor Vehicle':12
})
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
#!conda install -c conda-forge lightgbm --yes
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
def func_results(result):
    print('mean: ' + str(result.mean()))
    print('max: ' + str(result.max()))
    print('min: ' + str(result.min()))
    return result
def func_DecisionTreeClassifier(x_train, y_train):
    dec_tree = DecisionTreeClassifier()
    dec_tree = dec_tree.fit(x_train, y_train)

    dec_tree_pred = dec_tree.predict(x_test)

    dec_tree_score = f1_score(y_test, dec_tree_pred, average=None)
    return func_results(dec_tree_score)
func_DecisionTreeClassifier(x_train,y_train)
def func_BernoulliNB(x_train, y_train):
    bernoulli = BernoulliNB()
    bernoulli = bernoulli.fit(x_train, y_train)

    bernoulli_pred = bernoulli.predict(x_test)

    bernoulli_score = f1_score(y_test, bernoulli_pred, average=None)
    return func_results(bernoulli_score)
func_BernoulliNB(x_train,y_train)
def func_ext_tree_cls(x_train,y_train):
    ext_tree=ExtraTreeClassifier()
    ext_tree=ext_tree.fit(x_train,y_train)
    ext_tree_pred=ext_tree.predict(x_test)
    ext_tree_score=f1_score(y_test,ext_tree_pred,average=None)
    return func_results(ext_tree_score)
func_ext_tree_cls(x_train,y_train)
def func_KNeighborsClassifier(x_train, y_train,n):
    Kneigh = KNeighborsClassifier(n_neighbors=n)
    Kneigh.fit(x_train, y_train) 

    Kneigh_pred = Kneigh.predict(x_test)

    Kneigh_score = f1_score(y_test, Kneigh_pred, average=None)
    return func_results(Kneigh_score),Kneigh_pred

KNN_score, KNN_pred=func_KNeighborsClassifier(x_train,y_train,5)
KNN_score
from sklearn import metrics
Ks=20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
mean_acc - 1 * std_acc
mean_acc + 1 * std_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
def func_GaussianNB(x_train, y_train):
    gaussian = GaussianNB()
    gaussian = gaussian.fit(x_train, y_train)

    gaussian_pred = gaussian.predict(x_test)

    gaussian_score = f1_score(y_test, gaussian_pred, average=None)
    return func_results(gaussian_score)

func_GaussianNB(x_train,y_train)
def func_RandomForestClassifier(x_train, y_train):
    rfc = RandomForestClassifier()
    rfc = rfc.fit(x_train, y_train)

    rfc_pred = rfc.predict(x_test)

    rfc_score = f1_score(y_test, rfc_pred, average=None)
    return func_results(rfc_score)
func_RandomForestClassifier(x_train,y_train)
def func_LGBMClassifier(x_train, y_train):
    lgbm = LGBMClassifier()
    lgbm = lgbm.fit(x_train, y_train)

    lgbm_pred = lgbm.predict(x_test)

    lgbm_score = f1_score(y_test, lgbm_pred, average=None)
    return func_results(lgbm_score)
func_LGBMClassifier(x_train,y_train)
df_model_2 = df[['OFFENSE_CODE', 'DISTRICT','MONTH','DAY_OF_WEEK','HOUR','Day','Night']]
df_model_2.head()
df_model_2['OFFENSE_CODE'] = pd.to_numeric(df_model_2['OFFENSE_CODE'], errors='coerce')
df_model_2['DISTRICT'] = df_model_2['DISTRICT'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})

df_model_2['DISTRICT'].unique()
df_model_2['DAY_OF_WEEK'] = df_model_2['DAY_OF_WEEK'].map({
    'Tuesday':2, 
    'Saturday':6, 
    'Monday':1, 
    'Sunday':7, 
    'Thursday':4, 
    'Wednesday':3,
    'Friday':5
})

df_model_2['DAY_OF_WEEK'].unique()
df_model_2.isnull().sum()
df_model_2 = df_model_2.dropna()
df_model_2.isnull().sum()
df_model_2.shape
x = df_model_2[['OFFENSE_CODE','MONTH','DAY_OF_WEEK','HOUR','Day','Night']]
y = df_model_2['DISTRICT']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
func_BernoulliNB(x_train,y_train)
func_DecisionTreeClassifier(x_train,y_train)
func_ext_tree_cls(x_train,y_train)
func_GaussianNB(x_train,y_train)
func_KNeighborsClassifier(x_train,y_train,5)
Ks=10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
    yhat=neigh.predict(x_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
func_LGBMClassifier(x_train,y_train)
func_RandomForestClassifier(x_train,y_train)
df_model3 = df[['DISTRICT','REPORTING_AREA', 'MONTH','DAY_OF_WEEK','HOUR','UCR_PART','Lat','Long']]
df_model3['DISTRICT'] = df_model3['DISTRICT'].map({
    'B3':1, 
    'E18':2, 
    'B2':3, 
    'E5':4, 
    'C6':5, 
    'D14':6, 
    'E13':7, 
    'C11':8, 
    'D4':9, 
    'A7':10, 
    'A1':11, 
    'A15':12
})
df_model3['REPORTING_AREA'] = pd.to_numeric(df_model3['REPORTING_AREA'], errors='coerce')
df_model3['DAY_OF_WEEK'] = df_model3['DAY_OF_WEEK'].map({
    'Tuesday':2, 
    'Saturday':6, 
    'Monday':1, 
    'Sunday':7, 
    'Thursday':4, 
    'Wednesday':3,
    'Friday':5
})
df_model3['UCR_PART'].unique()
df_model3['UCR_PART'] = df_model3['UCR_PART'].map({
    'Part Three':3, 
    'Part One':1, 
    'Part Two':2, 
#    'Other':4
})
df_model3 = df_model3.dropna()
print(df_model3.shape)
df_model3.isnull().sum()
x = df_model3[['DISTRICT','REPORTING_AREA', 'MONTH','DAY_OF_WEEK','HOUR','Lat','Long']]
y = df_model3['UCR_PART']
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y, 
    test_size = 0.1,
    random_state=42
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
func_BernoulliNB(x_train,y_train)
func_DecisionTreeClassifier(x_train,y_train)
func_ext_tree_cls(x_train,y_train)
func_GaussianNB(x_train,y_train)
func_KNeighborsClassifier(x_train,y_train,5)
func_LGBMClassifier(x_train,y_train)
func_RandomForestClassifier(x_train,y_train)
location.isnull().sum()
location.shape
x = location['Long']
y = location['Lat']

colors = np.random.rand(len(location))

plt.figure(figsize=(20,20))
plt.scatter(x, y,c=colors, alpha=0.5)
plt.show()
X = location
X = X[~np.isnan(X)]
from sklearn.cluster import KMeans
def Kmeanscl(X, nclust):
    kmeansmodel = KMeans(nclust)
    kmeansmodel.fit(X)
    clust_labels = kmeansmodel.predict(X)
    cent = kmeansmodel.cluster_centers_
    return (clust_labels, cent)
clust_labels, cent = Kmeanscl(X, 2)
kmeans = pd.DataFrame(clust_labels)
X.insert((X.shape[1]),'kmeans',kmeans)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
scatter = ax.scatter(X['Long'],X['Lat'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Long')
ax.set_ylabel('Lat')
plt.colorbar(scatter)
X = location
X = X[~np.isnan(X)]
clust_labels, cent = Kmeanscl(X, 3)
kmeans = pd.DataFrame(clust_labels)
X.insert((X.shape[1]),'kmeans',kmeans)
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)
scatter = ax.scatter(X['Long'],X['Lat'],
                     c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('Long')
ax.set_ylabel('Lat')
plt.colorbar(scatter)
#!conda install -c districtdatalabs yellowbrick --yes
from yellowbrick.cluster import KElbowVisualizer
X = location
X = X[~np.isnan(X)]
KMdl=KMeans()
visualizer = KElbowVisualizer(KMdl, k=(4,12),locate_elbow=True)

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.poof()  
KMdl2=KMeans()
visualizer2 = KElbowVisualizer(KMdl2, k=(4,12), metric='calinski_harabasz',locate_elbow=True)
visualizer2.fit(X)        # Fit the data to the visualizer
visualizer2.poof()   
ucr_prt1_shoot_crm=df[(df['Lat']>=40) & (df['Long']<=-70) &(df['UCR_PART']=='Part One') & (df['SHOOTING']==1)].fillna(0).reset_index()
m = folium.Map( [42.3601,-71.0589],zoom_start=13, tiles='OpenStreetMap')
for i in range(0,len(ucr_prt1_shoot_crm)):
    #folium.Marker(ucr_prt1_crm.iloc[i]['Lat'], ucr_prt1_crm.iloc[i]['Long'], popup=ucr_prt1_crm.iloc[i]['OFFENSE_CODE_GROUP']).add_to(m)
    folium.Marker([ucr_prt1_shoot_crm.iloc[i]['Lat'], ucr_prt1_shoot_crm.iloc[i]['Long']], popup=ucr_prt1_shoot_crm.iloc[i]['OFFENSE_CODE_GROUP']).add_to(m)
    
m