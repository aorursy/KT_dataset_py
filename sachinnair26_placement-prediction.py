import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns



df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.drop('sl_no',inplace=True,axis=1)

df['salary'].fillna('0',inplace=True)

df.head()

plt.figure(figsize=(8,8))

cross_gender = pd.crosstab(df['gender'],df['status'])

plt.xlabel('Gender')

plt.ylabel('Count')

plt.bar(['F','M'],cross_gender['Placed'])

plt.bar(['F','M'],cross_gender['Not Placed'])

print(cross_gender)
cross_ssc_b = pd.crosstab(df['ssc_b'],df['status'])

plt.figure(figsize=(8,8))

plt.xlabel('SSC_B')

plt.ylabel('Count')

plt.bar(['Central','Others'],cross_ssc_b['Placed'])

plt.bar(['Central','Others'],cross_ssc_b['Not Placed'])

print(cross_ssc_b)
cross_hsc_b = pd.crosstab(df['hsc_b'],df['status'])

plt.figure(figsize=(8,8))

plt.xlabel('HSC_B')

plt.ylabel('Count')

plt.bar(['Central','Others'],cross_hsc_b['Placed'])

plt.bar(['Central','Others'],cross_hsc_b['Not Placed'])

print(cross_hsc_b)
cross_hsc_s = pd.crosstab(df['hsc_s'],df['status'])

plt.figure(figsize=(8,8))

plt.xlabel('HSC_S')

plt.ylabel('Count')

plt.bar(['Arts','Commerce','Science'],cross_hsc_s['Placed'])

plt.bar(['Arts','Commerce','Science'],cross_hsc_s['Not Placed'])

print(cross_hsc_s)
cross_degree_t = pd.crosstab(df['degree_t'],df['status'])

plt.figure(figsize=(8,8))

plt.xlabel('DEGREE_T')

plt.ylabel('Count')

plt.bar(['Comm&Mgmt','Others','Sci&Tech'],cross_degree_t['Placed'])

plt.bar(['Comm&Mgmt','Others','Sci&Tech'],cross_degree_t['Not Placed'])

print(cross_degree_t)
cross_workex = pd.crosstab(df['workex'],df['status'])

plt.figure(figsize=(8,8))

plt.xlabel('WORK_EXP')

plt.ylabel('Count')

plt.bar(['No','Yes'],cross_workex['Placed'])

plt.bar(['No','Yes'],cross_workex['Not Placed'])

print(cross_workex)
cross_specialisation = pd.crosstab(df['specialisation'],df['status'])

plt.figure(figsize=(8,8))

plt.xlabel('SPECIALISATION')

plt.ylabel('Count')

plt.bar(['Mkt&Fin','Mkt&HR'],cross_specialisation['Placed'])

plt.bar(['Mkt&Fin','Mkt&HR'],cross_specialisation['Not Placed'])

print(cross_specialisation)
sns.violinplot(x="status", y='ssc_p', data=df)
sns.violinplot(x="status", y='hsc_p', data=df)

sns.violinplot(x="status", y='degree_p', data=df)

sns.violinplot(x="status", y='etest_p', data=df)
sns.violinplot(x="status", y='mba_p', data=df)
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression
LE = LabelEncoder()

df['gender_new'] = LE.fit_transform(df['gender'])

df['ssc_b_new'] = LE.fit_transform(df['ssc_b'])

df['hsc_b_new'] = LE.fit_transform(df['hsc_b'])

df['hsc_s_new'] = LE.fit_transform(df['hsc_s'])

df['workex_new'] = LE.fit_transform(df['workex'])

df['status_new'] = LE.fit_transform(df['status'])



df.drop(['gender','ssc_b','hsc_b','hsc_s','degree_t','workex','specialisation','status','salary'],inplace=True,axis=1)

fig,ax = plt.subplots(figsize=(14,10))

sns.heatmap(df.astype(int).corr(),ax=ax,annot=True,robust=True)
y = df['status_new']

df.drop('status_new',axis=1,inplace=True)
x_train,x_test,y_train,y_test = train_test_split(df,y)

lr = LogisticRegression(max_iter=200)

lr.fit(x_train,y_train)

y_predict=lr.predict(x_test)

lr_score = lr.score(x_test,y_test)

lr_class_report = classification_report(y_test,y_predict)

lr_conf_mat = confusion_matrix(y_test,y_predict)

print(lr_conf_mat)

print(lr_class_report)

print(lr_score)
rf = RandomForestClassifier()

rf.fit(x_train,y_train)

pred = rf.predict(x_test)

rf_conf_matrix =  confusion_matrix(y_test,pred)

rf_class_report = classification_report(y_test,pred)

rf_score = rf.score(x_test,y_test)

print(rf_conf_matrix)

print(rf_class_report)

print(rf_score)


