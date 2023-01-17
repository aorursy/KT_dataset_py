%config IPCompleter.greedy=True
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

from statsmodels.formula.api import logit

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
heart_disease=pd.read_csv('../input/heart.csv')
#separate the column by numerical and category in list for future use

cat_column=['sex','cp','fbs','restecg','exang','slope','ca','thal']

num_column=['age','trestbps','chol','thalach','oldpeak']

#checking if the data's value matching the description

heart_disease.cp.unique()

heart_disease.fbs.unique()

heart_disease.restecg.unique()

heart_disease.exang.unique()

heart_disease.slope.unique()

heart_disease.ca.unique()

heart_disease.thal.unique()
sns.catplot('target',data=heart_disease,kind='count',height=5,aspect=.8)
sns.catplot(x='age',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)

sns.catplot(x='age',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot('sex',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
heart_disease.groupby(['sex','target']).describe()
sns.catplot('cp',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot(x='thalach',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)

sns.catplot(x='thalach',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot('exang',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot(x='oldpeak',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)
sns.catplot('slope',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot('ca',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot('thal',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
sns.catplot(x='trestbps',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)

sns.catplot(x='chol',y='target',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)

sns.catplot(x='chol',data=heart_disease,kind='violin',orient='h',height=5,dodge=True)

sns.catplot('fbs',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)

sns.catplot('restecg',col='target',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)
heart_disease[['age','trestbps','chol','thalach','oldpeak']].describe()
#Normalize Data 

scaler=StandardScaler()

numerical=pd.DataFrame(scaler.fit_transform(heart_disease[num_column]),columns=num_column,index=heart_disease.index)

heart_disease_transform=heart_disease.copy(deep=True)

heart_disease_transform[num_column]=numerical[num_column]

heart_disease_transform.head()
# adding intercept term for logit regression

heart_disease_transform['intercept']=1.0

#rearranging the column of data set , put target and intercept in front will make lifes better 

heart_disease_transform.head()

cols=num_column.copy()

cols.insert(0,'target')

cols.insert(1,'intercept')

cols.extend(cat_column)

heart_disease_transform=heart_disease_transform.reindex(columns=cols)

#one hot encoding for cat variable ,with no drop first 

heart_disease_transform_nodrop=pd.get_dummies(heart_disease_transform,prefix_sep='_',columns=cat_column)

#one hot encoding for cat variable ,with drop first 

heart_disease_transform=pd.get_dummies(heart_disease_transform,prefix_sep='_',columns=cat_column,drop_first=True)
heart_disease_transform.head()

# Create correlation map on the transformed data set( one hot encode drop first = true)  

corr=heart_disease_transform.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr,mask=mask, cmap=cmap,linewidths=.5 ,center=0,square=True, cbar_kws={"shrink": 1})

# Create correlation map on the transformed data set( one hot encode drop first = false)
corr2=heart_disease_transform_nodrop.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr2, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(15, 10))

sns.heatmap(corr2,mask=mask, cmap=cmap,linewidths=.5 ,center=0,square=True, cbar_kws={"shrink": 1})
#prepare training and testing dataset for sklearn logit

X_train, X_test, y_train, y_test = train_test_split(heart_disease_transform.drop('target',axis=1), 

                                                    heart_disease_transform['target'], test_size=0.30, random_state=101)

#sklearn Logistic instance

sklogit=LogisticRegression(solver='lbfgs')

sklogitv2=LogisticRegression(solver='lbfgs')

sklogitv3=LogisticRegression(solver='lbfgs')

sklogitv4=LogisticRegression(solver='lbfgs')

sklogitv5=LogisticRegression(solver='lbfgs')
#statsmodels for summary report on Full Model

logitv1=sm.Logit(heart_disease_transform['target'],heart_disease_transform[heart_disease_transform.columns[1:]])

result=logitv1.fit()
result.summary2()
# Full Model on sklearn

sklogit.fit(X_train,y_train)

sk_predict=sklogit.predict(X_test)

sklogit.score(X_test,y_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, sk_predict))
#To prepare the dataset of reduced model,drop variable for those P-value is >0.05 from the summary report of statsmodels 

# W

heart_disease_transform_v2=pd.DataFrame(heart_disease_transform[['target','intercept','trestbps','sex_1','cp_1','cp_2','cp_3','ca_1','ca_2','ca_3','ca_4']],

                                        columns=['target','intercept','trestbps','sex_1','cp_1','cp_2','cp_3','ca_1','ca_2','ca_3','ca_4'],index=heart_disease_transform.index)
#statsmodels for summary report on Reduced Model

logitv2=sm.Logit(heart_disease_transform_v2['target'],heart_disease_transform_v2[heart_disease_transform_v2.columns[1:]])

resultv2=logitv2.fit()
resultv2.summary2()
#prepare training and testing dataset for sklearn on reduced model

X_trainv2, X_testv2, y_trainv2, y_testv2 = train_test_split(heart_disease_transform_v2.drop('target',axis=1),

                                                            heart_disease_transform_v2['target'], test_size=0.30, random_state=101)

# Reduced Model on sklearn

sklogitv2.fit(X_trainv2,y_trainv2)

sk_predictv2=sklogitv2.predict(X_testv2)

sklogitv2.score(X_testv2,y_testv2)

from sklearn.metrics import classification_report

print(classification_report(y_testv2, sk_predictv2))

# exp(coefficient of the model) 

np.exp(resultv2.params)
sns.catplot('thal',col='target',row='sex',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)

sns.catplot('thal',col='sex',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)

heart_disease.groupby(['target','sex','thal']).describe()
sns.catplot('ca',col='target',row='thal',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)

sns.catplot('cp',col='target',row='thal',data=heart_disease,kind='count',height=5,aspect=.8,dodge=True)