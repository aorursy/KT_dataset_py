import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing



sns.set()

print(os.listdir("../input"))
churn_data = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv',

                         index_col='RowNumber')
churn_data.info()
churn_data.head()
churn_data.describe()
churn_data.CreditScore.value_counts()
churn_data.CreditScore.isna().any()
churn_data.drop(labels=['CustomerId','Surname'],

                axis=1,

                inplace=True)
churn_data.head()
churn_data.Geography.value_counts(dropna=False)
churn_data.Gender.value_counts(dropna=False)
churn_data_cleaned = pd.get_dummies(churn_data, 

                                    prefix=['Geo','Gen'], 

                                    prefix_sep='_',

                                    dummy_na=False, 

                                    columns=['Geography','Gender'],

                                    sparse=False,

                                    drop_first=False,

                                    dtype=int) 
churn_data_cleaned
churn_data_cleaned.hist(bins=10,

                        figsize=(20,20),

                        xrot=30)
labels=churn_data_cleaned.columns

print(labels)

scaler=preprocessing.StandardScaler()

scaled_churn_data_cleaned=scaler.fit_transform(churn_data_cleaned)
scaled_churn_data_cleaned=pd.DataFrame(scaled_churn_data_cleaned)

scaled_churn_data_cleaned.columns=labels
scaled_churn_data_cleaned.hist(bins=10,

                               figsize=(20,20),

                               xrot=30)
fig,ax = plt.subplots(1,1,figsize=(20,20))

for i in scaled_churn_data_cleaned.columns:

    sns.kdeplot(scaled_churn_data_cleaned[i],

                 label=[i],

                 bw=1.5,

                 ax=ax)
corr=scaled_churn_data_cleaned.corr()
fig,ax=plt.subplots(1,1,figsize=(20,10))

sns.heatmap(corr,

            annot=True,

            cmap='RdYlGn',

            ax=ax)
nr=7

nc=2

fig,ax=plt.subplots(nrows=nr,ncols=nc,figsize=(20,20))

i=0

for j in range(nr):

    for k in range(nc):

        axes=ax[j,k]

        

        sns.boxplot(x=scaled_churn_data_cleaned['Exited'],

                    y=scaled_churn_data_cleaned.iloc[:,i],

                    ax=axes)

        i+=1
scaled_churn_data_cleaned=scaled_churn_data_cleaned.drop('Exited',

                                                         axis=1)
scaled_churn_data_cleaned.columns
from sklearn.decomposition import PCA



n_comp = 2

pca=PCA(n_components=n_comp)

principal_components=pca.fit_transform(scaled_churn_data_cleaned)

len(principal_components)
pc_df=pd.DataFrame(principal_components,

                  columns=['principal_components_%s'%(i+1) for i in range(n_comp)],

                  index=range(1,len(principal_components)+1))

print(pc_df)
input_components=pc_df

output_components=churn_data.Exited

print(input_components.shape,output_components.shape)

final_df=pd.concat([input_components,output_components],axis=1)
fig,ax=plt.subplots(1,1,figsize=(20,20))

ax.set_xlabel('principal_components_1',fontsize=20)

ax.set_ylabel('principal_components_2',fontsize=20)

ax.set_title('Customers Exited on PC1 & PC2',fontsize=20)



Targets=[0,1]

colors=['r','k']



for target,color in zip(Targets,colors):

    index_no_target=final_df['Exited']==target

    ax.scatter(final_df.loc[index_no_target,'principal_components_1'],

               final_df.loc[index_no_target,'principal_components_2'],

              c=color)

    ax.legend(Targets)

    ax.grid()
pca.explained_variance_ratio_
n_comp=10

pca_10=PCA(n_components=n_comp)

pca10_comp=pca_10.fit_transform(scaled_churn_data_cleaned)

df_PCA_10=pd.DataFrame(pca10_comp,

                       columns=['Principal_component_%s'%(i+1) for i in range(n_comp)],

                      index=range(1,len(pca10_comp)+1))

print(df_PCA_10)
sum(pca_10.explained_variance_ratio_)
#Test Train split of the datdset

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(df_PCA_10,

                                               output_components,

                                               test_size=0.4,

                                               random_state=0)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report

from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve,auc,log_loss



model=LogisticRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

y_pred_proba=model.predict_proba(x_test)[:, 1]

[fpr,tpr,thr]=roc_curve(y_test,y_pred_proba)



print('Train/Test split results:')

print(model.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))

print(model.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))

print(model.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

print(model.__class__.__name__+" score is  %.2f" % model.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



model=DecisionTreeClassifier(random_state=0)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)



score=model.score(x_test,y_test)

print(score)
cross_val_score(model,x_train,y_train,cv=10)
from sklearn.ensemble import RandomForestClassifier



model=RandomForestClassifier(n_estimators=100,

                            bootstrap=True,

                            max_features='sqrt')

model.fit(x_train,y_train)

y_pred=model.predict(x_train)

print(model.score(x_test,y_test))