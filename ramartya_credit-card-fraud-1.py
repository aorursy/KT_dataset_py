# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data=pd.read_csv('../input/creditcard.csv')



# Any results you write to the current directory are saved as output.
print(data.head())

print(data.columns)

#print(data.iloc[30])

#print(data.where(data['Class']>0).count())

print(data.describe())
data=data.sample(frac=1)#shuffling the data
#we need to scale the data

from sklearn.preprocessing import StandardScaler, RobustScaler

#RObust Scaler is more robust to outliers

rob_sca=RobustScaler()

data['sc_time']=rob_sca.fit_transform(data['Time'].values.reshape(-1,1))

data['sc_amo']=rob_sca.fit_transform(data['Amount'].values.reshape(-1,1))
data.drop(['Time','Amount'], axis=1, inplace=True)
sc_time=data['sc_time']

sc_amo=data['sc_amo']

data.drop(['sc_time','sc_amo'], axis=1,inplace=True)

data.insert(0,'sc_time',sc_time)

data.insert(1,'sc_amo',sc_amo)
print(data.columns)

print("non fraud:",round((data['Class'].value_counts()[0]/len(data))*100,2),"fraud:",round((data['Class'].value_counts()[1]/len(data))*100,2))
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

x=data.iloc[:,0:-1]

y=data.iloc[:,-1]

sss=StratifiedKFold(n_splits=5,shuffle=False,random_state=None)#Provides train/test indices to split data in train/test sets

"""

#this loop provides 5 different combinations of trainign and test set

for train_indices, test_indices in sss.split(x,y):

    print("train indices:",train_indices, "test_indices:",test_indices)

    x_tr, x_val=x.iloc[train_indices],x.iloc[test_indices]

    y_tr, y_val=y.iloc[train_indices],y.iloc[test_indices]

    

"""



x_tr, x_val, y_tr, y_val=train_test_split(x,y,test_size=0.3, random_state=1)

x_tr=x_tr.values#converting dataframe to array

x_val=x_val.values

y_tr=y_tr.values

y_val=y_val.values



#checkng if training and test set have similar distribtions or not

train_unique_label, train_counts_label = np.unique(y_tr, return_counts=True)

#by default, unique returns an array of unique elements in the array. when return_counts=True, it also returns an array

#containg the no. of times each unique element occurs

test_unique_label, test_counts_label = np.unique(y_val, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(y_tr))

print(test_counts_label/ len(y_val))



fraud_data=data.loc[data['Class']==1]

non_fraud_data=data.loc[data['Class']==0][:len(fraud_data)]# making both the equall in new data

new_data=pd.concat([fraud_data, non_fraud_data])

new_data=new_data.sample(frac=1)

print(new_data.columns)
import matplotlib.pyplot as plt

import seaborn as sns



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

corr = data.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)



sub_sample_corr = new_data.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':23})

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

plt.show()



print(sub_sample_corr)
pos_corr=sub_sample_corr.index[sub_sample_corr['Class'] >0.30].tolist()#these eatures have strong +ve correlation

neg_corr=sub_sample_corr.index[sub_sample_corr['Class'] <-0.50].tolist()#these eatures have strong -ve correlation
f,axes =plt.subplots(ncols=4,figsize=(20,4))

sns.boxplot(x='Class', y='V2',data=new_data, ax=axes[0])

axes[0].set_title("V2 vs Class")



sns.boxplot(x='Class', y='V4',data=new_data, ax=axes[1])

axes[1].set_title("V4 vs Class")



sns.boxplot(x='Class', y='V11',data=new_data, ax=axes[2])

axes[2].set_title("V11 vs Class")



sns.boxplot(x='Class', y='V19',data=new_data, ax=axes[3])

axes[3].set_title("V19 vs Class")



plt.show()
f,axes =plt.subplots(ncols=4,figsize=(20,4))



sns.boxplot(x='Class', y='V10',data=new_data, ax=axes[0])

axes[0].set_title("V10 vs Class")



sns.boxplot(x='Class', y='V12',data=new_data, ax=axes[1])

axes[1].set_title("V12 vs Class")



sns.boxplot(x='Class', y='V14',data=new_data, ax=axes[2])

axes[2].set_title("V14 vs Class")



sns.boxplot(x='Class', y='V17',data=new_data, ax=axes[3])

axes[3].set_title("V17 vs Class")



plt.show()
from scipy.stats import norm

f,axes=plt.subplots(ncols=3,figsize=(20,4))



sns.distplot(new_data['V10'].loc[new_data['Class']==0].values, ax=axes[0],fit=norm)

axes[0].set_title("V10")



sns.distplot(new_data['V12'].loc[new_data['Class']==0].values, ax=axes[1],fit=norm,color='#FB8861')

axes[1].set_title("V12")



sns.distplot(new_data['V14'].loc[new_data['Class']==0].values, ax=axes[2],fit=norm,color='#0B8861')

axes[2].set_title("V14")



plt.show()



#from the graph, its clear that V14 has distribution clos to normal distribution
# ---------------------------------------------------->removing outliers from v14 of fraud transactions

V14_fraud=new_data['V14'].loc[new_data['Class']==1].values

q25,q75=np.percentile(V14_fraud,25),np.percentile(V14_fraud,75)

itr=q75-q25

print('Inter quartle range: {}'.format(itr))

upp_thre=q75+1.5*itr

low_thre=q25-1.5*itr

print('upper threshod: {} | lower threshold:{}'.format(upp_thre,low_thre))



outliers_v14=[x for x in V14_fraud if x>upp_thre or x<low_thre]

new_data = new_data.drop(new_data[(new_data['V14'] > upp_thre) | (new_data['V14'] < low_thre)].index)

print('No of outliers: {}'.format(len(outliers_v14)))

print("-"*100)



# ---------------------------------------------------->removing outliers from v12 of fraud transactions

V12_fraud=new_data['V12'].loc[new_data['Class']==1].values

q25_v12,q75_v12=np.percentile(V12_fraud,25),np.percentile(V12_fraud,75)

itr_v12=q75_v12-q25_v12

print('Inter quartle range: {}'.format(itr_v12))

upp_thre_v12=q75_v12+1.5*itr_v12

low_thre_v12=q25_v12-1.5*itr_v12

print('upper threshod: {} | lower threshold:{}'.format(upp_thre_v12,low_thre_v12))



outliers_v12=[x for x in V14_fraud if x>upp_thre_v12 or x<low_thre_v12]

new_data = new_data.drop(new_data[(new_data['V12'] > upp_thre_v12) | (new_data['V12'] < low_thre_v12)].index)

print('No of outliers: {}'.format(len(outliers_v12)))

print("-"*100)

# ---------------------------------------------------->removing outliers from v14 of fraud transactions

V10_fraud=new_data['V10'].loc[new_data['Class']==1].values

q25_v10,q75_v10=np.percentile(V10_fraud,25),np.percentile(V10_fraud,75)

itr_v10=q75_v10-q25_v10

print('Inter quartle range: {}'.format(itr_v10))

upp_thre_v10=q75_v10+1.5*itr_v10

low_thre_v10=q25_v10-1.5*itr_v10

print('upper threshod: {} | lower threshold:{}'.format(upp_thre_v10,low_thre_v10))



outliers_v10=[x for x in V14_fraud if x>upp_thre_v10 or x<low_thre_v10]

new_data = new_data.drop(new_data[(new_data['V12'] > upp_thre_v10) | (new_data['V10'] < low_thre_v10)].index)

print('No of outliers: {}'.format(len(outliers_v10)))

print("-"*100)
f, axes=plt.subplots(ncols=3, figsize=(20,3))



sns.boxplot(x='Class', y='V14',data=new_data, ax=axes[0])

axes[0].set_title("V14")



sns.boxplot(data=new_data, x='Class', y='V12', ax=axes[1])

axes[1].set_title('V12')



sns.boxplot(data=new_data, x='Class', y='V10', ax=axes[2])

axes[2].set_title('V10')
x_new=new_data.iloc[:,0:-1]

y_new=new_data.iloc[:,-1]



import time

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.manifold import TSNE



# T-SNE Implementation

t0=time.time()

x_new_tsne=TSNE(n_components=2,random_state=42).fit_transform(x_new.values)

t1=time.time()

print('time taken by TSNE: {}'.format(t1-t0))



# PCA Implementation

t2=time.time()

x_new_pca=PCA(n_components=2,random_state=42).fit_transform(x_new.values)

t3=time.time()

print('time taken by PCA: {}'.format(t3-t2))





# TruncatedSVD

t4=time.time()

x_new_tsvd=TruncatedSVD(n_components=2,algorithm='randomized',random_state=42).fit_transform(x_new.values)

t5=time.time()

print('time taken by TruncatedSVD: {}'.format(t5-t4))
import matplotlib.patches as mpatches

f, axes=plt.subplots(ncols=3, figsize=(20,6))

f.suptitle("Clusters using Dimensionality Reduction\n")



blue=mpatches.Patch(color='#0A0AFF',label="Fraud" )

red=mpatches.Patch(color='#AF0000',label="Non Fraud" )



# t-SNE scatter plot

axes[0].scatter(x_new_tsne[:,0], x_new_tsne[:,1], c=(y_new == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

axes[0].scatter(x_new_tsne[:,0], x_new_tsne[:,1], c=(y_new == 1), cmap='coolwarm', label='Fraud', linewidths=2)

axes[0].set_title('t-SNE', fontsize=14)



axes[0].grid(True)



axes[0].legend(handles=[blue, red])





# PCA scatter plot

axes[1].scatter(x_new_pca[:,0], x_new_pca[:,1], c=(y_new == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

axes[1].scatter(x_new_pca[:,0], x_new_pca[:,1], c=(y_new == 1), cmap='coolwarm', label='Fraud', linewidths=2)

axes[1].set_title('PCA', fontsize=14)



axes[1].grid(True)



axes[1].legend(handles=[blue, red])







# TruncatedSVD scatter plot

axes[2].scatter(x_new_tsvd[:,0], x_new_tsvd[:,1], c=(y_new == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

axes[2].scatter(x_new_tsvd[:,0], x_new_tsvd[:,1], c=(y_new == 1), cmap='coolwarm', label='Fraud', linewidths=2)

axes[2].set_title('Trucated SVD', fontsize=14)



axes[2].grid(True)



axes[2].legend(handles=[blue, red])

                 

                 
x_new_tr,x_new_val, y_new_tr, y_new_val=train_test_split(x_new,y_new, test_size=0.2)

#converting values into array

x_new_tr=x_new_tr.values

y_new_tr=y_new_tr.values

x_new_val=x_new_val.values

y_new_val=y_new_val.values
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



Classif={"Logistic Regression": LogisticRegression(), "Knearest":KNeighborsClassifier(),"SVC": SVC(), "DecisionTree" :DecisionTreeClassifier}

       
from sklearn.model_selection import cross_val_score

for key in Classif:

    Classif[key].fit(x_new_tr,y_new_tr)

    print(key," " ,cross_val_score(Classif[key],x_new_val,y_new_val,cv=5).mean())

    
from sklearn.model_selection import GridSearchCV



log_param={'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}

grid_log_cv=GridSearchCV(LogisticRegression(),log_param)

grid_log_cv.fit(x_tr,y_tr)

log_reg_best=grid_log_cv.best_estimator_



knear_param={'n_neighbors':list(range(2,5,1)),'algorithm':['auto','ball_tree','kd_tree','brute']}

grid_knear_cv=GridSearchCV(KNeighborsClassifier(),knear_param)

grid_knear_cv.fit(x_tr,y_tr)

knear_reg_best=grid_knear_cv.best_estimator_



svc_param={'kernel':['linear','sigmoid','poly','rbf'],'C':[0.5,0.7,0.9,1,1.2,1.4]}

grid_svc_cv=GridSearchCV(SVC(),svc_param)

grid_svc_cv.fit(x_tr,y_tr)

log_svc_best=grid_svc_cv.best_estimator_





dtc_param={'criterion':['mini','entropy'],'max_depth':list(range(2,5,1)),'min_samples_leaf':list(range(4,7,1))}

grid_dtc_cv=GridSearchCV(DecisionTreeClassifier(),dtc_param)

grid_dtc_cv.fit(x_tr,y_tr)

log_dtc_best=grid_dtc_cv.best_estimator_
log_reg_score=cross_val_score(grid_log_cv,x_new_tr,y_new_tr,cv=5).mean()

print("Logistic Regression:", log_reg_score)



kn_score=cross_val_score(grid_knear_cv,y_new_tr,cv=5).mean()

print("KNeighbor CLassifie:", knear_score)



svc_score=cross_val_score(grid_svc_cv,x_new_tr,y_new_tr,cv=5).mean()

print("Logistic Regression:", svc_score)



dtc_score=cross_val_score(grid_dtc_cv,x_new_tr,y_new_tr,cv=5).mean()

print("Logistic Regression:", dtc_score)


from imblearn.over_sampling import SMOTE

sm=SMOTE(sampling_strategy="not majority", random_state=2)

x_tr, y_tr=sm.fit_sample(x_tr,y_tr.values)
x_tr=pd.DataFrame(x_tr)

y_tr=pd.DataFrame(y_tr)
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)

model.fit(x_tr, y_tr)
y_pred_prob=model.predict_proba(x_val)
print(y_val.shape)

print(y_pred_prob.shape)

y_pred_prob=y_pred_prob[:,1]

print(y_pred_prob)

from sklearn.metrics import precision_recall_curve

precision, recall,threshold=precision_recall_curve(y_val,y_pred_prob)
table=np.zeros((precision.shape[0],2))#table is 71x2

table[:,0]=precision

table[:,1]=recall

sorted(table, key=lambda x: x[0], reverse=True)

print(type(table))
precision=table[:,0]

recall=table[:,1]
from sklearn.metrics import auc

print(auc(precision, recall, reorder=True))