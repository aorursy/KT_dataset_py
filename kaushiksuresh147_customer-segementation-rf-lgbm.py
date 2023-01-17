

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import missingno as mno
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import itertools
from scipy.stats import spearmanr
from collections import defaultdict
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train=pd.read_csv('../input/customer-segmentation/Train.csv')
test=pd.read_csv('../input/customer-segmentation/Test.csv')

#Joining both the testa and train data frames together 
df=pd.concat([train,test],axis=0)
#Lets look at the data type of the features!
print(df.info())
#Visualize the missing values in the dataframe! 
mno.matrix(df)
df.isnull().sum()
#obtainign the categroical columns alone
catcols = []
for i in df.columns:
  if df[i].dtype == "object":
      catcols.append(i)
catcols     
catcols[:-1]
#Replacing the missing values in the categorical variables as "not_available"
df[catcols[:-1]] = df[catcols[:-1]].fillna("not_available")
df.isnull().sum()
gender_map = {'Female': 1, 'Male': 0}
marriage_map = {'not_available': 99, 'No': 0, 'Yes': 1}
graduate_map = {'not_available': 99, 'No': 0, 'Yes': 1}
profession_map = {'Artist': 0,'Doctor': 1,'Engineer': 2,'Entertainment': 3,'Executive': 4,'Healthcare': 5,
                   'Homemaker': 6,'Lawyer': 7,'Marketing': 8,'not_available': 99}
spending_map = {'Average': 1, 'High': 2, 'Low': 0}
var_map = {'Cat_1': 1,'Cat_2': 2,'Cat_3': 3,'Cat_4': 4,'Cat_5': 5, 'Cat_6': 6, 'Cat_7': 7,'not_available': 99}
target_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Prof+Grad"] = df["Profession"]+"_"+df["Graduated"].astype(str)
df["Prof+Grad"] = le.fit_transform(df['Prof+Grad'].astype(str))

df["Gender"] =  df["Gender"].map(gender_map)
df["Ever_Married"] =  df["Ever_Married"].map(marriage_map)
df["Graduated"] =  df["Graduated"].map(graduate_map)
df["Profession"] =  df["Profession"].map(profession_map)
df["Spending_Score"] =  df["Spending_Score"].map(spending_map)
df["Var_1"] =  df["Var_1"].map(var_map)
df["Segmentation"] =  df["Segmentation"].map(target_map)

# Now lets move with replacing the numeric variables 
df.isnull().sum()
train['Work_Experience'].hist()
print("Median value of Family size feature is:",train['Work_Experience'].median())
train['Family_Size'].hist()
print("Median value of Family size feature is:",train['Family_Size'].median())
df['Work_Experience']=df['Work_Experience'].fillna(train['Work_Experience'].median())
df['Family_Size']=df['Family_Size'].fillna(train['Family_Size'].median())
age_bins=[0,20,40,60,80,100]
age_labels=[ "<=20","21-40","41-60", "61-80",">80"]

df['Age']=pd.cut(df['Age'], bins=age_bins,labels=age_labels)
df['Age'].value_counts()
df['Age']=le.fit_transform(df['Age'])
df
temp = df.groupby(['Age']).agg({'Spending_Score':['count','mean','sum'],
                                   'Work_Experience':['count','sum','min','max','mean'],
                                   'Profession':['min','max'],
                                       'Family_Size':['sum','min','max'],
                                       'Age':['count'],
                                    'Var_1':['count','max','min']})
temp.columns = ['_'.join(x) for x in temp.columns]
df = pd.merge(df,temp,on=['Age'],how='left')
temp = df.groupby(['Profession']).agg({
                                       'Age':['count','sum','min','max']})
temp.columns = ['_Prof_'.join(x) for x in temp.columns]
df = pd.merge(df,temp,on=['Profession'],how='left')
df=df.set_index('ID')
df_corr=df[df.columns[~df.columns.isin(['Segmentation','train_or_test'])]]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(60, 30))
corr = spearmanr(df_corr).correlation
corr_linkage = hierarchy.ward(np.nan_to_num(corr))
dendro = hierarchy.dendrogram(corr_linkage, labels=df_corr.columns, ax=ax1,
                              leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()
fig.savefig('test2png.png', dpi=400)
sns.heatmap(corr)
numvar=[]
for i in np.arange(0.0, 2.1, 0.2):
    cluster_ids = hierarchy.fcluster(corr_linkage, i, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    numvar.append([i,len(selected_features)])
# selecting features based on chosen dendrogram y-axis value
cluster_ids = hierarchy.fcluster(corr_linkage, 1.4, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features = list(np.array(selected_features)+1)
selected_features.append(-1)
selected_features.insert(0,0)
df_corr = df_corr.iloc[:,selected_features[0:-2]]
#df_corr.to_csv('modelreadydf.csv',index=0)
#df_corr = pd.read_csv('modelreadydf.csv')
featured_df=df_corr
featured_df.head(10)
X_train = featured_df.iloc[0:len(train),:]
X_test = featured_df.iloc[len(train):,:]
y_train=train['Segmentation']
y_test=test['Segmentation']
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
pred_X=classifier.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, pred_X)))

pred_rf_baseline = classifier.predict(X_test)
print('Testing accuracy is {}'.format(accuracy_score(y_test, pred_rf_baseline)))
from sklearn.ensemble import RandomForestClassifier

n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 
forest= RandomForestClassifier()

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, y_train)
print("Best depth:",bestF.best_estimator_.get_params()['max_depth'])
print("Best n_estimators:",bestF.best_estimator_.get_params()['n_estimators'])
print("Best min_samples_split:",bestF.best_estimator_.get_params()['min_samples_split'])
print("Best min_samples_leaf:",bestF.best_estimator_.get_params()['min_samples_leaf'])
model_rf = RandomForestClassifier(random_state = 1,
                                  n_estimators = 300,
                                  max_depth = 8, 
                                  min_samples_split = 15,  min_samples_leaf = 1) 
model_rf = model_rf.fit(X_train, y_train)
pred_X = model_rf.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, pred_X)))
pred_rf= model_rf.predict(X_test)
print('Testing accuracy is {}'.format(accuracy_score(y_test, pred_rf)))
pred_rf
import lightgbm as lgbm

lgbm_clf = lgbm.LGBMClassifier(n_estimators=3000, cat_feature = [0,1,2,3,7,8,9,10,11,12,13,14,15,16,17,18], label_gain = [5], num_leaves=8, max_depth=20, 
                               learning_rate=0.01, random_state=42)
lgbm_clf.fit(X_train, y_train)
pred_X = lgbm_clf.predict(X_train)
print('Training accuracy is {}'.format(accuracy_score(y_train, pred_X)))
pred_lgbm= lgbm_clf.predict(X_test)
print('Testing accuracy is {}'.format(accuracy_score(y_test, pred_lgbm)))

pred_lgbm
### LGBM - RandomCV
parameters = {'n_estimators':[1000,2000,3000,5000,10000], 
             'num_leaves':[15,25,30],
             'learning_rate':[0.001,0.003,0.01,0.03],
             'max_depth':[8,12,18,25],
             'min_data_in_leaf':[40,50,60],
             'reg_alpha':[i for i in np.arange(1,2,0.2)],
             'reg_lambda':[i for i in np.arange(1,2,0.2)],
             'subsample':[0.5,0.7,1]}
lgbm_grid = RandomizedSearchCV(estimator=lgbm_clf1, param_grid=parameters, n_jobs=-1, cv = 5, scoring='accuracy', verbose=10)
#Implement the lgbm_grid in the above lgbm baseline model to obtain better accuracy!
output=pd.DataFrame(columns=['ID','Segmentation'])
output['ID']=test['ID']
output['Segmentation']=pred_lgbm
output.to_csv('output.csv',index=False)
