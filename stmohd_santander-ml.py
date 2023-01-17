# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
test=pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
train.head()
import seaborn as sns

plt.figure(figsize=(10,4))

sns.boxplot(x=train.var_0)



plt.figure(figsize=(10,4))

sns.boxplot(x=train.var_3)
print(train.isna().any().sum(),test.isna().any().sum())
# Outlier detection 

from collections import Counter

def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

#         print(Q1 )

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

#         print(Q3,type(Q3))

        # Interquartile range (IQR)

        IQR = Q3 - Q1

#         print(IQR,type(IQR))

        # outlier step

        outlier_step = 1.5 * IQR

#         print(outlier_step,"step")

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    print(multiple_outliers)

    return multiple_outliers   



Outliers_to_drop = detect_outliers(train,2,train.drop(['ID_code','target'],axis=1).columns.tolist())
# train.iloc[Outliers_to_drop]
train =train.drop(Outliers_to_drop)
import scipy.stats as stats

corrltd =[]

for  i in train.select_dtypes(exclude=['object']).columns:

#     print(i)

    val,pval = stats.pearsonr(train[i],train['target'])

    if abs(pval)<0.05:

        corrltd.append(i)

print(corrltd,len(corrltd))
corrltd.append('ID_code')
train= train[corrltd]
train.head()
from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
train.target.value_counts()
x=train.drop(['target','ID_code'],axis=1)

y=train.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= .3,random_state= 42)
lr =LogisticRegression(random_state=102)

dc =DecisionTreeClassifier(random_state=102)

rf =RandomForestClassifier(random_state=102)

gn =GaussianNB()

knn =KNeighborsClassifier()

mlp =MLPClassifier(random_state=102)
lr.fit(x_train, y_train)
import sklearn.metrics as metrics
pred =lr.predict(x_test)
print(metrics.classification_report(y_test,pred))
metrics.roc_auc_score(y_test,pred)
dc.fit(x_train, y_train)

pred =dc.predict(x_test)

metrics.roc_auc_score(y_test,pred)
rf.fit(x_train, y_train)

pred =rf.predict(x_test)

metrics.roc_auc_score(y_test,pred)
gn.fit(x_train, y_train)

pred =gn.predict(x_test)

metrics.roc_auc_score(y_test,pred)
# knn.fit(x_train, y_train)

# pred =knn.predict(x_test)

# metrics.roc_auc_score(y_test,pred)
mlp.fit(x_train, y_train)

pred =mlp.predict(x_test)

metrics.roc_auc_score(y_test,pred)
# Trying boosting to accuracy

from sklearn.ensemble import GradientBoostingRegressor

gbcl = GradientBoostingRegressor(n_estimators = 300)#default is decision tree

gbcl = gbcl.fit(x_train,y_train)

pred =gbcl.predict(x_test)

metrics.roc_auc_score(y_test,pred)
# list(x_train.columns)
columns_v= list(x_train.columns)

y_pred = gbcl.predict(test[columns_v])



submission = pd.DataFrame({

    "ID_code": test.ID_code, 

    "target": y_pred

})

submission.to_csv('submission_santansergcbl.csv', index=False)
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(random_state=0, sampling_strategy = 'auto')

eec.fit(x_train, y_train) 



pred =eec.predict(x_test)

metrics.roc_auc_score(y_test,pred)

predeec =eec.predict(test[columns_v])
# import time

# ts = time.time()



# model = XGBClassifier(

#     max_depth=8,

#     n_estimators=800,

#     scale_pos_weight=1,

#     min_child_weight=300, 

#     colsample_bytree=0.8, 

#     subsample=0.8, 

#     reg_alpha = 0.3,

#     seed=42)



# model.fit(

#     x_train, 

#     y_train, 

#     eval_metric="auc", 

#     eval_set=[(x_train, y_train), (x_test, y_test)], 

#     verbose=True, 

#     early_stopping_rounds = 10)



# (time.time() - ts)/60
# 3593.6/60
# pred =model.predict(x_test)

# metrics.roc_auc_score(y_test,pred)
# xg_pred = model.predict(test[columns_v])

# metrics.roc_auc_score(y_test,pred)

# submission = pd.DataFrame({

#     "ID_code": test.ID_code, 

#     "target": y_pred

# })

# submission.to_csv('xgb_submission_santanser_xg_pred.csv', index=False)
y_pred = gbcl.predict(test[columns_v])

len(y_pred)
len(y_pred),len(predeec)


submission = pd.DataFrame({

    "ID_code": test.ID_code, 

    "target": (y_pred+predeec)/2

})

submission.to_csv('xgb_submission_santanser_average.csv', index=False)