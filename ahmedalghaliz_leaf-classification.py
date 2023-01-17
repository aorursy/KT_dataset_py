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
train_data= pd.read_csv("../input/leaf-classification/train.csv.zip",index_col=False)

test_data= pd.read_csv("../input/leaf-classification/test.csv.zip", index_col=False)

train_data.head()
#obj = data.isnull().sum()

#for key, value in obj.iteritems():

#    print(key,":",value)
#obj_2 = test_data.isnull().sum()

#for key, value in obj_2.iteritems():

#    print(key,":",value)
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

le=encoder.fit(train_data.species)

labels=le.transform(train_data.species)

classes=list(le.classes_)
train_data=train_data.drop(['id','species'],axis=1)

test_id=test_data.id

test_data=test_data.drop(['id'],axis=1)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_data,labels,test_size=.2,shuffle=True,stratify=labels)
from sklearn.ensemble import ExtraTreesClassifier

lda = ExtraTreesClassifier(bootstrap=False,

                           ccp_alpha=0.0,

                           class_weight=None,

                           criterion='gini',

                           max_depth=60,

                           max_features='sqrt',

                           max_leaf_nodes=None,

                           max_samples=None,

                           min_impurity_decrease=0.0,

                           min_impurity_split=None,

                           min_samples_leaf=2,

                           min_samples_split=10,

                           min_weight_fraction_leaf=0.0,

                           n_estimators=195,

                           n_jobs=None, oob_score=False,

                           random_state=6713, verbose=0,

                           warm_start=False)



lda.fit(X_train,y_train)
lda.score(X_train,y_train), lda.score(X_test,y_test)
predicted=lda.predict_proba(test_data)



sample_df=pd.read_csv('../input/leaf-classification/sample_submission.csv.zip',index_col=False)

sample_df.head(2)



df_sub=pd.DataFrame(predicted,columns=sample_df.columns[1:])

df_sub.head(2)
df_sub1=pd.DataFrame(test_id)

df_sub1.head(2)
final_sub=pd.concat([df_sub1,df_sub],axis=1)

final_sub.to_csv('sample_submission.csv',index=False)