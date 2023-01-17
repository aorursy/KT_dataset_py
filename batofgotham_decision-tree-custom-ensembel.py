# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
input_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

df = pd.read_csv(input_path+'train.csv')

df_test = pd.read_csv(input_path+'test.csv')
id_test = df_test['id']
df.shape
target = df['target']
target.value_counts()
df.drop(columns=['target'],inplace=True)
import pickle

df_metadata = pickle.load(open('/kaggle/input/picklefile/df_metedata_pickle','rb'))
df_metadata
def preprocessing(df):

    df.replace(to_replace=-1,value=np.nan,inplace=True)

    for col in df.columns:

        #Dropping Insignificant Columns

        if df_metadata.loc[col,'Dropped']:

            df.drop(columns=[col],inplace=True)

            continue

        #Filling Missing Values

        df[col].fillna(df_metadata.loc[col,'Missing'],inplace=True)

        #Changing the datatype of columns

        if (df_metadata.loc[col,'DTypes'] == 'Categorical') or (df_metadata.loc[col,'DTypes'] == 'Ordinal'):

            df[col] = df[col].astype('category')
df.shape
preprocessing(df)

preprocessing(df_test)
df.isnull().sum()
df_test.isnull().sum()
def outlier_processing(df,df_test):

    for col in df.columns:

        if df[col].dtype.name != 'category':

            first_quartile, third_quartile = np.percentile(df[col],[25,75])

            first_percetnile, ninetynine_percentile = np.percentile(df[col],[1,99])

            IQR = third_quartile - first_quartile

            lower_bound = first_quartile - (1.5*IQR)

            upper_bound = third_quartile + (1.5*IQR)

            df[col].loc[df[col]>upper_bound] = ninetynine_percentile

            df_test[col].loc[df_test[col]>upper_bound] = ninetynine_percentile

            df[col].loc[df[col]<lower_bound] = first_percetnile

            df_test[col].loc[df_test[col]<lower_bound] = first_percetnile

        
outlier_processing(df,df_test)
ordinal_columns = [col for col in df.columns if df_metadata.loc[col,'DTypes'] == 'Ordinal' and df[col].nunique() > 2]
categorical_columns_great_2 = [col for col in df.columns if df_metadata.loc[col,'DTypes'] == 'Categorical' and df[col].nunique() > 2]
from sklearn.preprocessing import LabelEncoder

for col in ordinal_columns:

    label_encode = LabelEncoder()

    df[col+'label'] = label_encode.fit_transform(df[col])

    df_test[col+'label'] = label_encode.transform(df_test[col])

    df.drop(columns=[col],inplace=True)

    df_test.drop(columns=[col],inplace=True)
df = pd.get_dummies(df,prefix=col,columns=categorical_columns_great_2,drop_first=True)

df_test = pd.get_dummies(df_test,columns=categorical_columns_great_2,prefix=col,drop_first=True)
df.shape
df_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_train_scale = scaler.fit_transform(df)
df_test_scale = scaler.transform(df_test)
df_train_scale = pd.DataFrame(df_train_scale,columns=df.columns)

df_test_scale = pd.DataFrame(df_test_scale,columns=df_test.columns)
chunks = [df_train_scale,target]

df_train_scale_target = pd.concat(chunks,axis=1)

df_minority = df_train_scale_target.loc[df_train_scale_target['target'] == 1].copy()

df_majority = df_train_scale_target.loc[df_train_scale_target['target'] == 0].copy()
splitted_frame = np.array_split(df_majority, 20)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

pipe = Pipeline([('classifier', DecisionTreeClassifier())])
grid_param ={

        "criterion": ['entropy'],

        "splitter": ['best'],

        "max_depth": [15,25,30,None],

        "min_samples_split": range(39,42),

        "min_samples_leaf": range(40,42)

    }
from sklearn.utils import shuffle

final_ensemble = []

for frames in splitted_frame:

    Glog_reg = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 17),param_grid = grid_param, scoring = 'accuracy', cv=5)

    chunks_temp = [frames,df_minority]

    df_temp_train = shuffle(pd.concat(chunks_temp,axis=0))

    target_train = df_temp_train['target']

    df_temp_train.drop(columns=['target'],inplace=True)

    Glog_reg.fit(df_temp_train,target_train)

    best_model = Glog_reg.best_estimator_

    best_model.fit(df_temp_train,target_train)

    final_ensemble.append(best_model)

    
y_train_pred_proba = 0

y_test_pred_proba = 0

for model in final_ensemble:

    y_train_pred_proba = y_train_pred_proba + model.predict_proba(df_train_scale)[:,1]

    y_test_pred_proba = y_test_pred_proba + model.predict_proba(df_test_scale)[:,1]

y_train_pred_proba = y_train_pred_proba/20

y_test_pred_proba = y_test_pred_proba/20
from sklearn.metrics import roc_curve, roc_auc_score

fpr,tpr,thresold = roc_curve(target,y_train_pred_proba)

auc = roc_auc_score(target,y_train_pred_proba)

plt.figure(figsize=(14,8))

plt.title('Reciever Operating Charactaristics')

plt.plot(fpr,tpr,'b',label = 'AUC = %0.2f' % auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')
auc
submit = pd.DataFrame({'id':id_test,'target':y_test_pred_proba})

submit.to_csv('logreg_porto.csv',index=False) 

submit.head()