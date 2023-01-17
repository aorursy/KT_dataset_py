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
Train=pd.read_csv('/kaggle/input/zschall/input_train_v1.csv')

Test=pd.read_csv('/kaggle/input/zschall/input_test_v1.csv')
columns = Train.columns

percent_missing =Train.isnull().sum() * 100 / len(Train)

missing_value_df = pd.DataFrame({'column_name': columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True)

missing_value_df
df=Train[['Event_name']]

df=df.join(df['Event_name'].str.split('_', 2, expand=True).rename(columns={0:'Event_type',1:'sub_type',2:'sub_type_2'}))

df=df[['Event_type','sub_type']]

df=pd.DataFrame(df)

Train=pd.concat((Train,df),axis=1)
Train
indices=list(df.loc[pd.isna(Train["lab_result_numeric"]), :].index)
groupmean=pd.DataFrame(Train.groupby('Event_name')['lab_result_numeric'].mean(numeric_only=True))

groupmean.reset_index(level=0, inplace=True)

groupmean
Train.iloc[0]
groupmean.loc[groupmean['Event_name']=='dx_2m_1'].iloc[0][1]
Train.loc[pd.isna(Train['lab_result_numeric'])]
for i in range(len(Train)):

    if(pd.isna(Train.iloc[i][3])):

        Train.iloc[i][3]=groupmean.loc[groupmean['Event_name']=='dx_2m_1'].iloc[0][1]
df1=Test[['Event_name']]

df1=df1.join(df1['Event_name'].str.split('_', 2, expand=True).rename(columns={0:'Event_type',1:'sub_type',2:'sub_type_2'}))

df1=df1[['Event_type','sub_type',]]

df1=pd.DataFrame(df1)

Test=pd.concat((Test,df1),axis=1)
Train=Train.drop(['pat_iden','Event_name','Event_desc'],axis=1)

Test=Test.drop(['pat_iden','Event_name','Event_desc','Unnamed: 5'],axis=1)
Train
#Label_encoding

# Import label encoder 

from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

Train['Event_type']=label_encoder.fit_transform(Train['Event_type'])

Train['sub_type']=label_encoder.fit_transform(Train['sub_type'])



Test['Event_type']=label_encoder.fit_transform(Test['Event_type'])

Test['sub_type']=label_encoder.fit_transform(Test['sub_type'])
X=Train.drop(['y_flag'],axis=1)

y=Train['y_flag']

from sklearn.model_selection import cross_val_score, train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=7)
Test
import xgboost as xgb

from sklearn.metrics import accuracy_score

model=xgb.XGBRegressor()

model.fit(X_train,y_train)

pred=model.predict(Test)

y=pd.DataFrame(pred)

y.rename(columns={0:pred_prob_1})