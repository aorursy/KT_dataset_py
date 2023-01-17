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
df=pd.read_csv('../input/train.csv')

y=df.iloc[:,[5]]

X=df.drop(['scalar_coupling_constant','type','molecule_name','id'],axis=1)

df_type=pd.get_dummies(df['type'])

df_type=df_type.drop(['1JHC'],axis=1)

df_type.head()

X=pd.concat([df_type,X],axis=1)

X1=np.array(X)

y1=np.array(y)

from sklearn.linear_model import LinearRegression

lg=LinearRegression()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3)

lg.fit(X_train,y_train)

r2_score=lg.score(X_test,y_test)

print(r2_score)
df_test=pd.read_csv('../input/test.csv')

X_test=df_test.drop(['id','molecule_name','type'],axis=1)

type_test=pd.get_dummies(df_test['type'])



type_test=type_test.drop(['1JHC'],axis=1)

X_test_final=pd.concat([type_test,X_test],axis=1)

X_test_arr=np.array(X_test_final)

y_test_arr=lg.predict(X_test_arr)

id_df=df_test.drop(['molecule_name','type','atom_index_0','atom_index_1'],axis=1)

y_df=pd.DataFrame(y_test_arr)

final_df=pd.concat([id_df,y_df],axis=1)

final_df.columns=['id','scalar_coupling_constant']



final_df.to_csv('mycsvfile.csv',index=False)
