
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder
df=pd.read_excel('../input/label-encoding/Label_Encoding.xlsx')
df
##Label Encondin

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Marrige_Status']=le.fit_transform(df['Marrige_Status'])
df
##One Hot Encoding

from sklearn.preprocessing import OneHotEncoder
df=pd.get_dummies(df,columns=['Marrige_Status'])
df
