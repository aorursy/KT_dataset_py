import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('ga_map_compact_all_58K.CSV', skiprows=[0])

data_iv_x= data.OpCode
data_dv_y = data['Cd-SSN']

print('Train columns with null values:\n', data_iv_x.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_dv_y.isnull().sum())
print("-"*10)

data_iv_x.describe()

full_data= data.loc[:,['OpCode','Cd-SSN']]
full_data.describe()

# Remove all NULLS in the training column
full_data["OpCode"]=full_data["OpCode"].fillna(' Send authentication info')

full_data.describe()

#categorising data -iv
OpCode = pd.get_dummies( full_data.OpCode , prefix='OpCode' )
OpCode.head()
OpCode.drop(['OpCode_Send authentication info'], axis=1, inplace=True)
OpCode.head()
full_data = full_data.join(OpCode)

full_data.describe()

#categorising data - dv
Cd_SSN = pd.get_dummies( full_data["Cd-SSN"] , prefix='SSN' )
Cd_SSN.head()
Cd_SSN.drop(['SSN_Home location register'], axis=1, inplace=True)
Cd_SSN.head()
full_data = full_data.join(Cd_SSN)

full_data.describe()
