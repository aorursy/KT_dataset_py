
# coding: utf-8

# In[108]:

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# In[187]:

df = pd.read_csv('../input/train.csv')
tdf = pd.read_csv('../input/test.csv')

# In[191]:

catdf = df.select_dtypes(['object'])
tcatdf = tdf.select_dtypes(['object'])
catdf.fillna("NA",inplace=True)
tcatdf.fillna("NA",inplace=True)

v22df = catdf.loc[:,['v22']]
tv22df = tcatdf.loc[:,['v22']]
v22df.fillna("NA",inplace=True)
tv22df.fillna("NA",inplace=True)
del catdf['v22']
del tcatdf['v22']

# In[195]:

vec = DictVectorizer()
vectorized_catdf = vec.fit_transform(catdf.to_dict('records')).toarray()
tvectorized_catdf = vec.transform(tcatdf.to_dict('records')).toarray()

# In[197]:

numtrainrows = v22df['v22'].shape[0]
numtestrows = tv22df['v22'].shape[0]

totalv22 = pd.concat([v22df,tv22df])
x = pd.Categorical(totalv22['v22'])

# In[198]:

v22columncodes = x.codes
v22column = v22columncodes[:numtrainrows]
tv22column = v22columncodes[numtrainrows:]
# In[199]:

v22column = np.matrix(v22column).T
tv22column = np.matrix(tv22column).T

# In[200]:

vectorized_catdf = np.hstack((vectorized_catdf,v22column))
tvectorized_catdf = np.hstack((tvectorized_catdf,tv22column))

# In[202]:

df_withoutid_target = df.drop(['ID'],axis=1)
tdf_withoutid_target = tdf.drop(['ID'],axis=1)


# In[205]:

categorical_columns = df.select_dtypes(['object']).columns
df_without_categ_also = df_withoutid_target.drop(categorical_columns,axis=1)
tdf_without_categ_also = tdf_withoutid_target.drop(categorical_columns,axis=1)

df_without_categ_also.fillna(df.mean(axis=0),inplace=True)
tdf_without_categ_also.fillna(df.mean(axis=0),inplace=True)
# In[209]:

noncat_columns = df_without_categ_also.as_matrix()
tnoncat_columns = tdf_without_categ_also.as_matrix()
print(noncat_columns.shape,vectorized_catdf.shape,tnoncat_columns.shape,tvectorized_catdf.shape)
final_data_matrix = np.hstack((noncat_columns,vectorized_catdf))
tfinal_data_matrix = np.hstack((tnoncat_columns,tvectorized_catdf))
header = range(final_data_matrix.shape[1])
theader = range(tfinal_data_matrix.shape[1])
final_data_matrix = np.vstack((header,final_data_matrix))
tfinal_data_matrix = np.vstack((theader,tfinal_data_matrix))
index = np.matrix(range(final_data_matrix.shape[0])).T
tindex = np.matrix(range(tfinal_data_matrix.shape[0])).T
final_data_matrix = np.hstack((index,final_data_matrix))
tfinal_data_matrix = np.hstack((tindex,tfinal_data_matrix))
df = pd.DataFrame(data=final_data_matrix[1:,1:],index=final_data_matrix[1:,0],columns=final_data_matrix[0,1:])
testdfforid =pd.read_csv("../input/test.csv")
labeldf = df.loc[:,['0.0']]
iddf = testdfforid.loc[:,['ID']]
