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
%%time

import warnings

warnings.filterwarnings("ignore")

import sklearn

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import gc

%matplotlib inline

import seaborn as sns

seed = 42

np.random.seed(seed)
# from google.colab import drive

# drive.mount('/content/drive')
data = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

test = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')
def transform(train, test):



	merged = pd.concat([train, test], axis = 0, sort = True)





	num_merged = merged.select_dtypes(include = ['int64','float64'])





	merged.loc[:,['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6','b28','b25','b41','b44','b54','b71','b83','b89']] = merged.loc[:,['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6','b28','b25','b41','b44','b54','b71','b83','b89']].astype('object')





	df_train = merged.iloc[:len(train), :].drop(columns = ['id'], axis = 1)

	df_test = merged.iloc[len(train):, :].drop(columns = ['id', 'label'], axis = 1) 





	df_train.drop(df_train[df_train.b0>300].index, inplace = True)

	df_train.reset_index(drop = True, inplace = True)

	df_train.drop(df_train[df_train.b71==3].index, inplace = True)

	df_train.reset_index(drop = True, inplace = True)





	y_train = df_train.label





	df_train.drop('label', axis = 1, inplace = True)





	df_merged = pd.concat([df_train, df_test], axis = 0)





	df_merged_num = df_merged.select_dtypes(include = ['int64','float64'])





	df_merged_skewed = np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew()>0.75].index])

	df_merged_normal = df_merged_num[df_merged_num.skew()[df_merged_num.skew()< 0.75].index] # Normal variables

	df_merged_num_all = pd.concat([df_merged_skewed, df_merged_normal], axis = 1)





	df_merged_num.update(df_merged_num_all)





	df_merged_num['time_mod'] = df_merged_num['time'] % 1439



	# to remove cols having 0 vals mostly 

	overfit = []

	for i in df_merged_num.columns:

	    counts = df_merged_num[i].value_counts()

	    zeros = counts.iloc[0]

	    if zeros / len(df_merged_num) * 100 > 99.94:

	        overfit.append(i)



	overfit = list(overfit)

	print(overfit)



	df_merged_num = df_merged_num.drop(overfit, axis=1).copy()





	from sklearn.preprocessing import RobustScaler



	'''Initialize robust scaler object.'''

	robust_scl = RobustScaler()





	robust_scl.fit(df_merged_num)





	df_merged_num_scaled = robust_scl.transform(df_merged_num)





	df_merged_num_scaled = pd.DataFrame(data = df_merged_num_scaled, columns = df_merged_num.columns, index = df_merged_num.index)





	df_merged_cat = df_merged.select_dtypes(include = ['object']).astype('category')



	df_merged_cat.a0.replace(to_replace = [0, 1], value = [0, 1], inplace = True)

	df_merged_cat.a1.replace(to_replace = [0, 1], value = [0, 1], inplace = True)

	df_merged_cat.a2.replace(to_replace = [0, 1], value = [0, 1], inplace = True)

	df_merged_cat.a3.replace(to_replace = [0, 1], value = [0, 1], inplace = True)

	df_merged_cat.a4.replace(to_replace = [0, 1], value = [0, 1], inplace = True)

	df_merged_cat.a5.replace(to_replace = [0, 1], value = [0, 1], inplace = True)

	df_merged_cat.a6.replace(to_replace = [0, 1], value = [0, 1], inplace = True)



	df_merged_cat.b25.replace(to_replace = [12.0, 13.0, 17.0], value = [1,2,3], inplace = True)

	df_merged_cat.b28.replace(to_replace = [0.0, 58.0], value = [0, 1], inplace = True)

	df_merged_cat.b41.replace(to_replace = [0.0, 8.0], value = [0, 1], inplace = True)

	df_merged_cat.b44.replace(to_replace = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

	df_merged_cat.b54.replace(to_replace = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

	df_merged_cat.b71.replace(to_replace = [0.0, 1.0, 2.0], value = [0, 1, 2], inplace = True)

	df_merged_cat.b83.replace(to_replace = [1.0, 2.0, 3.0, 4.0], value = [1, 2, 3, 4], inplace = True)

	df_merged_cat.b89.replace(to_replace = [0.0, 5.0], value = [0, 1], inplace = True)



	df_merged_cat.loc[:, ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6','b28','b25','b41','b44','b54','b71','b83','b89']] = df_merged_cat.loc[:, ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6','b28','b25','b41','b44','b54','b71','b83','b89']].astype('int64')



	df_merged_label_encoded = df_merged_cat.select_dtypes(include = ['int64'])





	df_merged_processed = pd.concat([df_merged_num_scaled, df_merged_label_encoded], axis = 1)





	df_train_final = df_merged_processed.iloc[0:len(y_train), :]

	df_test_final = df_merged_processed.iloc[len(y_train):, :]





	y_train = y_train



	return df_train_final, y_train, df_test_final
data_agent1 = data[data.a0!=0]

test_agent1 = test[test.a0!=0]

id_agent1 = test_agent1['id']

Xr_agent1, Yr_agent1, X1_test = transform(data_agent1, test_agent1)

Xr_agent1 = Xr_agent1.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X1_test = X1_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)





data_agent2 = data[data.a1!=0]

test_agent2 = test[test.a1!=0]

id_agent2 = test_agent2['id']

Xr_agent2, Yr_agent2, X2_test = transform(data_agent2, test_agent2)

Xr_agent2 = Xr_agent2.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X2_test = X2_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)





data_agent3 = data[data.a2!=0]

test_agent3 = test[test.a2!=0]

id_agent3 = test_agent3['id']

Xr_agent3, Yr_agent3, X3_test = transform(data_agent3, test_agent3)

Xr_agent3 = Xr_agent3.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X3_test = X3_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)





data_agent4 = data[data.a3!=0]

test_agent4 = test[test.a3!=0]

id_agent4 = test_agent4['id']

Xr_agent4, Yr_agent4, X4_test = transform(data_agent4, test_agent4)

Xr_agent4 = Xr_agent4.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X4_test = X4_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)





data_agent5 = data[data.a4!=0]

test_agent5 = test[test.a4!=0]

id_agent5 = test_agent5['id']

Xr_agent5, Yr_agent5, X5_test = transform(data_agent5, test_agent5)

Xr_agent5 = Xr_agent5.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X5_test = X5_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)





data_agent6 = data[data.a5!=0]

test_agent6 = test[test.a5!=0]

id_agent6 = test_agent6['id']

Xr_agent6, Yr_agent6, X6_test = transform(data_agent6, test_agent6)

Xr_agent6 = Xr_agent6.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X6_test = X6_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)







data_agent7 = data[data.a6!=0]

test_agent7 = test[test.a6!=0]

id_agent7 = test_agent7['id']

Xr_agent7, Yr_agent7, X7_test = transform(data_agent7, test_agent7)

Xr_agent7 = Xr_agent7.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)

X7_test = X7_test.drop(columns = ['a0','a1','a2','a3','a4','a5','a6'], axis=1)





del data

del test

gcc.collect()
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.decomposition import PCA

# Get some classifiers to evaluate

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, AdaBoostRegressor

from sklearn.linear_model import Lasso, Ridge

from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import BaggingRegressor



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold



from sklearn.model_selection import TimeSeriesSplit

rf1 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)

rf2 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)

rf3 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)

rf4 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)

rf5 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)

rf6 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)

rf7 = RandomForestRegressor(n_estimators=1500, random_state=seed, n_jobs=-1)
rf1.fit(Xr_agent1, Yr_agent1)

y_pred1 = rf1.predict(X1_test)

del Xr_agent1

del Yr_agent1

del rf1

gc.collect()
rf2.fit(Xr_agent2, Yr_agent2)

y_pred2 = rf2.predict(X2_test)

del Xr_agent2

del Yr_agent2

del rf2

gc.collect()
rf3.fit(Xr_agent3, Yr_agent3)

y_pred3 = rf3.predict(X3_test)

del Xr_agent3

del Yr_agent3

del rf3

gc.collect()
rf4.fit(Xr_agent4, Yr_agent4)

y_pred4 = rf4.predict(X4_test)

del Xr_agent4

del Yr_agent4

del rf4

gc.collect()
rf5.fit(Xr_agent5, Yr_agent5)

y_pred5 = rf5.predict(X5_test)

del Xr_agent5

del Yr_agent5

del rf5

gc.collect()
rf6.fit(Xr_agent6, Yr_agent6)

y_pred6 = rf6.predict(X6_test)

del Xr_agent6

del Yr_agent6

del rf6

gc.collect()
rf7.fit(Xr_agent7, Yr_agent7)

y_pred7 = rf7.predict(X7_test)

del Xr_agent7

del Yr_agent7

del rf7

gc.collect()
df1 = pd.DataFrame({'id':id_agent1,'label':y_pred1})

df2 = pd.DataFrame({'id':id_agent2,'label':y_pred2})

df3 = pd.DataFrame({'id':id_agent3,'label':y_pred3})

df4 = pd.DataFrame({'id':id_agent4,'label':y_pred4})

df5 = pd.DataFrame({'id':id_agent5,'label':y_pred5})

df6 = pd.DataFrame({'id':id_agent6,'label':y_pred6})

df7 = pd.DataFrame({'id':id_agent7,'label':y_pred7})
submission_df = pd.concat([df1,df2,df3,df4,df5,df6,df7], axis=0)
submission_df.sort_values('id', inplace=True)

submission_df.head(20)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

  csv = df.to_csv(index=False)

  b64 = base64.b64encode(csv.encode())

  payload = b64.decode()

  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

  html = html.format(payload=payload,title=title,filename=filename)

  return HTML(html)



create_download_link(submission_df)