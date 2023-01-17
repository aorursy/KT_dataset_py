!pip install h2o

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import plotly
%matplotlib inline
 
import matplotlib.pyplot as plt
 
from matplotlib import style

import h2o
from h2o.automl import H2OAutoML
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/housedata/data.csv")
df.head()
df.info()

df['price'] = (df['price']).astype(int)
df['floors'] = (df['floors']).astype(int)
df['bedrooms'] = (df['bedrooms']).astype(int)
sns.pairplot(df)
plt.show()
plt.figure(figsize=(20, 12))
 
sns.boxplot(x = 'city', y = 'price', data = df)

plt.show()
plt.figure(figsize=(20, 12)) 
sns.boxplot(x = 'statezip', y = 'price', data = df)

plt.show()
plt.figure(figsize=(20, 12))
plt.subplot(1,3,3)
sns.boxplot(x = 'country', y = 'price', data = df)
df['city'].unique
df = df.dropna()

sns.distplot(df['yr_built'])

# Move our features into the X DataFrame
X = houses_o.loc[:,['bedrooms', 'floors','view','condition','renovated_0_1']]

# Move our labels into the y DataFrame
y = houses_o.loc[:,['price']] 
# Initialize your cluster
h2o.init()
h2o_df = h2o.H2OFrame(df)
h2o_df.types
h2o_df.describe()
h2o_df_train,h2o_df_test,h2o_df_valid = h2o_df.split_frame(ratios=[.7, .15])
y = "price"
X = h2o_df.columns
X.remove(y)
X.remove("street")
X.remove("date")
h2o_df_train.drop(["street",'date'], axis=1)
h2o_df_test.drop(["street",'date'], axis=1)
h2o_df_valid.drop(["street",'date'], axis=1)

aml = H2OAutoML(max_runtime_secs=100,
                seed = 10, 
                #exclude_algos = ["StackedEnsemble", "DeepLearning"], 
                verbosity="info", 
                nfolds=0,
                #balance_classes=True,
                project_name='Completed')

aml.train(x=X, y=y, training_frame=h2o_df_train ,validation_frame=h2o_df_valid)
lb = aml.leaderboard
lb

aml.leader

y_preds = aml.leader.predict(h2o_df_test)

aml.leader.model_performance(h2o_df_test)
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
model_ids
h2o.get_model([mid for mid in model_ids if "DeepLearning" in mid][0])
h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])
out_XGBoost = h2o.get_model([mid for mid in model_ids if "XGBoost" in mid][0])
out_XGBoost.params
out_XGBoost.convert_H2OXGBoostParams_2_XGBoostParams()
out_XGBoost
out_XGBoost.varimp_plot()
model = h2o.get_model('XGBoost_grid__1_AutoML_20200826_061508_model_11')
model.model_performance(h2o_df_test)
