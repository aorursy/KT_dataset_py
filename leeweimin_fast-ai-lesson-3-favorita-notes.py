!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

!apt update &amp;&amp; apt install -y libsm6 libxext6
%load_ext autoreload
%autoreload 2

%matplotlib inline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from IPython.display import display
from fastai.imports import *
from fastai.structured import *
types = {'id': 'int64',
         'item_nbr': 'int32',
         'store_nbr': 'int8',
         'unit_sales': 'float32',
         'onpromotion': 'object'}

%time
# dtype=types: Provide data type of each column to reduce memory usage
df_all = pd.read_csv("../input/train.csv", parse_dates=['date'], dtype=types, infer_datetime_format=True)
# Fill NaN values in 'onpromotion' column
# Kaggle: "Approximately 16% of the onpromotion values in this file are NaN."
df_all.onpromotion.fillna(False, inplace=True)
# Map string values to boolean
df_all.onpromotion = df_all.onpromotion.map({'False': False, 'True': True})
df_all.onpromotion
