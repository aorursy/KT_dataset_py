import numpy as np
import pandas as pd
%matplotlib inline
import zipfile
with zipfile.ZipFile('/kaggle/input/restaurant-revenue-prediction/train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/output/')
with zipfile.ZipFile('/kaggle/input/restaurant-revenue-prediction/test.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/output/')
restaurants = pd.read_csv('/kaggle/output/train.csv')
restaurants.shape
