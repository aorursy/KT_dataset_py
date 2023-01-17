# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import plotly.express as px

import folium



# for ML:

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import eli5 # Feature importance evaluation







# set some display options:

sns.set(style="whitegrid")

pd.set_option("display.max_columns", 36)



# load data:

file_path = "../input/hotel-booking-demand/hotel_bookings.csv"

full_data = pd.read_csv(file_path)


hotel_data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')


hotel_data.info()


pd.crosstab(columns = hotel_data['reservation_status'], index = hotel_data['is_canceled'],

           margins=True, margins_name = 'Total')