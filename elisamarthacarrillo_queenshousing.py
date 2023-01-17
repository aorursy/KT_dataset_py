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
import math

%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
DATA_FOLDER = "Titanic"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, DATA_FOLDER, "images")

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi = resolution)
    
import warnings
warnings.filterwarnings(action="ignore", message ="^internal gelsd")

totalQueens = pd.read_csv("/kaggle/input/queenstotal/rollingsales_queens.csv", thousands = ',')
print(totalQueens)
from sklearn.model_selection import train_test_split 
train_set, test_set = train_test_split( totalQueens, test_size = 0.2, random_state = 42)


qhousing = train_set.copy()
print(qhousing.describe())
qhousing.head()
#notes
#get rid of rows that have 0 total units
# drop easements
qhousing.drop("ADDRESS", axis=1, inplace = True)


qhousing.drop("EASE-MENT", axis=1, inplace = True)

qhousing.drop("APARTMENT NUMBER", axis=1, inplace = True)

qhousing.head()
qhousing.drop("LAND SQUARE FEET", axis=1, inplace = True)
qhousing.drop("TOTAL UNITS", axis=1, inplace = True)
qhousing.drop("RESIDENTIAL UNITS", axis=1, inplace = True)
qhousing.drop("COMMERCIAL UNITS", axis=1, inplace = True)
# print(qhousing)
qhousing.drop("BOROUGH", axis=1, inplace = True)
qhousing = qhousing.dropna()
qhousing.count()
# print(qhousing)
qhousing.drop(qhousing[qhousing[' SALE PRICE '] == 0].index, inplace = True) 
qhousing = qhousing.reset_index(drop=True)
# print(qhousing)
# qhousing['NEIGHBORHOOD'].value_counts()
# qhousing['BUILDING CLASS CATEGORY'].value_counts()
# qhousing['TAX CLASS AT PRESENT'].value_counts()
# qhousing['BUILDING CLASS AT PRESENT'].value_counts()
qhousing = pd.get_dummies(qhousing, columns=["NEIGHBORHOOD",
                                             "BUILDING CLASS CATEGORY",
                                             "BUILDING CLASS AT TIME OF SALE",
                                             "TAX CLASS AT PRESENT",
                                             "BUILDING CLASS AT PRESENT",
                                            "YEAR BUILT", 
                                            "SALE DATE"])
# qhousing["NEIGHBORHOOD"] = qhousing["NEIGHBORHOOD"].cat.codes
qhousing.head()
corr_matrix = qhousing.corr()
corr_matrix[" SALE PRICE "].sort_values(ascending = False)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        
        ('std_scaler', StandardScaler()),
    ])
qhousing_tr = num_pipeline.fit_transform(qhousing)
# qhousing_tr
from sklearn.compose import ColumnTransformer


num_attribs = list(qhousing)

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        
    ])

qhousing_prepared = full_pipeline.fit_transform(qhousing)