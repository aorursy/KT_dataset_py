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
import numpy as np;import numpy as np;

import pandas as pd;

import matplotlib.pyplot as plt;

import seaborn as sns;

from sklearn.impute import SimpleImputer;

from sklearn.compose import ColumnTransformer;

from sklearn.pipeline import Pipeline;

from sklearn.preprocessing import LabelEncoder;

from sklearn.preprocessing import StandardScaler;

from sklearn.preprocessing import MinMaxScaler;

from sklearn.model_selection import train_test_split;

from sklearn.linear_model import LinearRegression ;

from sklearn.linear_model import Ridge, Lasso;

from sklearn.metrics import mean_squared_error;

from sklearn.metrics import r2_score;

from sklearn.preprocessing import PolynomialFeatures;

from sklearn.svm import SVR;

from sklearn.svm import SVC;

from sklearn.tree import DecisionTreeClassifier;

from sklearn.ensemble import RandomForestClassifier;

from sklearn.ensemble import RandomForestRegressor;

from sklearn.neighbors import KNeighborsClassifier;

from sklearn.naive_bayes import GaussianNB;

import pickle;
a= pd.read_csv('../input/breast-cancer-prediction-dataset/Breast_cancer_data.csv');

a.shape
sns.heatmap(a.corr(), annot = True)
x_a = a.drop(columns = 'diagnosis', axis = 1);

y_a = a[['diagnosis']];

x_train, x_test, y_train, y_test = train_test_split(x_a,y_a, test_size = 0.2, random_state = 55);

sc =  StandardScaler();

sc.fit_transform(x_train);

x_train_sc = sc.fit_transform(x_train);

x_test_sc = sc.fit_transform(x_test);

sv = SVC(kernel='rbf');

sv.fit(x_train_sc, y_train)
sv.score(x_test_sc,y_test)