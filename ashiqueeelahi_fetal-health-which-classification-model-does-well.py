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
import numpy as np;
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
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;

import keras;
from keras_preprocessing import image;
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam;
from keras.callbacks import ModelCheckpoint;
from keras.models import Sequential;
from tensorflow.keras.applications import VGG16;
from tensorflow.keras.applications import InceptionResNetV2;
from keras.applications.vgg16 import preprocess_input;
from tensorflow.keras.applications.vgg16 import decode_predictions;

import os;
from os import listdir;
from PIL import Image as PImage;
import cv2
data = pd.read_csv('../input/fetal-health-classification/fetal_health.csv')
data.columns
plt.figure(figsize = (25,25))
sns.heatmap(data.corr(), annot = True)

plt.figure(figsize = (25,25))
x = data.drop(['fetal_health'], axis = 1) 
sns.barplot(x.columns,x.corrwith(data.fetal_health))
x.drop(columns = ['fetal_movement','light_decelerations','histogram_width','histogram_min','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes'], axis = 1)
#                                  
                                           
                           

y = data['fetal_health']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 51)
x_train.shape, y_train.shape
x_train.columns
xx =XGBClassifier();
xx.fit(x_train, y_train)
xx.score(x_test, y_test)
rfc = RandomForestClassifier();
rfc.fit(x_train, y_train)
rfc.score(x_test, y_test)
knn = KNeighborsClassifier();
knn.fit(x_train, y_train)
knn.score(x_test, y_test)
dcc = DecisionTreeClassifier();
dcc.fit(x_train, y_train)
dcc.score(x_test, y_test)
ss = StandardScaler();
x_train_ss = ss.fit_transform(x_train);
x_test_ss = ss.fit_transform(x_test)
svc = SVC();
svc.fit(x_train_ss, y_train)
svc.score(x_test_ss, y_test)
