# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import statsmodels.api as sm
from sklearn import linear_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

print(os.listdir("../input"))
import plotly.tools as tls
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings

%matplotlib inline
# Any results you write to the current directory are saved as output.
df_features = pd.read_csv("../input/scapFeaturesData.csv", encoding='ISO-8859-1' )
#We delete the MeshID feature from our dataset
del df_features['MeshID']
del df_features['Acromion Shape']
newdf_features = df_features.tail(n=100)
df_features = df_features.head(n=4000)
df_features.head(n=3).transpose()
newdf_features.head(n=3).transpose()
# We define the targets
X = pd.DataFrame(df_features, columns=["CSA","Version","Tilt","Glene Width","Glene Length","Scapula Length","Spine Length","Lat Acromion Angle","Glene Radius"])
X = X.values
# We define the predictors
y = pd.DataFrame(df_features, columns=["First PC","Second PC","Third PC","Fourth PC","Fifth PC","Sixth PC","Seventh PC","Eighth PC","Ninth PC","Tenth PC"])
y = y.values
import sklearn
print("sklearn version: {0}".format(sklearn.__version__))
from sklearn.gaussian_process.kernels import RBF,DotProduct,ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor

kernel_skl = kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
gp_skl = GaussianProcessRegressor(kernel_skl,
                                  optimizer=None,
                                  copy_X_train=False,
                                  normalize_y=True,
                                 )
gp_skl.fit(X, y)
print(gp_skl.log_marginal_likelihood(kernel_skl.theta))
gp_skl.predict(X[0:5])
newX = pd.DataFrame(newdf_features, columns=["CSA","Version","Tilt","Glene Width","Glene Length","Scapula Length","Spine Length","Lat Acromion Angle","Glene Radius"])
newX = newX.values
# We define the predictors
newy = pd.DataFrame(newdf_features, columns=["First PC","Second PC","Third PC","Fourth PC","Fifth PC","Sixth PC","Seventh PC","Eighth PC","Ninth PC","Tenth PC"])
newy = newy.values
gp_skl.predict(newX)

gp_skl.score(newX, newy)
newdf_features.plot.scatter("Lat Acromion Angle","Third PC") 
