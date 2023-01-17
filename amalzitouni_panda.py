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
# Ignore warnings :

import warnings

warnings.filterwarnings('ignore')





# Handle table-like data and matrices :

import numpy as np

import pandas as pd

import math 

import itertools



# Modelling Helpers :

from sklearn.preprocessing import Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score







# Evaluation metrics :



# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 



# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score





# Deep Learning Libraries

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.utils import to_categorical





# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno





# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)
# Ignore warnings :

import warnings

warnings.filterwarnings('ignore')





# Handle table-like data and matrices :

import numpy as np

import pandas as pd

import math 

import itertools



# Modelling Helpers :

from sklearn.preprocessing import   Normalizer , scale

from sklearn.impute import SimpleImputer



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score







# Evaluation metrics :



# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 



# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score





# Deep Learning Libraries

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.utils import to_categorical





# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno





# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)
# Center all plots

from IPython.core.display import HTML

HTML("""

<style>

.output_png {

    display: table-cell;

    text-align: center;

    vertical-align: middle;

}

</style>

""");





# Make Visualizations better

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

}

plt.rcParams.update(params)
data = pd.read_csv("/kaggle/input/rtmatrix1/rtMatrix1.csv")
userdata = pd.read_csv("/kaggle/input/qosdata/userlist.csv")

wsdata = pd.read_csv("/kaggle/input/wsliste/wslist.csv")
df_d=data.copy()
df_u=userdata.copy()
df_ws=wsdata.copy()
data.head()
data.shape
data.describe()

df_combined = pd.merge( wsdata,userdata)
userdata.head()
from csv import reader

from surprise import Reader, Dataset, KNNBasic, SVD, NMF

from surprise.model_selection import GridSearchCV, cross_validate
reader = Reader(rating_scale=(0.5, 5.0))



data = Dataset.load_from_df( data[[]], reader = reader )
# Compute Mean Squared Distance Similarity

sim_options = {'name' : 'msd'}



algo = KNNBasic(k=10, sim_options=sim_options )

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)

from sklearn.model_selection import KFold

kf = KFold(n_splits=4)
kf.get_n_splits(data)
print(kf)
cross_validate(algo=algo, data=data, measures=['RMSE'], cv=4, verbose=True)
n_neighbours = [10, 20, 30]

param_grid = {'n_neighbours' : n_neighbours}
gs = GridSearchCV(KNNBasic, measures=['RMSE'], param_grid=param_grid)

gs.fit(data)
