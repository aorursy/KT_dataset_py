# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# This is my practice to understand how python works
#load in data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# see data contents
df_train.head()

#Check the missing data find Null values exist per column
df_train.isnull().sum()
df_train.describe().T


# Create figures to understand what data really looks like.
# visualization
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
%matplotlib inline
fig1 = sb.FacetGrid(df_train, col='Survived')
fig1.map(plt.hist, 'Age', bins=20)
fig2 = sb.FacetGrid(df_train, col='Survived')
fig2.map(plt.hist, 'Sex', bins=20)
fig3 = sb.FacetGrid(df_train, col='Survived')
fig3.map(plt.hist, 'Pclass', bins=20)
df_train.head()
df_train.corr()
plt.figure(figsize = (10,10))
plt.imshow(df_train.corr(), 
           cmap = 'Blues', 
           #annot = True,
           #annot_kws = { 'fontsize' : 8 },
           vmin = -1.0, 
           vmax = 1.0)
plt.grid('off')
plt.colorbar()
plt.show()
# Transform Sex into binary values 0 and 1
sex = pd.Series( np.where( df_train.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
df_train.head()
from sklearn import metrics, linear_model, neighbors, model_selection
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
df_train.set_index('PassengerId', inplace=True)
titanic = df_train.join(pd.get_dummies(df_train.Pclass))
titanic['is_male'] = df_train.Sex.apply(lambda x: 1 if x == 'male' else 0)