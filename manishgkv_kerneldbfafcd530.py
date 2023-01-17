# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/pricewholesale/whoesale price.xlsx'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score
# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
# Modelling Helpers
from sklearn.preprocessing import Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer 
Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
train = pd.read_excel("../input/pricewholesale/whoesale price.xlsx")
df = train
df
df.shape
df.describe()
X = [0 , 1 , 2 , 3 , 4 ]
y = np.array(df.loc[ df.CROP == "APPLE" , "_2013" : "_2017" ])
X
y
X = np.array(X)
X
y
y = y.reshape(5 , 1)
X
X = X.reshape(5 , 1)
X
y.shape
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

reg = svm.SVR(C=0.5 , epsilon=0.01)
reg.fit(X , y )

X_test = [6,7,8,9,10]
X_test = np.array(X_test)
pred = reg.predict(X_test.reshape(5 , 1))
pred
Years = ["2004-05" , "2005-06" ,"2006-07", "2007-08" , "2008-09" , "2009-10" , "2010-11" , "2011-12" , "2012-13" , "2013-14" , "2014-15" , "2015-16" , "2016-17" , "2017-18" , "2018-19" , "2019-20" ]
def calc_pred(Crop):
    X = [0 , 1 , 2 , 3 , 4 ]
    y = np.array(df.loc[ df.CROP == Crop , "_2013" : "_2017" ])
    X = np.array(X)
    y = y.reshape(5,1)
    X = X.reshape(5,1)
    reg = svm.SVR(C=0.5 , epsilon=0.01)
    reg.fit(X , y ) 
    X_test = [6,7,8,9,10]
    X_test = np.array(X_test)
    pred = reg.predict(X_test.reshape(5 , 1))
    pred = pred.reshape(5 , 1)
    temp = y.reshape(5 , 1)
    manish = []
    for i in temp:
        manish.append(i)
    for i in pred:
        manish.append(i)
    manish = np.array(manish)
    manish = manish.reshape(10,)
    Years = ["_2013" , "_2014" , "_2015" , "_2016" , "_2017" , "_2018" , "_2019","_2020"]
    Years = np.array(Years)
    d = { "Years" : Years , "Prediction" : manish}
    final = pd.DataFrame.from_dict(d, orient='index')
    final.transpose()    
    fig, axes = plt.subplots(1,1)
    fig.set_size_inches(10, 5)
    sns.pointplot(data=final ,x = "Years" ,y = "Prediction", orient="v" )
    fig.savefig("5.png")
    final.to_csv( 'price_prediction.csv' , index = False )
    print(pred)
calc_pred("APPLE")
