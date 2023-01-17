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
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import math



#classification

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

 

#Regression

from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, ElasticNet

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor



# Modelling Helpers 

from sklearn.preprocessing import Normalizer , scale

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFECV

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score



#preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder



#evaluation matrics



# Regression

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 



# Classification

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score





#visualisation

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

params = { 

    'axes.labelsize': "large",

    'xtick.labelsize': 'x-large',

    'legend.fontsize': 20,

    'figure.dpi': 150,

    'figure.figsize': [25, 7]

    }

plt.rcParams.update(params)
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
df = pd.read_csv("/kaggle/input/insurance/insurance.csv")

insurance = df.copy
df.info()
df.describe()
df.describe( include = ['O'])
df.head(10)
print(df.loc[df.bmi == 0])
sns.factorplot(data = df, kind = 'box', size = 7, aspect = 2.5)
df_correlation = df.corr()

fig = plt.figure(0,figsize = (5,5))

sns.heatmap(data = df_correlation, annot = True)
sns.kdeplot(df.age, shade = True, color = 'orange')
sns.factorplot(data = df, x = "age", y = "charges", kind = "box", aspect = 3.5)
sns.kdeplot(df.bmi, shade = True, color = 'r')
sns.regplot(data = df, x = "bmi", y = "charges")
sns.kdeplot(df.children, shade = True, color = 'r')
sns.factorplot(data = df, x = "children", y = "charges", kind = "box", aspect = 3.5)
plt.hist(df.sex)

plt.show()
sns.factorplot(data = df, x = "sex", y = "charges", kind = "box", aspect = 3.5)
plt.hist(df.smoker)

plt.show()
sns.factorplot(data = df, x = "smoker", y = "charges", kind = "box", aspect = 3.5)
plt.hist(df.region)

plt.show()
sns.factorplot(data = df, x = "region", y = "charges", kind = "box", aspect = 3.5)
label_sex = LabelEncoder()

label_smoker = LabelEncoder()

label_region = LabelEncoder()



df.sex = label_sex.fit_transform(df.sex)

df.smoker = label_smoker.fit_transform(df.smoker)

df.region = label_region.fit_transform(df.region)
df.head()
input_cols = ["age","sex","bmi","children","smoker","region"]

output_cols = ["charges"]
X_train,X_test,y_train,y_test = train_test_split(df[input_cols],df[output_cols],test_size = 0.3, random_state = 30)
ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)
accuracy = []

models = ["Linear regression","Lasso","Ridge","Random Forest Regressor","Gradient Boost", "AdaBoost", "SVR","KNeighbors","MLPRegressor"]
mod = LinearRegression()

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
mod = Lasso()

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
mod = Ridge()

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
for i in range (1,10):

    mod = RandomForestRegressor( max_depth = i)

    mod.fit(X_train,y_train)

    a = mod.score(X_test,y_test)

    print(a)

#accuracy.append(a)
no_of_test = [100]

params_dict = {"n_estimators": no_of_test,

               "max_depth":[4],

              "n_jobs":[-1],

              "max_features":["auto","sqrt","log2"]}

mod = GridSearchCV(estimator = RandomForestRegressor(),param_grid = params_dict, scoring = "r2")

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
mod = GradientBoostingRegressor()

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
mod = AdaBoostRegressor()

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
mod = SVR(kernel = "linear", C=5.0)

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
for i in range(1,50):  

    mod = KNeighborsRegressor(n_neighbors=i)

    mod.fit(X_train,y_train)

    a = mod.score(X_test,y_test)

    print(a)

#accuracy.append(a)
no_of_test = [100]

params_dict = {"n_neighbors": no_of_test,

              "n_jobs":[-1]

              }

mod = GridSearchCV(estimator = KNeighborsRegressor(),param_grid = params_dict, scoring = "r2")

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
mod = MLPRegressor(hidden_layer_sizes = (100,))

mod.fit(X_train,y_train)

a = mod.score(X_test,y_test)

print(a)

accuracy.append(a)
sns.factorplot(data = df, x = models, y = accuracy, size = 6, aspect = 4)