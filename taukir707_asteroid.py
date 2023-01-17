# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
nRowRead = 3000 # specify 'None' if want to read whole file

original_dataset = pd.read_csv("/kaggle/input/prediction-of-asteroid-diameter/Asteroid_Updated.csv", delimiter = ',', nrows = nRowRead)

dataset = original_dataset.copy()


dataset.head(10)
dataset.describe()
dataset.info()
dataset.hist(bins = 50, figsize = (20,15))
## Convert diameter To float

convertDict = {'diameter' : float}

dataset = dataset.astype(convertDict) 
corr_matrix = dataset.corr()

corr_matrix.columns

corr_matrix['diameter'].sort_values(ascending = False)

dataset.isnull().sum()
#(extent : 10/3000,GM : 11/3000, 113/3000, 'G' : 113/3000, IR : 0 /3000) Thse rows have maximun null value

dropColumn = ['extent','GM','G','IR']

dataset = dataset.drop(dropColumn, axis = 1)
dataset.info()
dataset['diameter'].describe()
dataset['diameter'].median()
# As per Analysis of  columns diameter, we should feel this column with its mean value

#dataset['diameter'].filna(dataset['diameter'].mean())

dataset['diameter'].fillna(dataset['diameter'].mean(), inplace=True)
dataset['diameter'].describe()
# As per Analysis of  columns albedo, we should feel this column with its median value

dataset['albedo'].fillna(dataset['albedo'].median(), inplace=True)

dataset['albedo'].describe()
# As per Analysis of  columns rot_per, we should feel this column with its mean value

dataset['rot_per'].fillna(dataset['rot_per'].mean(), inplace=True)

dataset['rot_per'].describe()
# As per Analysis of  columns BV,UB, we should feel this column with its mean value

dataset['BV'].fillna(dataset['BV'].mean(), inplace=True)

dataset['UB'].fillna(dataset['UB'].mean(), inplace=True)
dataset.info()
# Looking for Coorelation

corr_matrix = dataset.corr()

corr_matrix['diameter'].sort_values(ascending = False)
# dataset.plot(kind = 'scatter', x = 'rot_per',y = 'diameter', alpha = 0.6)

import seaborn as sns

#dataset.info()

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

num_data = dataset.select_dtypes(include=numerics)

#num_data.info()

plt.subplots(figsize=(15,12))

sns.heatmap(num_data.corr(),annot=True,annot_kws={'size':10})

#num_data.corr()
# After analysing HeatMap we can element some columns which have no multicolinearity

#e,i,w, condition_cofde, n_obs_use,albedo,not_per,ma

dropNumColumn = ['e','i','w','condition_code','n_obs_used','rot_per','ma']

dataset = dataset.drop(dropNumColumn, axis = 1)

plt.subplots(figsize=(15,12))

num_data = dataset.select_dtypes(include=numerics)

sns.heatmap(num_data.corr(),annot=True,annot_kws={'size':10})
#corr_matrix.columns

dataset.head(10)
dataset.columns
categoricalData = dataset.select_dtypes(include=['object']).copy()

categoricalData.head(5)
categoricalData.isnull().sum()
#categoricalData['spec_B'].value_counts()

categoricalData = categoricalData.fillna(categoricalData['spec_B'].value_counts().index[0])

categoricalData = categoricalData.fillna(categoricalData['spec_T'].value_counts().index[0])
# Columns wise Distribution

print(categoricalData.isnull().sum())
# As we can see that

class_count = categoricalData['class'].value_counts()

sns.set(style="darkgrid")

sns.barplot(class_count.index, class_count.values, alpha=0.9)

plt.title('Frequency Distribution of Class')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Carrier', fontsize=12)

plt.show()
#dataset['neo'].value_counts()

categoricalData['pha'].value_counts()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc = LabelEncoder()

categoricalData['neo'] = labelEnc.fit_transform(categoricalData['neo'])

categoricalData['pha'] = labelEnc.fit_transform(categoricalData['pha'])
categoricalData.head()
# Now do one hot encoder

categoricalData = pd.get_dummies(categoricalData, columns=['neo','pha'])

categoricalData.head()
from sklearn.preprocessing import LabelBinarizer

#categoricalDataClass = categoricalDataCopy.copy()

lb = LabelBinarizer()

lb_results = lb.fit_transform(categoricalData['class'])

lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

categoricalData = pd.concat([categoricalData, lb_results_df], axis=1)

categoricalData.head()
categoricalData['class'].value_counts()


categoricalData['spec_B'] = labelEnc.fit_transform(categoricalData['spec_B'])

categoricalData['spec_T'] = labelEnc.fit_transform(categoricalData['spec_T'])

categoricalData.head()
# Now Drob Class column beacse it has been converted into LabelBinarizor

# Drop name column it jus a name

categoricalData.drop(['name','class'], inplace = True, axis = 1)

categoricalData.head(3)
num_data.head(3)
cleanDataset = pd.concat([categoricalData,num_data],axis = 1)
cleanDataset.head()
#Split Data into features and target

y = cleanDataset['diameter']

X = cleanDataset.drop(['diameter'],axis = 1)
X = X.iloc[:,:].values

X.shape
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([

     ('std_scaler', StandardScaler()),

    # Add as many as you can

])
X_std = my_pipeline.fit_transform(X)
X_std.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

#model = LinearRegression()

#model = DecisionTreeRegressor()

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.predict(X_test)
from sklearn.metrics import mean_squared_error

diameterPrediction  = model.predict(X_test)

lin_mse = mean_squared_error(y_test, diameterPrediction)

lin_mse = np.sqrt(lin_mse)
lin_mse
from sklearn.metrics import r2_score

r2 = r2_score(y_test,diameterPrediction)

print("R2 : ",r2)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = 10)
rm_error = np.sqrt(-scores)# - because sqrt does not calculate negative value

rm_error
def print_score(score):

    print("Score: ", score)

    print("Mean: ", score.mean())

    print("Std: ", score.std())
print_score(rm_error)