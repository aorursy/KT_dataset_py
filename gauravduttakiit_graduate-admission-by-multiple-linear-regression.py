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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
grad= pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
grad.head()
grad.info()
grad.describe()
grad.shape
# percentage of missing values in each column

round((100*(grad.isnull().sum())/len(grad)),2).sort_values(ascending=False)
# percentage of missing values in each rows

round((100*(grad.isnull().sum(axis=1))/len(grad)),2).sort_values(ascending=False)
grad_dub=grad.copy()
# Checking for duplicates and dropping the entire duplicate row if any

grad_dub.drop_duplicates(subset=None, inplace=True)
grad_dub.shape
grad.shape
# Check the datatypes before Removing

grad.info()
grad.drop('Serial No.',axis=1, inplace=True)
grad.shape
grad.info()
for col in grad:

    print(grad[col].value_counts(ascending=False))
# Check the shape before spliting



grad.shape
from sklearn.model_selection import train_test_split



# We should specify 'random_state' so that the train and test data set always have the same rows, respectively



np.random.seed(0)

df_train, df_test = train_test_split(grad, train_size = 0.70, test_size = 0.30, random_state = 100)
df_train.info()
df_train.shape
df_test.info()
df_test.shape
df_train.info()
df_train.columns
# Create a new dataframe of only numeric variables:



grad_num=df_train[[ 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',

       'Research', 'Chance of Admit ']]



sns.pairplot(grad_num, diag_kind='kde')

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated. Note:

# here we are considering only those variables (dataframe: grad) that were chosen for analysis



plt.figure(figsize = (25,20))

sns.heatmap(grad.corr(), annot = True, cmap="RdBu")

plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
df_train.head()
df_train.columns
# Apply scaler() to all the numeric variables



num_vars = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',

        'Chance of Admit ']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
# Checking values after scaling

df_train.head()
df_train.describe()
y = df_train.pop('Chance of Admit ')

X= df_train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=80)
from sklearn.metrics import mean_squared_error, r2_score 

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)



print('Error', np.sqrt(mean_squared_error(y_test, y_pred)))
# feature selection

def select_features(X_train, y_train, X_test):

    # configure to select all features

    fs = SelectKBest(score_func=f_regression, k='all')

    # learn relationship from training data

    fs.fit(X_train, y_train)

    # transform train input data

    X_train_fs = fs.transform(X_train)

    # transform test input data

    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs
# feature selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features

from matplotlib import pyplot

for i in range(len(fs.scores_)):

    print('Feature %d: %f' % (i, fs.scores_[i]))

# plot the scores

pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)

pyplot.show()
# print the coefficients

list(zip(X_train.columns, lin_reg.coef_))
print(lin_reg.score(X_test, y_test))

print(r2_score(y_test, y_pred))
coef = pd.Series(lin_reg.coef_, X.columns).sort_values()

coef.plot(kind='bar', title =  'Model Coeff\'s');
fig = plt.figure()

sns.regplot(y_test, y_pred)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 

plt.show()