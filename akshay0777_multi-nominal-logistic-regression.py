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
# Required Python Packages

import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn import metrics

from sklearn.model_selection import train_test_split



!pip install chart-studio
!pip install plotly==4.1.0
DATASET_PATH = "/kaggle/input/glass.csv"
df = pd.read_csv("/kaggle/input/glass.csv")

features = df.columns[:-1].tolist()
df.head()
df.describe()
df.info()
import matplotlib.pyplot as plt # visualization

import seaborn as sns # statistical visualizations and aesthetics

for feat in features:

    skew = df[feat].skew()

    sns.distplot(df[feat], kde=False, label='Skew = %.3f' %(skew), bins=30)

    plt.legend(loc='best')

    plt.show()
df['Type'].value_counts()
sns.countplot(df['Type'])

plt.show()
# Define X as features and y as lablels

X = df[features] 

y = df['Type'] 

# set a seed and a test size for splitting the dataset 

seed = 7

test_size = 0.2



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size , random_state = seed)

X_train
X_test
from scipy.stats import boxcox # data transform

features_boxcox = []



for feature in features:

    bc_transformed, _ = boxcox(df[feature]+1)  # shift by 1 to avoid computing log of negative values

    features_boxcox.append(bc_transformed)



features_boxcox = np.column_stack(features_boxcox)

df_bc = pd.DataFrame(data=features_boxcox, columns=features)

df_bc['Type'] = df['Type']
df_bc.describe()
#check if skew is closer to zero after a box-cox transform

for feature in features:

    delta = np.abs( df_bc[feature].skew() / df[feature].skew() )

    if delta < 1.0 :

        print('Feature %s is less skewed after a Box-Cox transform' %(feature))

    else:

        print('Feature %s is more skewed after a Box-Cox transform'  %(feature))


# Train multi-class logistic regression model

    lr = linear_model.LogisticRegression()

    lr.fit(X_train, y_train)
# Train multinomial logistic regression model

    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y_train)
print (mul_lr.score(X_test,y_test))