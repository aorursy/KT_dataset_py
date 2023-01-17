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

import seaborn as sns

plt.style.use("seaborn-deep")

%matplotlib inline
df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col='PassengerId')

df.head()
df.info()
def get_features(data, target='Target', delete=[]):

    features = list(data.columns)

    if target in features:

        features.remove(target)

    for f in delete:

        features.remove(f)

    return target, features, delete



def infer_categorical(data, features=None, threshold=10):

    if(not features):

        features = list(data.columns)

    numerical=[]

    categorical=[]

    other=[]

    for f in features:

        if data[f].nunique() <= threshold:

            categorical.append(f)

        else:

            try:

                c = pd.to_numeric(data[f])

                numerical.append(f)

            except ValueError:

                other.append(f)

    return numerical, categorical, other



def auto_process(datasets=[], target='Target', delete=[], cat_threshold=10, std_threshold=0.1, corr_threshold=.01):

    target, features, delete = get_features(data=datasets[0], target=target, delete=delete)

    # delete selected features

    for i in range(len(datasets)):

        datasets[i].drop(columns=delete, inplace=True)



    numerical, categorical, other = infer_categorical(data=datasets[0], features=features)

       

    # ensure numeric

    for i in range(len(datasets)):

        for f in numerical:

            datasets[i][f] = pd.to_numeric(datasets[i][f])

        

    # dummy variables for categorical

    cat_with_na=[]

    cat_without_na = []

    

    for i in range(len(datasets)):

        datasets[i] = pd.get_dummies(data=datasets[i], columns=categorical,drop_first=True, dummy_na=True)

    

    # remove low variance

    low_variance = []

    for f in datasets[0].columns:

        std = datasets[0][f].std()

        if std < std_threshold:

            low_variance.append(f)

    for i in range(len(datasets)):

        datasets[i].drop(columns=low_variance, inplace=True)

        

    #remove low correlation with target

    low_correlation = list(datasets[0].columns[np.abs(datasets[0].corr())[target] < corr_threshold])

    for i in range(len(datasets)):

        datasets[i].drop(columns=low_correlation, inplace=True)

        

    #summarize result

    return dict(datasets=datasets, target=target, features=features, deleted=delete, numerical=numerical, 

                categorical=categorical, other=other, low_variance=low_variance, low_correlation=low_correlation)
# Auto Process

df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col='PassengerId')

df['Age'], bins = pd.qcut(x=df['Age'],q=10, retbins=True) #bin the ages

df_test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col='PassengerId')

df_test['Age'] = pd.cut(x=df_test['Age'], bins=list(bins))

result = auto_process(datasets=[df, df_test], target='Survived', delete=['Name', 'Ticket', 'Cabin'], 

                      cat_threshold=10, std_threshold=0.05, corr_threshold=0.01)

df = result['datasets'][0]

df_test = result['datasets'][1]

target = result['target']

features = list(df.columns)

if target in features:

    features.remove(target)

df.info()
sns.heatmap(np.sqrt(np.abs(df.corr())), cmap='cubehelix')
df_test.fillna(value=0, inplace=True)
df_test.columns.symmetric_difference(df.columns)
df_test['Parch_5.0'] += df_test['Parch_9.0']

df_test['Parch_5.0'].value_counts()

df_test.drop(columns=['Parch_9.0'], inplace=True)
X = df[features]

y = df[target]
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs', max_iter=100000)
model.fit(X, y)
X_test = df_test[features]
X_test.info()
y_pred = model.predict(X_test)
submission = X_test.copy()

col = submission.columns

submission[target] = y_pred

submission.drop(columns=col, inplace=True)

submission.describe()
submission.to_csv("submission.csv")