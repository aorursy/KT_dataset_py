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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
data.head()
# check the shape
data.shape
# check the datatypes
data.info()
# let's drop Serial no. column
data.drop('Serial No.', axis=1, inplace=True)
data.head()
# let's check missing values
data.isnull().sum()
data.describe()
for feature in data.columns:
    plt.figure(figsize=(16,5))
    sns.distplot(data[feature])
    plt.show()
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data.head()
data.columns=[x.strip(" ") for x in data.columns]
X=data.drop('Chance of Admit', axis=1)
y=data['Chance of Admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Let's apply transformation
sc=StandardScaler()
X_train_tx=sc.fit_transform(X_train)
X_test_tx=sc.transform(X_test)
pca=PCA(n_components=2)
X_train_tx_pca=pca.fit_transform(X_train_tx)
X_test_tx_pca=pca.transform(X_test_tx)
X_train_tx_pca=pd.DataFrame(X_train_tx_pca, columns=['PC1', 'PC2'])
X_train_tx_pca.head()
sns.scatterplot(X_train_tx_pca['PC1'], X_train_tx_pca['PC2'])
pca_max=np.argmax(X_train_tx_pca['PC1'])
pca_min=np.argmin(X_train_tx_pca['PC2'])
pca_max, pca_min
X_train.loc[pca_max, :]
X_train.iloc[pca_min, :]
X_train_tx_pca
lin_reg=LinearRegression()
lin_reg.fit(X_train_tx, y_train)
y_pred=lin_reg.predict(X_test_tx)

print("Score: ", lin_reg.score(X_test_tx, y_test))
