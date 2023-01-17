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
import seaborn as sns
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
!pip install factor_analyzer
dataset = pd.read_csv('/kaggle/input/eergy-efficiency-dataset/ENB2012_data.csv')
dataset.info(null_counts = True)
print (dataset.head())
dataset.isna().sum()
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
chi_square_value,p_value=calculate_bartlett_sphericity(dataset)
print (chi_square_value)
print (p_value)
kmo_all,kmo_model=calculate_kmo(dataset)
print (kmo_all)
print (kmo_model)
fa = FactorAnalyzer(42, rotation="varimax")
fa.fit(dataset)
ev, v = fa.get_eigenvalues()
print (ev)
print (v)
plt.scatter(range(1,dataset.shape[1]+1),ev)
plt.plot(range(1,dataset.shape[1]+1),ev)
plt.title('Scatter Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
plt.scatter(range(1,dataset.shape[1]+1),v)
plt.plot(range(1,dataset.shape[1]+1),v)
plt.title('Scatter Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue-2')
plt.grid()
plt.show()
x = dataset.iloc[:,:-2].values
y = dataset.iloc[:,-2].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
pred_rf = reg_rf.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, pred_rf)