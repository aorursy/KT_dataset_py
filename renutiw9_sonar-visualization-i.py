# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas import set_option

from pandas.plotting import scatter_matrix,andrews_curves, parallel_coordinates

from sklearn.manifold import MDS

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



plt.rcParams['figure.figsize'] = [10, 7]

# Any results you write to the current directory are saved as output.
sonar_data = pd.read_csv('../input/sonar-mines-vs-rocks/sonar_all-data.csv', header=None)



print("   Data shape",sonar_data.shape)

print("============================================================")

print('   Class Distribution in the dataset : \n  ' ,sonar_data.groupby(60).size())

print("============================================================")

print ("Data type of values ",sonar_data.dtypes)

print("============================================================")
print("Overview of data")

print("============================================================")

sonar_data.head()
print("Stastical Overview")

print("============================================================")

sonar_data.describe()
#Storing all rows in X. 1st and Last(60) coloumn in Y.



X = sonar_data.iloc[:,0:60].values.astype(float)

Y = sonar_data.iloc[:,60]



encoder= LabelEncoder()

encoder.fit(Y)

en_Y= encoder.transform(Y)



print (Y)

print(en_Y)



print ("BASIC DATA PREPRATION ")

print("============================================================")

print("X has -\n")

print(X)

print("============================================================")

print("Y has -\n")

print(en_Y)





sonar_data.rename(columns={sonar_data.columns[60]:'Label'}, inplace=True)

sonar_data.Label = sonar_data.Label.astype('category')

sonar_data.head()
fig, axes = plt.subplots(2,1,figsize=(15,10))

andrews_curves(sonar_data, 'Label', samples=207, linewidth=0.5, ax=axes[0])

axes[0].set_xticks([])



parallel_coordinates(sonar_data, 'Label', linewidth=0.5, ax=axes[1],

                     axvlines_kwds={'linewidth': 0.5, 'color': 'black', 'alpha':0.5})

axes[1].set_xticks([])

axes[1].margins(0.05)

pass


mds = MDS(n_components=2)

mds_data = mds.fit_transform(sonar_data.iloc[:, :-1])

plt.scatter(mds_data[:, 0],mds_data[:, 1],c=sonar_data.Label.cat.codes, s=50);





heatmap = plt.pcolor(sonar_data.corr(), cmap='jet')

plt.colorbar(heatmap)

pass
 # Unimodal Density Visualization

sonar_data.plot(kind='density', subplots=True, sharex=False, sharey=False, layout=(10,8) ,fontsize=3)

plt.show()
sonar_data.plot.box(figsize=(13,7), xticks=[])

pass
plt.figure(figsize=(9,7))

plt.plot(sonar_data[Y == 'R'].values[0][:-1], label='Rock', color='pink')

plt.plot(sonar_data[Y == 'M'].values[0][:-1], label='Metal', color='skyblue', linestyle='--')

plt.legend()

plt.xlabel('Attribute Index')

plt.ylabel('Attribute Values')

plt.tight_layout()

plt.show()