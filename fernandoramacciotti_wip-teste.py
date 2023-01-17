# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
X = train.iloc[:, 2:]

y = train.iloc[:, 1]

test = test_data.iloc[:, 1:]



# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



labelencoder_X = LabelEncoder()

for i in range(0, 8):

    X.iloc[:, i] = labelencoder_X.fit_transform(X.iloc[:, i])

    test.iloc[:, i] = labelencoder_X.fit_transform(test.iloc[:, i])



from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, 'ward')

Z_test = linkage(test, 'ward')
plt.title('Hierarchical Clustering Dendrogram (truncated)')

plt.xlabel('sample index or (cluster size)')

plt.ylabel('distance')

dendrogram(

    Z,

    truncate_mode='lastp',  # show only the last p merged clusters

    p=12,  # show only the last p merged clusters

    leaf_rotation=90.,

    leaf_font_size=12.,

    show_contracted=True,  # to get a distribution impression in truncated branches

)

plt.show()
from scipy.cluster.hierarchy import fcluster

max_d = 400

clusters = fcluster(Z, max_d, criterion='distance')

clusters_test = fcluster(Z_test, max_d, criterion = 'distance')
cluster_df = pd.DataFrame(clusters.reshape(X.shape[0], 1))

cluster_df.rename(columns = {0: "cluster"}, inplace = True)



cluster_df_test = pd.DataFrame(clusters_test.reshape(test.shape[0], 1))

cluster_df_test.rename(columns = {0: "cluster"}, inplace = True)
X = pd.concat([X, cluster_df], axis = 1, join = 'inner', ignore_index = True)

test = pd.concat([test, cluster_df_test], axis = 1, join = 'inner', ignore_index = True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators = 300, random_state = 42)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score

r2 = r2_score(y_true = y_test, y_pred = y_pred)

r2
y_pred_test = regressor.predict(test)

output = pd.DataFrame()

output['ID'] = test_data['ID']

output['y'] = y_pred_test

output.to_csv('teste_20170605.csv', sep = ',', index = False)