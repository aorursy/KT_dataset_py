# This Python 3 environment comes with many helpful analytics libraries installedimport matplotlib.pyplot as plt

import pandas as pd

from sklearn.cluster import KMeans

import os





from sklearn.cluster import DBSCAN 

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import normalize 

from sklearn.decomposition import PCA 

import numpy as np # linear algebra

import pandas as pd # data processing
import pandas as pd

data = pd.read_csv("../input/heart-disease-uci/heart.csv")
data4 = data.drop('target', axis = 1) 
scaler = StandardScaler() 

data4_scaled = scaler.fit_transform(data4) 
data4_normalized = normalize(data4_scaled) 
data4_normalized = pd.DataFrame(data4_normalized)

pca = PCA(n_components = 2) 

data4_principal = pca.fit_transform(data4_normalized) 

data4_principal = pd.DataFrame(data4_principal) 

data4_principal.columns = ['P1', 'P2'] 

print(data4_principal.head())
db_default = DBSCAN(eps = 0.0375, min_samples = 2).fit(data4_principal) 

labels = db_default.labels_ 

#Building the label to colour mapping 

colours = {} 

colours[0] = 'r'

colours[1] = 'g'

colours[2] = 'b'

r = plt.scatter(data4_principal['P1'], data4_principal['P2'], color ='r'); 

g = plt.scatter(data4_principal['P1'], data4_principal['P2'], color ='g'); 

b = plt.scatter(data4_principal['P1'], data4_principal['P2'], color ='b'); 



import matplotlib.pyplot as plt



# Plotting P1 on the X-Axis and P2 on the Y-Axis  

# according to the colour vector defined 

plt.figure(figsize =(9, 9)) 

plt.scatter(data4_principal['P1'], data4_principal['P2']) 

  

# Building the legend 

plt.legend((r, g, b ), ('Label 0', 'Label 1', 'Label 2', 'Label -1')) 



  

plt.show()

db = DBSCAN(eps = 0.0375, min_samples = 50).fit(data4_principal) 

labels1 = db.labels_

colours1 = {} 

colours1[0] = 'r'

colours1[1] = 'g'

colours1[2] = 'b'

colours1[3] = 'c'

colours1[4] = 'y'

colours1[5] = 'm'

colours1[-1] = 'k'

  
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k' ] 

  

r = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[0]) 

g = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[1]) 

b = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[2]) 

c = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[3]) 

y = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[4]) 

m = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[5]) 

k = plt.scatter( 

        data4_principal['P1'], data4_principal['P2'], marker ='o', color = colors[6]) 

  

plt.figure(figsize =(9, 9)) 

plt.scatter(data4_principal['P1'], data4_principal['P2']) 

plt.legend((r, g, b, c, y, m, k), 

           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 2' , 

            'Label 5', 'Label -1'), 

           scatterpoints = 1, 

           loc ='upper left', 

           ncol = 3, 

           fontsize = 8) 

plt.show() 