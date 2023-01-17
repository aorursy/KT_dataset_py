import pandas as pd
import numpy as np
import gc
import os
import cv2
import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

test_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
ben_pred = pd.read_csv("../input/predictions-2/benign_with_preds (1).csv")
mal_pred = pd.read_csv("../input/predictions-2/malignant_with_preds (1).csv")
SAMPLE_NUM = 600

new_benign = ben_pred[0:SAMPLE_NUM]
benign_predictios = new_benign['predictions']

new_mal = mal_pred[0:SAMPLE_NUM]
malignant_predictions = new_mal['predictions']
test_df = pd.concat([test_df,pd.get_dummies(test_df['anatom_site_general_challenge'], prefix='site')],axis=1)
test_df.drop(['image_name'],axis=1, inplace=True)
test_df.drop(['patient_id'],axis=1, inplace=True)
test_df.drop(['anatom_site_general_challenge'],axis=1, inplace=True)

new_benign.drop(['anatom_site_general_challenge'],axis=1, inplace=True)
new_benign.drop(['patient_id'],axis=1, inplace=True)
new_benign.drop(['image_name'],axis=1, inplace=True)
new_benign.drop(['Unnamed: 0'],axis=1, inplace=True)
new_benign.drop(['target', 'site_anterior torso', 'site_lateral torso', 'site_posterior torso', 'site_nan'], inplace=True, axis=1)



new_mal.drop(['anatom_site_general_challenge'],axis=1, inplace=True)
new_mal.drop(['patient_id'],axis=1, inplace=True)
new_mal.drop(['image_name'],axis=1, inplace=True)
new_mal.drop(['Unnamed: 0'],axis=1, inplace=True)
new_mal.drop(['target', 'site_anterior torso', 'site_lateral torso', 'site_posterior torso', 'site_nan'], inplace=True, axis=1)




import random
from matplotlib import pyplot

from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')


#plot both histograms(range from -10 to 10), bins set to 100
pyplot.hist([benign_predictios, malignant_predictions], bins= 100, range=[0,1.1], label=['ben', 'mal'], density = True)
#plot legend
pyplot.legend(loc='upper right')
#show it
pyplot.show()
benign_first = new_benign['predictions'] > 0.15
new_benign = new_benign[benign_first]

m_1 =  new_mal['predictions'] < 0.85
new_mal = new_mal[m_1]

new_benign['group'] = 'A'
new_mal['group'] = 'M'
all_preds = pd.concat([new_benign,new_mal], axis = 0)

a = (all_preds['age_approx'] -all_preds['age_approx'].mean())/all_preds['age_approx'].std()
all_preds['age_approx'] = a

all_preds_meta = all_preds.loc[:, all_preds.columns.str.contains('^g')]

all_preds.drop(['predictions', 'group'], inplace=True, axis=1)
all_preds

# all_preds = test_df
# a = (all_preds['age_approx'] -all_preds['age_approx'].mean())/all_preds['age_approx'].std()
# all_preds['age_approx'] = a
# all_preds.fillna(0, inplace = True)
# all_preds['sex'] = all_preds['sex'].map({'male': 1, 'female': 0})
# all_preds['sex'] = all_preds['sex'].fillna(-1)
# all_preds
import numpy as np
u, s, v = np.linalg.svd(all_preds, full_matrices=True)

# S = np.loadtxt('/kaggle/input/matricesnew/S.txt', dtype=float)
# S = np.diag(S)
# V = np.loadtxt('/kaggle/input/matricesnew/V.txt', dtype=float)
# M = all_preds.values
# v = V.transpose() 
# s = np.linalg.inv(s)
# U = np.matmul(np.matmul(M, v), s)

var_explained = np.round(s**2/np.sum(s**2), decimals=3)
var_explained
 
sns.barplot(x=list(range(1,len(var_explained)+1)),
            y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('svd_scree_plot.png',dpi=100)
labels= ['SV'+str(i) for i in range(1,10)]
svd_df = pd.DataFrame(u[:,0:9], index=all_preds_meta["group"].tolist(), columns=labels)
svd_df=svd_df.reset_index()
svd_df.rename(columns={'index':'Group'}, inplace=True)
svd_df
A = svd_df[svd_df['Group'] == 'A']  
A_M = pd.concat([A,M],axis=0)
M = svd_df[svd_df['Group'] == 'M']  
from matplotlib.pyplot import figure
figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')


#plot both histograms(range from -10 to 10), bins set to 100
n, bins, _ =pyplot.hist([A['SV1'], M['SV1']], bins= 100, range=[-5,5], label=['A', 'M'], density = True)
#plot legend
pyplot.legend(loc='upper right')
#show it
pyplot.show()
import itertools
l = [1,2,3,4,5,6,7,8,9,10]

perms = list(itertools.combinations(l, 3)) 
print(len(perms))
print(perms)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px

i = -1


X = 'SV' + str(perms[i][0])
Y = 'SV' + str(perms[i][1])
Z = 'SV' + str(perms[i][2])

fig = px.scatter_3d(A_M, x=X, y=Y, z=Z, color='Group',
                    title="3D Scatter Plot")
fig.show()
import itertools
l = [1,2,3,4,5,6,7,8,9,10]

perms = list(itertools.combinations(l, 2))
print(len(perms))
print(perms)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px

i=perms.index((8, 9))
X = 'SV' + str(perms[i][0])
Y = 'SV' + str(perms[i][1])

fig = px.scatter(A_M, x=X, y=Y, color='Group',
             width=700, height=500,
             title="2D Scatter Plot")
fig.show()