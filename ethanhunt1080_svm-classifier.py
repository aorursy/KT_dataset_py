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
#importing modules



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
#Reading dataset

df = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df.info()
df.head()
#Renaming columns inorder to avoid confusion

df.rename({'PAY_0':'PAY_1','default.payment.next.month':'DEFAULT'},axis = 1,inplace = True)

df.head()
#Since ID is the least inportant in this data.So removing ID column

df.drop(['ID'],axis = 1,inplace = True)

df.head()
#Let's check the unique values of columns

print({'Sex':df['SEX'].unique()},{'Education':df['EDUCATION'].unique()},{'Marriage':df['MARRIAGE'].unique()},sep='\n')
#Let's see the number of rows of missing values

len(df[(df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION']== 6) | (df['MARRIAGE'] == 0) ])
print('Percentage of Missing Values is {} %'.format(round(399/len(df) * 100,2)))
#Although imputer techniques can be used but removing the rows won't effect the outcome much

df_no_missing = df[(df['EDUCATION'] != 0) & (df['EDUCATION'] != 5) & (df['EDUCATION'] != 6) & (df['MARRIAGE'] != 0) ]

len(df_no_missing)
#Let's check the length of rows having default as 0 and as 1

print('No of people defaulted is {0}'.format(len(df[df['DEFAULT'] == 1])))



print('No of people not defaulted is {0}'.format(len(df[df['DEFAULT'] == 0])))
#importing library

from sklearn.utils import resample
#Let's take 1000 samples from each category of default 



#First splitting the dataset into default and not default dataset

df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]

df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]



#Now downsizing the dataset

df_no_default_downsampled = resample(df_no_default,

                                    replace = False,

                                    n_samples = 1000,

                                    random_state = 42)



df_default_downsampled  = resample(df_default,

                      replace = False,

                      n_samples = 1000,

                      random_state = 42)



df_downsampled = pd.concat([df_no_default_downsampled,df_default_downsampled],axis = 0)

len(df_downsampled)
#Splitting dataset into dependent(X) and independent(y) dataset



X = df_downsampled.drop(['DEFAULT'],axis = 1).copy()

y = df_downsampled['DEFAULT'].copy()
X_encoded = pd.get_dummies(X,columns = ['SEX','EDUCATION','MARRIAGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])



X_encoded.head()
#importing library

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_encoded,y,random_state = 42)
#importing library

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
#importing library

from sklearn.svm import SVC
#First making a model without any hyperparameter tuning

svc = SVC(random_state=42)

svc.fit(X_train_scaled,y_train)
#importing library

from sklearn.metrics import plot_confusion_matrix
#Confusion matrix for training set



plot_confusion_matrix(svc,X_train_scaled,y_train,display_labels = ['Did not Default','Default'])
#Confusion Matrix for test set



X_test_scaled = scaler.transform(X_test)

plot_confusion_matrix(svc,X_test_scaled,y_test,display_labels = ['Did not Default','Default'])
from sklearn.model_selection import GridSearchCV
param_grid = [

    {'C':[0.5,1,10,100,1000],

     'gamma':[1,0.1,0.001,0.0001],

     'kernel':['rbf']}]



svc_optimised = GridSearchCV(

    SVC(),

    param_grid,

    cv = 4,

    scoring = 'accuracy',

    verbose = 0)



svc_optimised.fit(X_train_scaled,y_train)

print(svc_optimised.best_params_)

svc_final = SVC(random_state=42,C = 1,gamma=0.001)

svc_final.fit(X_train_scaled,y_train)
#Confusion matrix for training set



plot_confusion_matrix(svc_final,X_train_scaled,y_train,display_labels = ['Did not Default','Default'])
#Confusion Matrix for test set



plot_confusion_matrix(svc_final,X_test_scaled,y_test,display_labels = ['Did not Default','Default'])
#importing library

from sklearn.decomposition import PCA
pca = PCA()



X_train_pca = pca.fit_transform(X_train_scaled)



per_var = np.round(pca.explained_variance_ratio_*100,decimals = 1)

labels = [str(x) for x in range(1,len(per_var) + 1)]



plt.bar(x = range(1,len(per_var) +1),height = per_var)

plt.tick_params(

    axis = 'x',

    which = 'both',

    bottom = False,

    top = False,

    labelbottom = False)



plt.ylabel('Percentage of Explained Varaince')

plt.xlabel('Principle Component')

plt.title('Scree Plot')

plt.show()
train_pc1 = X_train_pca[:,0]

train_pc2 = X_train_pca[:,1]



pca_trained_scaled = scaler.fit_transform(np.column_stack((train_pc1,train_pc2)))



param_grid_pca = [

    {'C':[0.5,1,10,100,1000],

    'gamma':[1,0.1,0.01,0.001,0.001],

    'kernel':['rbf']}]



optimal_params_pca = GridSearchCV(

    SVC(),

    param_grid_pca,

    cv = 4,

    scoring = 'accuracy',

    verbose = 0 )



optimal_params_pca.fit(pca_trained_scaled,y_train)

print(optimal_params_pca.best_params_)
svc_pca = SVC(random_state=42,C = 1000,gamma = 0.1)

svc_pca.fit(pca_trained_scaled,y_train)
X_test_pca = pca.transform(X_train_scaled)



test_pc1 = X_test_pca[:,0]

test_pc2 = X_test_pca[:,1]





x_min = test_pc1.min() - 1

x_max = test_pc1.max() + 1



y_min = test_pc2.min() - 1

y_max = test_pc2.max() + 1



xx,yy = np.meshgrid(np.arange(start = x_min,stop = x_max,step = 0.1),

                   np.arange(start = y_min,stop = y_max,step = 0.1 ))





Z = svc_pca.predict(np.column_stack((xx.ravel(),yy.ravel())))

Z = Z.reshape(xx.shape)
import matplotlib.colors as colors
fig,ax = plt.subplots(figsize = (10,10))



ax.contourf(xx,yy,Z,alpha = 0.1 )



cmap = colors.ListedColormap(['#e41a1c','#4daf4a'])



scatter = ax.scatter(test_pc1,test_pc2,c = y_train,cmap = cmap,s = 100,edgecolors = 'k',alpha = 0.7)



legend = ax.legend(scatter.legend_elements()[0],

                  scatter.legend_elements()[1],

                  loc = 'upper right')



legend.get_texts()[0].set_text('Not Default')

legend.get_texts()[1].set_text('Default')



ax.set_ylabel('PC2')

ax.set_xlabel('PC1')

ax.set_title('Classifer Visualisation using transformed featured by PCA')

plt.show()