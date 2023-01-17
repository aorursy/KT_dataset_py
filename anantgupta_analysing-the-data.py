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
# Read the data

data=pd.read_csv("../input/diabetes.csv")



data.shape

#768,9



data.columns.values

#Pregnancies Glucose BloodPressure SkinThickness Insulin BMI DiabetesPedigreeFunction Age Outcome



# First let us find out if there are any unknowns etc. We will have to impute them

data.isnull().values.any()



columnNames=list(data.columns.values)

columnNames.remove('Outcome')



from sklearn.decomposition import PCA

pca=PCA(n_components=5)

pca.fit(data[columnNames])

pca.explained_variance_ratio_



from mpl_toolkits.mplot3d import Axes3D

plot3D = plt.figure().gca(projection='3d')

plt1=plot3D.scatter(data[data['Outcome']==0]['Pregnancies'],data[data['Outcome']==0]['Glucose'],

                    data[data['Outcome']==0]['BloodPressure'],c='r',

                    label='0')

plt2=plot3D.scatter(data[data['Outcome']==1]['Pregnancies'],data[data['Outcome']==1]['Glucose'],

                    data[data['Outcome']==1]['BloodPressure'],c='b',

                    label='1')

plot3D.set_xlabel('Pregnancies')

plot3D.set_ylabel('Glucose')

plot3D.set_zlabel('BloodPressure')

plt.legend()

plt.show()



# Let us now plot the same using the Principal components

plt.figure()

dataPCA=data

ratios=pca.components_

dataPCA['PCA1']=(ratios[0][0] * dataPCA['Pregnancies']) + (ratios[0][1] * dataPCA['Glucose']) + (ratios[0][2] * dataPCA['BloodPressure'])

dataPCA['PCA2']=(ratios[1][0] * dataPCA['Pregnancies']) + (ratios[1][1] * dataPCA['Glucose']) + (ratios[1][2] * dataPCA['BloodPressure'])

dataPCA['PCA3']=(ratios[2][0] * dataPCA['Pregnancies']) + (ratios[2][1] * dataPCA['Glucose']) + (ratios[2][2] * dataPCA['BloodPressure'])



plot3D = plt.figure().gca(projection='3d')

plt1=plot3D.scatter(dataPCA[dataPCA['Outcome']==0]['PCA1'],dataPCA[dataPCA['Outcome']==0]['PCA2'],

                    dataPCA[dataPCA['Outcome']==0]['PCA3'],c='r',

                    label='0')

plt2=plot3D.scatter(dataPCA[dataPCA['Outcome']==1]['PCA1'],dataPCA[dataPCA['Outcome']==1]['PCA2'],

                    dataPCA[dataPCA['Outcome']==1]['PCA3'],c='b',

                    label='1')

plot3D.set_xlabel('PCA1')

plot3D.set_ylabel('PCA2')

plot3D.set_zlabel('PCA3')

plt.legend()

plt.show()
# SCATTER PLOT OF ALL THE FEATURES SPLIT BY OUTCOME. A GOOD WAY OF IDENITFYING A STRONG PATTERN IF ANY

data=pd.read_csv("../input/diabetes.csv")

colnames=list(data.columns.values)

# Remove the target variable

colnames.remove('Outcome')

arr1=np.linspace(1,len(colnames),len(colnames))



fig=plt.figure(figsize=(30,20))

fig.subplots_adjust(hspace=1,wspace=1)

counter=1

import itertools

for p in itertools.permutations(arr1,2):

    plt.subplot(len(colnames),len(colnames),counter)

    plt.scatter(data.ix[:,int(p[0])-1].values,data.ix[:,int(p[1])-1].values,c=data['Outcome'],alpha=0.5)

    plt.xlabel(colnames[int(p[0])-1])

    plt.ylabel(colnames[int(p[1])-1])

    counter=counter + 1

    

plt.show()
# CORRELATION PLOT OF THE FEATURES

from matplotlib.collections import EllipseCollection



def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)

    if not M.ndim == 2:

        raise ValueError('data must be a 2D array')

    if ax is None:

        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})

        ax.set_xlim(-0.5, M.shape[1] - 0.5)

        ax.set_ylim(-0.5, M.shape[0] - 0.5)



    # xy locations of each ellipse center

    xy = np.indices(M.shape)[::-1].reshape(2, -1).T



    # set the relative sizes of the major/minor axes according to the strength of

    # the positive/negative correlation

    w = np.ones_like(M).ravel()

    h = 1 - np.abs(M).ravel()

    a = 45 * np.sign(M).ravel()



    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,

                           transOffset=ax.transData, array=M.ravel(), **kwargs)

    ax.add_collection(ec)



    # if data is a DataFrame, use the row/column names as tick labels

    if isinstance(data, pd.DataFrame):

        ax.set_xticks(np.arange(M.shape[1]))

        ax.set_xticklabels(data.columns, rotation=90)

        ax.set_yticks(np.arange(M.shape[0]))

        ax.set_yticklabels(data.index)



    return ec



data=pd.read_csv("../input/diabetes.csv")

data = data.corr()

fig, ax = plt.subplots(1, 1)

m = plot_corr_ellipses(data, ax=ax, cmap='Reds')

cb = fig.colorbar(m)

cb.set_label('Correlation coefficient')

ax.margins(0.1)



# There are some good inferences from this correlation chart

# AGE and PREGNANCIES are very much correlated

# GLUCOSE has a very high explanatory power on OUTCOME

# SKINTHICKNESS and AGE have no correlation in this data set which was weird

# Let us now do a SVC on different feature selection techniques

# ICA ( Independent Component Analysis)

data=pd.read_csv("../input/diabetes.csv")

colnames=list(data.columns.values)

# Remove the target variable

colnames.remove('Outcome')



# Independent Component Analysis

from sklearn.decomposition import FastICA

fastICA=FastICA(n_components=3)

fastICA.fit(data[colnames])



# So if we go by ICA, then the 3 new features will be

mat2=np.array(fastICA.components_)

mat2=mat2.T

newData=np.dot(np.array(data[colnames]),mat2)

newData=pd.DataFrame(newData)

newData['Outcome']=data['Outcome']



# Let us now split it into test train and check the prediction power. We will use 

from sklearn import linear_model

from sklearn.cross_validation import train_test_split

from sklearn.svm import LinearSVC

svc=LinearSVC(random_state=0)

# Train the model using the training sets

newData.columns=['ICA1','ICA2','ICA3','Outcome']

X_train,X_test,Y_train,Y_test=train_test_split(newData[['ICA1','ICA2','ICA3']],newData['Outcome'],train_size=0.4,random_state=42)

svc.fit(X_train,Y_train)



from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,svc.predict(X_test))
