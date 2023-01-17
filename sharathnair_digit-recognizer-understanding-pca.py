# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py

from numpy import linalg as LA

py.init_notebook_mode(connected=True)



import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
#var_x1=0, var_x2=3

mu1,mu2,N=0,3,10

def plot_gaussian(sigma1,sigma2):     #var = sigma*sigma

    x1=np.random.normal(mu1,sigma1,N)    # gaussian distributed feature x1

    x2=np.random.normal(mu2,sigma2,N)    # gaussian distributed feature x2

    trace=go.Scatter(

    x=x1,

    y=x2,

    mode='markers',

    )

    data=[trace]

    layout=go.Layout(

        xaxis=dict(

                range=[-100,100]

        ),

        yaxis=dict(

                range=[-100,100]

        ),

    )

    fig=dict(data=data,layout=layout)

    py.iplot(fig)
plot_gaussian(sigma1=0,sigma2=0)
plot_gaussian(sigma1=10,sigma2=0)
plot_gaussian(sigma1=10,sigma2=2)




from  numpy.random import multivariate_normal as mvr_gauss

from sklearn.preprocessing import StandardScaler



number_samples=5000

mu=np.array([6.0,4.0])   # define the means of the data distributed along both the axes, I chose mu-x=6 and mu-y=4

#now the desired covariance matrix

#the diagonal elements of the covariance matrix indicates the variance along the feature 

# while the off-diagonal elements shows the variance between the two features. i.e the covariance

cvr=np.array([

    [15,-4],                           

    [-4,4]

])



#Generate the random samples

xf=mvr_gauss(mu,cvr,number_samples)    # xf is now 5000 samples each with 2 features i.e a 5000 X 2 matrix



stdsc=StandardScaler()                #standardization of data

xf_std=stdsc.fit_transform(xf)



#Create a plotly trace

trace=go.Scatter(

    x=xf_std[:,0],              #xf[:,0] is the feature1

    y=xf_std[:,1],              #xf[:,1] is the feature2

    mode='markers',

        

)

data=[trace]



#layout for naming the axes

layout=go.Layout(

            title='example of 2D correlated data', autosize=False,

        

            xaxis=dict(title= 'feature 1',range=[-10,20]),

            yaxis=dict(title= 'feature 2')#,range=[-10,20]),

         )





fig=dict(data=data,layout=layout)

py.iplot(fig)





pxd=pd.DataFrame(xf_std)



#calculate the covariance matrix of the dataframe

sig=pxd.cov()



#calculate the eigenvectors and eigenvalues

eigvals,eigvecs=LA.eig(sig)

#Create a plotly trace

import plotly.figure_factory as ff

from plotly import tools



trace1=go.Scatter(

    x=xf_std[:,0],              #xf_std[:,0] is the feature1

    y=xf_std[:,1],              #xf_std[:,1] is the feature2

    mode='markers',

    opacity=0.5,

    name='data'

    

)



#fig=tools.make_subplots(1,1)

mu_x=0

mu_y=0

x0,y0=[mu_x,mu_x],[mu_y,mu_y]

u,v=eigvecs[:,0]*5,eigvecs[:,1]*5



scale=1

fig=ff.create_quiver(x0,y0,u,v,scale,arrow_scale=0.03)

'''fig['layout']['autosize']=False

fig['layout']['xaxis']['range']=[-8,8]

fig['layout']['yaxis']['range']=[-8,8]

'''

fig.data[0].name='eigenvectors'

#layout for naming the axes

'''layout=go.Layout(

            title='example of 2D correlated data',

            autosize= False,

            xaxis=dict(title= 'feature 1'),

            yaxis=dict(title= 'feature 2'),

            

         )'''

figg=tools.make_subplots(rows=1,cols=1)

#fig.append_trace(trace1,1,1)

figg.add_trace(fig.data[0],1,1)

figg.add_trace(trace1,1,1)

figg.layout.autosize=False

figg.layout.xaxis.range=[-6,6]

figg.layout.yaxis.range=[-6,6]



#fig.layout=layout



#fig=dict(data=data,layout=layout)

py.iplot(figg)
t=np.array(eigvecs).T

xf_p=np.dot(xf_std,t)

#Create a plotly trace

import plotly.figure_factory as ff

from plotly import tools



trace1=go.Scatter(

    x=xf_std[:,0],              #xf_std[:,0] is the feature1

    y=xf_std[:,1],              #xf_std[:,1] is the feature2

    mode='markers',

    opacity=0.5,

    name='original standardized data'

        

)

#Transformed data

trace_transformed=go.Scatter(

    x=xf_p[:,0],

    y=xf_p[:,1],

    mode='markers',

    opacity=0.8,

    name='transformed data' 

    

)



#fig=tools.make_subplots(1,1)

mu_x=0

mu_y=0

x0,y0=[mu_x,mu_x],[mu_y,mu_y]

u,v=eigvecs[:,0]*5,eigvecs[:,1]*5

figg=tools.make_subplots(rows=1,cols=2)

scale=1

fig=ff.create_quiver(x0,y0,u,v,scale,arrow_scale=0.03)

fig['layout']['autosize']=False

fig.data[0]['name']='eigenvectors'



#layout for naming the axes

'''layout=go.Layout(

            title='example of 2D correlated data',

            autosize= True,

            xaxis=dict(title= 'feature 1'),

            yaxis=dict(title= 'feature 2'),

            

         )

'''



figg.add_trace(fig.data[0],1,1)

figg.add_trace(trace1,1,1)

figg.add_trace(trace_transformed,1,2)

figg.layout['xaxis2']['range']=[-6,6]

figg.layout['yaxis2']['range']=[-6,6]

figg.layout['xaxis2']['title']='eigenvector 1'

figg.layout['yaxis2']['title']='eigenvector 2'

figg.layout['xaxis']['title']='feature 1'

figg.layout['yaxis']['title']='feature 2'



py.iplot(figg)
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

#constants

NUMBER_OF_TRAINING_IMGS=5000



#load the MNIST dataset

labeled_images=pd.read_csv('../input/train.csv') 

images=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,1:] # first NUMBER_OF_TRAINING_IMGS rows,column 2 onwards.

labels=labeled_images.iloc[0:NUMBER_OF_TRAINING_IMGS,:1] #first NUMBER_OF_TRAINING_IMGS rows, first column. 



#split into train-test

train_images,test_images,train_labels,test_labels=train_test_split(images,labels,test_size=0.2,random_state=13)



#standardize the data

#stdsc=StandardScaler()

stdsc=MinMaxScaler()

train_images_std=stdsc.fit_transform(train_images)

test_images_std=stdsc.transform(test_images)

#perform PCA on training data,getting all possible eigenvectors

pca=PCA(svd_solver='randomized',whiten=True)

pca.fit(train_images_std)

#print(pca.n_components_)

#print(pca.explained_variance_ratio_)

def cummulative(ll):

    nl=np.empty(len(ll))

    for i in range(len(ll)):

        if i==0:

            nl[i]=ll[i]

        else:

            nl[i]=nl[i-1]+ll[i]

    #print(nl)

    return nl/10





fig=tools.make_subplots(rows=1,cols=1)

bardata=go.Bar(

    x=[xx for xx in range(pca.n_components_)],

    y=[xx for xx in pca.explained_variance_ratio_],

    opacity=1.0,

    name='explained variance ratio'

)



cummulativeData=go.Bar(

    x=[xx for xx in range(pca.n_components_)],

    y=cummulative(pca.explained_variance_ratio_),

    opacity=0.4,

    name='cumulative sum (divided by 10)'

    )





#data=[bardata,cummulativeData]

fig.add_trace(bardata,1,1)

fig.add_trace(cummulativeData,1,1)

py.iplot(fig)



pca=PCA(n_components=144,svd_solver='randomized',whiten=True)

train_images_pca=pca.fit_transform(train_images_std)

test_images_pca=pca.transform(test_images_std)
train_images_pca=pd.DataFrame(train_images_pca)

train_images_pca.head()
import matplotlib.pyplot as plt





pd_train_images_std=pd.DataFrame(train_images_std)



fig,axes=plt.subplots(figsize=(10,10),ncols=2,nrows=2)

axes=axes.flatten()

for i in range(0,4):

    jj=np.random.randint(0,train_images_std.shape[0])          #pick a random image

    if i%2==0 :

        IMG_HEIGHT=12

        IMG_WIDTH=12

        axes[i].imshow(train_images_pca.iloc[[jj]].values.reshape(IMG_HEIGHT,IMG_WIDTH))

    else:

        IMG_HEIGHT=28

        IMG_WIDTH=28

        axes[i].imshow(pd_train_images_std.iloc[[jj]].values.reshape(IMG_HEIGHT,IMG_WIDTH))

    
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(criterion='gini',random_state=1)

train_labels.shape

forest.fit(train_images_pca,train_labels.values.ravel())



forest.score(train_images_pca,train_labels.values.ravel())
forest.score(test_images_pca,test_labels.values.ravel())
subTest=pd.read_csv('../input/test.csv')

subTest_sc=stdsc.transform(subTest)

pred=forest.predict(pca.transform(subTest_sc))

submissions=pd.DataFrame({'ImageId':list(range(1,len(pred)+1)), 'Label':pred})

submissions.head()
submissions.to_csv("mnist_pca_randForests_submit.csv",index=False,header=True)
!ls