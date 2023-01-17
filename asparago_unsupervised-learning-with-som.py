#As usual we start importing a number of libraries that will be come in handy later on
import numpy as np 
import pandas as pd 
import seaborn as sns
from imageio import imwrite
#from scipy.misc import imsave
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageChops
import SimpSOM as sps
np.random.seed(0)

#We load here the data from the provided training set, we randomly select 500 landmark points and separate the labels. 
train = pd.read_csv('../input/train.csv')
train = train.sample(n=500, random_state=0)
labels = train['label']
train = train.drop("label",axis=1)

#Let's plot the distribution and see if the distribution is uniform
sns.distplot(labels.values,bins=np.arange(-0.5,10.5,1))

#Then we normalize the data, a crucial step to the correct functioning of the SOM algorithm
trainSt = StandardScaler().fit_transform(train.values)
#We build a 40x40 network and initialise its weights with PCA 
net = sps.somNet(40, 40, trainSt, PBC=True, PCI=True)

#Now we can train it with 0.1 learning rate for 10000 epochs
net.train(0.1, 10000)

#We print to screen the map of the weights differences between nodes, this will help us identify cluster centers 
net.diff_graph(show=True,printout=True)
#Here we first define a few useful functions
def autocrop(fileName):
    im = Image.open(fileName)
    im=im.crop((0,100,2900,im.size[1]))
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def posMap(x,y):
     if y%2==0:
        return [y, x*2/np.sqrt(3)*3/4]
     else:
        return [y+0.5, x*2/np.sqrt(3)*3/4]
    
def posCount(x,y):
     return y*40+x

def posCountR(x):
     return [np.int(x%40),np.int(x/40)]
#Let's print a few trained nodes' weights and see how good they are
listNodes=[[20,0],[23,11],[1,6],[13,37],[7,33],[18,31]]
listCount=[posCount(20,0), posCount(23,11), posCount(1,6), posCount(13,37), posCount(7,33), posCount(18,31)]

i=0
for node in net.nodeList:
    if i in listCount:
        print('Node\'s position: {:d} {:d}'.format(posCountR(i)[1], posCountR(i)[0]) )
        plt.imshow(np.asarray(node.weights).reshape(28,28))
        plt.axis('off')
        plt.show()
    i+=1
projData=net.project(trainSt[:500])
#We first save a cropped version of the original map to superimpose and then we add the scatterpoints
cropped = autocrop('nodesDifference.png')
cropped.save('cropped.png')
#And here we prepare the plotly graph. 
trace0 = go.Scatter(
    x = [x for x,y in projData],
    y = [y for x,y in projData],
#    name = labels,
    hoveron = [str(n) for n in labels],
    text = [str(n) for n in labels],
    mode = 'markers',
    marker = dict(
        size = 8,
        color = labels,
        colorscale ='Jet',
        showscale = False,
        opacity = 1
    ),
    showlegend = False

)
data = [trace0]

layout = go.Layout(
    images= [dict(
                  source= "cropped.png",
                  xref= "x",
                  yref= "y",
                  x= -0.5,
                  y= 39.5*2/np.sqrt(3)*3/4,
                  sizex= 40.5,
                  sizey= 40*2/np.sqrt(3)*3/4,
                  sizing= "stretch",
                  opacity= 0.5,
                  layer= "below")],
    width = 800,
    height = 800,
    hovermode= 'closest',
    xaxis= dict(
        range=[-1,41],
        zeroline=False,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    yaxis=dict(
        range=[-1,41],
        zeroline=False,
        showgrid=False,
        ticks='',
        showticklabels=False
    ),
    showlegend= True
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')