# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import random # for random value

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting 

from PIL import Image # Image processing

import json  #json file I/O

from mpl_toolkits.basemap import Basemap
with open('../input/shipsnet.json') as data_file:

    data = json.load(data_file)

Shipsnet= pd.DataFrame(data)

print(Shipsnet.head())

print(Shipsnet.shape)
# Function to Plotting Image of ships 

def Plotting_Image(array,pos):

    nrows = 2

    

    # Creating Figure and Axis to plot 

    fig, axes = plt.subplots(nrows)

    

    # appending size of plot

    fig.set_figheight(20)

    fig.set_figwidth(20)

    

    for row in axes:

        position = random.choice(pos)

        pixels = np.asarray(array[position])

        pix = pixels.reshape((3,6400))

        pix_T = pix.T

        imge = pix_T.reshape((80,80,3)).astype('uint8')

        plot(row,imge)

    plt.show()



def plot(axrow,imge):

    axrow.imshow(imge)
pos = [random.randint(0,9) for p in range(0,2000)]

Plotting_Image(Shipsnet['data'],pos)
a,b =[],[]

for i,j in Shipsnet['locations']:

    a.append(i)

    b.append(j)
plt.figure(figsize=(25,10))

m = Basemap(width=12000000,height=9000000,projection='lcc',

            resolution='c',lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

X, Y = m(a,b)

m.plot(X,Y,'o',color='r',markersize = 3)

m.drawcoastlines()

m.drawmapboundary(fill_color='aqua')

m.fillcontinents(color='g',lake_color='aqua')

plt.show()
def Loc_append(Location):

    a,b =[],[]

    for i,j in Location:

        a.append(i)

        b.append(j)

    return a,b

Location_Ship = Shipsnet[Shipsnet['labels']==1]['locations']

Location_NoShip = Shipsnet[Shipsnet['labels']==0]['locations']

ship_l1,ship_l2 = Loc_append(Location_Ship)

No_ship_l1,No_ship_l2 = Loc_append(Location_NoShip)

 
def plot_Ship(l1,l2,color):

    

    plt.figure(figsize=(25,10))

    m = Basemap(projection='mill',llcrnrlat=35,llcrnrlon=-124,urcrnrlat=39,urcrnrlon=-120,resolution='h')

    X, Y = m(l1,l2)

    m.plot(X,Y,'o',color=color,markersize = 3)

    m.drawcoastlines()

    m.drawcountries()

    m.drawmapboundary(fill_color='aqua')

    m.fillcontinents(color='g',lake_color='aqua')

    plt.show()
plot_Ship(ship_l1,ship_l2,'r')
plot_Ship(No_ship_l1,No_ship_l2,'b')
Shipsnet['labels'].value_counts()
#creating Target and Train Data 

Target = data['labels'] 

Train = np.asarray(data['data']).astype('uint8')
# Reducing Size of Train Data

train = Train/255
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(train)

train_pca = pca.transform(train)
from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train_pca,Target,test_size = 0.2 ,random_state = 123)
# importing Algorithms to fit model and predict Output



from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
# creating list of tuple wth model and its name  

models = []

models.append(('GNB',GaussianNB()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('DT',DecisionTreeClassifier()))

models.append(('RF',RandomForestClassifier()))

models.append(('LG',LogisticRegression()))

models.append(('SVC',SVC()))
from sklearn.cross_validation import cross_val_score



acc = []   # list for collecting Accuracy of all model

names = []    # List of model name



for name, model in models:

    

    acc_of_model = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')

    

    # appending Accuray of different model to acc List

    acc.append(acc_of_model)

    

    # appending name of models

    names.append(name)

    

    # printing Output 

    Out = "%s: %f" % (name, acc_of_model.mean())

    print(Out)
# Compare Algorithms Accuracy with each other on same Dataset

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(acc)

ax.set_xticklabels(names)

plt.show()