# use numpy and pandas

import numpy as np

import pandas as pd



# We need sklearn for preprocessing and for the TSNE Algorithm.

import sklearn

from sklearn.preprocessing import Imputer, scale

from sklearn.manifold import TSNE



# WE employ a random state.

RS = 20150101



# We'll use matplotlib for graphics.

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline



# We import seaborn to make nice plots.

import seaborn as sns

# palette so that each person is having a color

palette = np.array(sns.color_palette("hls", 6))
data = pd.read_csv("../input/dataset_no_outlier.csv")

data.head()
data.info()
sns.countplot(x='ClassLabel', data=data, palette=palette[1:6])
data.describe()
#convert pandas data to np array type as DS

DS=np.array(data.copy())

#slice classlabels as CL

CL=DS[:,[4]]

#slice X channel as DX

DX=DS[:,[0,4]]

DSS=np.zeros(901)

#print(DX)

print("DX shape:",DX.shape)

for x in range(1,6):

    bl=np.where(DX[:,1]==x)

    DX_1=DX[bl]

    DX_1=DX_1[0:900,0]

    DX_1=np.hstack((DX_1,np.ones(1)*x))

    DSS=np.vstack((DSS,DX_1))

DSS=DSS[1:6,:]

print(DSS.shape)

print(DSS)

    





#X = data.copy()

X=DSS.copy()



# now we sort for the target

#X.sort_values(by='ClassLabel', inplace=True)



# We split the target off the features and store it separately

#y = X['ClassLabel']

#X.drop('ClassLabel', inplace=True, axis=1)

y=X[:,900]

X=X[:,0:900]



# make sure the target is not part of the input data any more

#assert 'ClassLabel' not in X.columns



# make sure the target is as expected and turn it into an array

#assert set(y.unique()) == {1, 2, 3, 4, 5}

#y = np.array(y)





# we scale the data

X = scale(X) 

print(X.shape)

print(y.shape)

# run the Algorithm

handtremor_proj = TSNE(random_state=RS).fit_transform(X)

handtremor_proj.shape
# choose the palette

palette = np.array(sns.color_palette("hls", 6))



# plot the result

def scatter_plot(x, colors, ax):

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,

                    c=palette[colors.astype(np.int)])

    ax.axis('off')

    ax.axis('tight')    

    return sc



# plot the legend

def legend_plot(font_size=14):

    patch1 = mpatches.Patch(color=palette[1], label='Person 1')

    patch2 = mpatches.Patch(color=palette[2], label='Person 2')

    patch3 = mpatches.Patch(color=palette[3], label='Person 3')

    patch4 = mpatches.Patch(color=palette[4], label='Person 4')

    patch5 = mpatches.Patch(color=palette[5], label='Person 5')

    plt.legend(handles=[patch1, patch2, patch3, patch4, patch5], fontsize=font_size, loc=4)
f = plt.figure(figsize=(8, 8))

f.suptitle('Geometry of Handtremor for 5 persons', fontsize=20)

ax = plt.subplot(aspect='equal')

scatter_plot(handtremor_proj, y, ax)

legend_plot()