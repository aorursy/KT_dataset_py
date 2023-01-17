# IMPORT LIBRARIES

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.animation import ArtistAnimation

from sklearn.linear_model import LogisticRegression



# LOAD THE DATA

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

train.head()
plt.figure(figsize=(15,15))

for i in range(5):

    for j in range(5):

        plt.subplot(5,5,5*i+j+1)

        plt.hist(test[str(5*i+j)],bins=100)

        plt.title('Variable '+str(5*i+j))

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(train['33'],train['65'],c=train['target'])

plt.plot([-1.6,1.4],[3,-3],':k')

plt.xlabel('variable 33')

plt.ylabel('variable 65')

plt.title('Training data')

plt.show()
# FIND NORMAL TO HYPERPLANE

clf = LogisticRegression(solver='liblinear',penalty='l2',C=0.1,class_weight='balanced')

clf.fit(train.iloc[:,2:],train['target'])

u1 = clf.coef_[0]

u1 = u1/np.sqrt(u1.dot(u1))
# CREATE RANDOM DIRECTION PERPENDICULAR TO U1

u2 = np.random.normal(0,1,300)

u2 = u2 - u1.dot(u2)*u1

u2 = u2/np.sqrt(u2.dot(u2))
# CREATE RANDOM DIRECTION PERPENDICULAR TO U1 AND U2

u3 = np.random.normal(0,1,300)

u3 = u3 - u1.dot(u3)*u1 - u2.dot(u3)*u2

u3 = u3/np.sqrt(u3.dot(u3))
# CREATE AN ANIMATION

images = []

steps = 60

fig = plt.figure(figsize=(8,8))

for k in range(steps):

    

    # CALCULATE NEW ANGLE OF ROTATION

    angR = k*(2*np.pi/steps)

    angD = round(k*(360/steps),0)

    u4 = np.cos(angR)*u1 + np.sin(angR)*u2

    u = np.concatenate([u4,u3]).reshape((2,300))

    

    # PROJECT TRAIN AND TEST ONTO U3,U4 PLANE

    p = u.dot(train.iloc[:,2:].values.transpose())

    p2 = u.dot(test.iloc[:,1:].values.transpose())

    

    # PLOT TEST DATA

    img1 = plt.scatter(p2[0,:],p2[1,:],c='gray')

    

    # PLOT TRAIN DATA (KEEP CORRECT COLOR IN FRONT)

    idx0 = train[ train['target']==0 ].index

    idx1 = train[ train['target']==1 ].index

    if angD<180:

        img2 = plt.scatter(p[0,idx1],p[1,idx1],c='yellow')

        img3 = plt.scatter(p[0,idx0],p[1,idx0],c='blue')

    else:

        img2 = plt.scatter(p[0,idx0],p[1,idx0],c='blue')

        img3 = plt.scatter(p[0,idx1],p[1,idx1],c='yellow')

        

    # ANNOTATE AND ADD TO MOVIE

    img4 = plt.text(1.5,-3.5,'Angle = '+str(angD)+' degrees')

    images.append([img1, img2, img3, img4])

    

# SAVE MOVIE TO FILE

ani = ArtistAnimation(fig, images)

ani.save('data.gif', writer='imagemagick', fps=15)
from IPython.display import Image

Image("../working/data.gif")