# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.svm import SVC # support vector machine

from scipy import misc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input/test"]).decode("utf8"))



n_samples=100

def load_image(imagename,showimg=False):

    rawimg=misc.imread(imagename)

    scaledimage=misc.imresize(rawimg,(100,100))

    if showimg:

        misc.imshow(scaledimage)

    return scaledimage.ravel()



#face = misc.imread('../input/test/cat.1.jpg')

cat_images=[load_image("../input/train/cat.%d.jpg" %i) for i in range(n_samples)]

dog_images=[load_image("../input/train/dog.%d.jpg" %i) for i in range(n_samples)]

load_image("../input/train/dog.1.jpg",True)

training_samples=np.array(cat_images+dog_images)

print(training_samples.shape)

labels=["cat" for c in cat_images]+["dog" for d in dog_images]

clf = SVC()

clf.fit(training_samples,labels)



clf.predict([load_image("../input/test/%d.jpg" %i) for i in range(10,15)])



#type(face)      

#face.shape, face.dtype

# Any results you write to the current directory are saved as output.