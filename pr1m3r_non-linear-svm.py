# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.random.seed(0)   # If you use the same seed you will get the same random numbers, good for repeatability



def create_target():

    x = []

    y = []

    for i in range(100):

        a = np.random.random() * 2 * np.pi   # Angle from the origin

        r = np.random.random() * 5   # This is the distance

        x.append(np.cos(a) * r)   # Getting Cartesian coords from polar coords

        y.append(np.sin(a) * r)   # Getting Cartesian coords from polar coords

    return x, y

        

            

X, y = create_target()   # Create lists





# Now we can plot the points

fig = plt.figure()

plt.scatter(X, y)

ax = fig.add_subplot(111)

plt.xlim(-10,10)

plt.ylim(-10,10)

ax.set_aspect('equal')

plt.xlabel('x')

plt.ylabel('y')

plt.title("One evenly distributed circle")

plt.show()

            
def create_outside():

    x = []

    y = []

    for i in range(100):

        a = np.random.ranf() * 2 * np.pi   # Angle from the origin

        r = np.random.ranf() * 5 + 5   # This is the distance, going to create a ring around the previous circle because of the +5

        x.append(np.cos(a) * r)   # Getting Cartesian coords from polar coords

        y.append(np.sin(a) * r)   # Getting Cartesian coords from polar coords

    return x, y



X2, y2 = create_outside()



fig2 = plt.figure()

plt.scatter(X, y, c='b')   # The first group is blue

plt.scatter(X2, y2, c='r')   # The second group is red

ax2 = fig2.add_subplot(111)

plt.xlim(-10, 10)

plt.ylim(-10, 10)

ax2.set_aspect('equal')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
data = pd.DataFrame(list(zip(X, y)), columns=['X', 'y'])

data['Target'] = 0   # The first set of points, closer to the origin, will be identified as 0

print(data.shape)

data.head()
data2 = pd.DataFrame(list(zip(X2, y2)), columns=['X', 'y'])

data2['Target'] = 1   # The second set of points will be identified as 1

print(data2.shape)

data2.head()
dataframes = [data, data2]

joined_data = pd.concat(dataframes)

joined_data.sample(n=5)
train = joined_data.drop(['Target'], axis=1)

target = joined_data['Target']



train.head()
target.head()
from sklearn.model_selection import train_test_split

train, test_train, target, test_target = train_test_split(train, target, test_size=0.2)   # Note that the test_size is 0.2, or 20% of the entire dataset, while 80% is used for training
from sklearn.svm import SVC

model = SVC(kernel='rbf', gamma=0.1, C=10, random_state=0)   # The model is a Support Vector Classifier with a Radial Basis Function kernel. gamma and C are adjustable hyperparameters.

model.fit(train, target)
training_accuracy = round(model.score(train, target) * 100, 2)

training_accuracy
accuracy = round(model.score(test_train, test_target) * 100, 2)

accuracy