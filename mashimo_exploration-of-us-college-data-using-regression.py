# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

X = pd.read_csv("../input/College.csv", index_col=0)

X.info()
X.head()
X.Private = X.Private.map({'Yes':1, 'No':0})
X_rb  = X['Room.Board'] # series

y = X['Accept']



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_rb, y, 

                                                    test_size=0.3, random_state=7)
from sklearn import linear_model

model = linear_model.LinearRegression()
# fit(), score() and predict() expect 2d arrays

model.fit(X_train.values.reshape(-1,1), y_train)

X_test = X_test.values.reshape(-1,1)

score = model.score(X_test, y_test)

score
import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



matplotlib.style.use('ggplot') # Look Pretty





def drawLine(model, X_test, y_test, title, R2):

  # This convenience method will take care of plotting the

  # test observations, comparing them to the regression line,

  # and displaying the R2 coefficient

  fig = plt.figure()

  ax = fig.add_subplot(111)

  ax.scatter(X_test, y_test, c='g', marker='o')

  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)



  title += " R2: " + str(R2)

  ax.set_title(title)

  print (title)

  print ("Intercept(s): ", model.intercept_)



  plt.show()

drawLine(model, X_test, y_test, "Accept(Room&Board)", score)
X_en  = X['Enroll'] # series

X_train, X_test, y_train, y_test = train_test_split(X_en, y, 

                                                    test_size=0.3, random_state=7)



model.fit(X_train.values.reshape(-1,1), y_train)

X_test = X_test.values.reshape(-1,1)

score = model.score(X_test, y_test)



drawLine(model, X_test, y_test, "Accept(Enroll)", score)
X_fu  = X['F.Undergrad'] # series

X_train, X_test, y_train, y_test = train_test_split(X_fu, y, 

                                                    test_size=0.3, random_state=7)



model.fit(X_train.values.reshape(-1,1), y_train)

X_test = X_test.values.reshape(-1,1)

score = model.score(X_test, y_test)



drawLine(model, X_test, y_test, "Accept(F.Undergrad)", score)

X_rb_en = X[['Room.Board', 'Enroll']] # data frame

X_train, X_test, y_train, y_test = train_test_split(X_rb_en, y, 

                                                    test_size=0.3, random_state=7)



model.fit(X_train, y_train)

score = model.score(X_test, y_test)

score
import numpy as np



def drawPlane(model, X_test, y_test, title, R2):

  # This convenience method will take care of plotting the

  # test observations, comparing them to the regression plane,

  # and displaying the R2 coefficient

  fig = plt.figure()

  ax = Axes3D(fig)

  ax.set_zlabel('prediction')



  # You might have passed in a DataFrame, a Series (slice),

  # an NDArray, or a Python List... so let's keep it simple:

  X_test = np.array(X_test)

  col1 = X_test[:,0]

  col2 = X_test[:,1]



  # Set up a Grid. We could have predicted on the actual

  # col1, col2 values directly; but that would have generated

  # a mesh with WAY too fine a grid, which would have detracted

  # from the visualization

  x_min, x_max = col1.min(), col1.max()

  y_min, y_max = col2.min(), col2.max()

  x = np.arange(x_min, x_max, (x_max-x_min) / 10)

  y = np.arange(y_min, y_max, (y_max-y_min) / 10)

  x, y = np.meshgrid(x, y)



  # Predict based on possible input values that span the domain

  # of the x and y inputs:

  z = model.predict(  np.c_[x.ravel(), y.ravel()]  )

  z = z.reshape(x.shape)



  ax.scatter(col1, col2, y_test, c='g', marker='o')

  ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)

  

  title += " R2: " + str(R2)

  ax.set_title(title)

  print (title)

  print ("Intercept(s): ", model.intercept_)

  

  plt.show()

drawPlane(model, X_test, y_test, "Accept(Room&Board,Enroll)", score)