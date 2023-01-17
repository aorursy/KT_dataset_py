import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


from surprise import Reader, Dataset

# Define the format

reader = Reader(line_format='user item rating timestamp', sep='\t')



# 1. Rating dataset

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=r_cols,

                      encoding='latin-1')



data = Dataset.load_from_file('../input/ml-100k/u.data', reader=reader)



#ratings

print(ratings.shape)

ratings.head(4)
from surprise import NormalPredictor

from surprise import Dataset

from surprise.model_selection import cross_validate
# Split data into 5 folds

data.split(n_folds=5)
# We'll use the famous NormalPredictor  algorithm from Random.

algo = NormalPredictor()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
# Retrieve the trainset.

trainset = data.build_full_trainset()

algo.fit(trainset)
userid = str(196)

itemid = str(302)

actual_rating = 4

print(algo.predict(userid, 302, 4))
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D



# Our 2-dimensional distribution will be over variables X and Y

N = 60

X = np.linspace(-3, 3, N)

Y = np.linspace(0, 5, N)

X, Y = np.meshgrid(X, Y)



# Mean vector and covariance matrix

mu = np.array([0., 1.])

Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])



# Pack X and Y into a single 3-dimensional array

pos = np.empty(X.shape + (2,))

pos[:, :, 0] = X

pos[:, :, 1] = Y



def multivariate_gaussian(pos, mu, Sigma):

    """Return the multivariate Gaussian distribution on array pos.



    pos is an array constructed by packing the meshed arrays of variables

    x_1, x_2, x_3, ..., x_k into its _last_ dimension.



    """



    n = mu.shape[0]

    Sigma_det = np.linalg.det(Sigma)

    Sigma_inv = np.linalg.inv(Sigma)

    N = np.sqrt((2*np.pi)**n * Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized

    # way across all the input variables.

    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)



    return np.exp(-fac / 2) / N



# The distribution on the variables X, Y packed into pos.

Z = multivariate_gaussian(pos, mu, Sigma)



# Create a surface plot and projected filled contour plot under it.

fig = plt.figure(figsize=(20,10))

ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,

                cmap=cm.viridis)



cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)



# Adjust the limits, ticks and view angle

ax.set_zlim(-0.15,0.2)

ax.set_zticks(np.linspace(0,0.2,5))

ax.view_init(27, -21)



plt.show()
import pandas as pd 

  

# initialize list of lists 

data = [[196,243,3], [186,302,3], [22,377,1],[244,51,5]] 

  

# Create the pandas DataFrame 

df = pd.DataFrame(data, columns = ['user_id', 'movie_id','ratings']) 

  

# print dataframe. 

df 
from scipy.stats import multivariate_normal

F = multivariate_normal(mu, Sigma)

Z = F.pdf(pos)
#CONVERTING DATAFRAME TO ARRAY

dfarr = df.values

print(dfarr[0])

dfarr
#CALCULATING COVARIANCE OF ARRAY

covarr = np.cov(dfarr)

covarr
#CALCULATING MEAN OF ARRAY

umean = np.mean(dfarr,axis=1)

umean

import matplotlib.pyplot as plt

x, y = np.mgrid[-1:1:.01, -1:1:.01]

pos = np.empty(x.shape + (2,))

pos[:, :, 0] = x; pos[:, :, 1] = y

rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])

plt.contourf(x, y, rv.pdf(pos))
# import matplotlib.pyplot as plt

# plt.plot(x,y)