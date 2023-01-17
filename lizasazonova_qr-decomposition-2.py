# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
def mag(a):

    """Calculate the magnitude of vector a"""

    mag_value = np.sqrt(np.sum(a**2))

    return mag_value



def get_qr_decomposition(A):

    q = []

    U = []

    cols = A.shape[1]



    for i in np.arange(cols):    



        a = A[:, i]



        for j in np.arange(i+1):



            if j == 0:        

                u = a



            else: 

                u -= np.dot(q[j], a)*q[j]



        U = np.append(U, u)

        q = np.append(q, (u/mag(u)))



    Q = np.array(q.reshape(int(len(q)/cols), cols)).T

    R = np.zeros(A.shape)

    

    for i in range(R.shape[0]):

        rowvals    = np.dot(Q[:,i], A)

        R[i,i]     = mag(U[i])

        R[i, i+1:] = rowvals[i+1:]

    print(np.dot(Q, Q.T))

    return Q, R

#Q.T.dot(A).dot(Q), Q



def find_eigenvalues(A, tol=0.1, vecs=True, A_progress=False):

    """A must be Hermitian. Find eigenvalues with QR decomposition.

    tol:  tolerance threshold where the matrix is diagonal enough

    vecs: return eigenvectors? (otherwise, only eigenvalues)

    A_progress: return intermediate QR-decompositions of input matrix?"""

    

    decompose = True

    

    As = []; 

    V = np.ones(A.shape)

    while decompose:



        # Check if the off-diagonal elements are less than the threshold

        A_offiag  = A.copy()

        np.fill_diagonal(A_offdiag, 0)        # Replace diagonal elements with 0 for the check

        A_offdiag = np.abs(A_offdiag)         # Find absolute values of off-diagonal elements

        mask      = A_offdiag > tol           # Set all elements above tolerance to 1, otherwise 0

        if np.sum(tol_check) == 0:            # If any one element is above tolerance, continue QR

            decompose = False



        # Calculate QR decomposition.

        A, Q = get_qr(A, qs)

        

        if A_progress: As.append(A)

        if vecs: V.dot(Q)

    

    # Compute eigenvalues

    eigvals = np.diag(A)

    return eigvals, V, As

    





A = np.array([[1, 4, 8, 4],

            [4, 2, 3, 7],

            [8, 3, 6, 9],

            [4, 7, 9, 2]],

            float)





# eigvals, V, As = find_eigenvalues(A, tol=0.1, vecs=True, A_progress=True)
Q, R = get_qr_decomposition(A)
As = []



for n in range(20):

    As.append(np.random.rand(100, 100))
plt.imshow(As[0])
import matplotlib.pyplot as plt

import matplotlib.animation as animation



fig = plt.figure(); ax = plt.axes()





# ims is a list of lists, each row is a list of artists to draw in the

# current frame; here we are just animating one artist, the image, in

# each frame

ims = []

for step, Astep in enumerate(As):

    im = plt.imshow(Astep, animated=True)

    t  = ax.annotate(step,(1,1))

    ims.append([im, t])



ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,

                                repeat_delay=1000)

ani.save('dynamic_images.mp4',fps = 2)

plt.show()
