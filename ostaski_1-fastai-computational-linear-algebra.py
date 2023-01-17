import numpy as np



#Exercise: Use Numpy to compute the answer to the above

A = np.array([[.9, .07, .02, .01],

             [0, .93, .05, .02],

             [0, 0, .85, .15],

             [0, 0, 0, 1]])



x = np.array([[.85, .1, .05, 0]])



A.T @ x.T # = (x @ A).T (T is transpose, @ is matrix multiplication in Python 3)
#Exercise: Use Numpy to compute the answer to the above

A = np.array([[6, 5, 3, 1],

             [3, 6, 2, 2],

             [3, 4, 3, 1]])



B = np.array([[1.5, 1],

             [2, 2.5],

             [5, 4.5],

             [16, 17]])



A @ B # @ is matrix multiplication in Python 3
def f(x):

    if x <= 1/2:

        return 2 * x

    if x > 1/2:

        return 2*x - 1
x = 1/10

for i in range(80):

    print(x)

    x = f(x)
import scipy.linalg as la 



A = np.array([[1., 1000], [0, 1]])

B = np.array([[1, 1000], [0.001, 1]])



print(A)



print(B)
np.set_printoptions(suppress=True, precision=4)



wA, vrA = la.eig(A)

wB, vrB = la.eig(B)



wA, wB
from IPython.display import YouTubeVideo

YouTubeVideo("PK_yguLapgA")
from IPython.display import YouTubeVideo

YouTubeVideo("3uiEyEKji0M")