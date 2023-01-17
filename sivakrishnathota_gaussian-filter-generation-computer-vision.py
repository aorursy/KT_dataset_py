import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sci
import scipy.stats as st
from scipy import signal
from scipy.ndimage import  filters
import matplotlib.pyplot as plt
x,y=np.mgrid[-1:3,-1:3]
print("X",x)
print("----------")
print("Y", y)

print("Gaussian Filter matrix : ")
filtermatrix=Gausianfilter(2,x,y)
print(filtermatrix)
print("Gaussian Filter shape : ", filtermatrix.shape)
def Gausianfilter(sigma,x,y):
    num=np.exp(-(x**2+y**2))/2*sigma**2
    den=np.sqrt(2*np.pi)*sigma**2
    return num/den
s=np.random.normal(0,1,[5,5])

x=np.linspace(-1,2,6)
ker1d=np.diff(st.norm.cdf(x))
print ("X",x)
print("Kernal 1D",ker1d)
print("Kernal 2D")
print(np.outer(ker1d,ker1d.T))
plt.plot([np.linspace(-1,2,5)],ker1d.reshape(1,5))
plt.show()
x=np.arange(-1,2,0.6)
print("X",x)
y=np.arange(-1,3,0.8)
print("Y",y)
matrix=Gausianfilter(2,x,y)
print("Gaussian 1D Filter",matrix)
filtermatrix=cv2.getGaussianKernel(5,2)
np.outer(filtermatrix,filtermatrix.T)
