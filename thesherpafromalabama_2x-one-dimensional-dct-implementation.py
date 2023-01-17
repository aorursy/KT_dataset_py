
"""
Code based on resorces from:
https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
https://github.com/gk7huki/pydct/blob/master/dct.py
https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms
https://www.youtube.com/watch?v=mGWSbGoMrI4
https://stackoverflow.com/questions/13171329/dct-2d-without-fft
https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/dct.htm
https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
"""

# Import all relevant libraries here:

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import r_
from skimage import io
from math import cos
print(os.listdir("../input"))

# Load in the image
f = io.imread('../input/zelda.png', as_gray=True)
print('image matrix size: ', f.shape )

## Sample smaller size
#pos = 175
#size = 32
#f = f[pos:pos+size,pos:pos+size]

# Let's see how she looks!
plt.imshow(f, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
n = 8  # This will be the window in which we perform our DCT
sumd = 0 # INI value

# Create some blank matrices to store our data

dctmatrix = np.zeros(np.shape(f)) # Create a DCT matrix in which to plug our values :)
f = f.astype(np.int16) # Convert so we can subtract 128 from each pixel. The default uint only ranges from 0-255
f = f-128 # As said above
f2 = np.zeros(np.shape(f)) # This will be where the compressed image goes

def cosp(i,j,n): # This is the funky cos function inside the DCT
    output = cos(((2*i)+1)*j*math.pi/(2*n))
    return output

def OneDPassX(f,n,p,a): # p denoes which row in the matrix to go after
    sum1D = 0
    temp = []
    for u in range(n):
        for m in range(n):   #range(a,a+n)
            sum1D += f[p,m+a]*cosp(m,u,n)
        if u == 0: sum1D *= math.sqrt(1/n)
        if u > 0: sum1D *= math.sqrt(2/n)
        temp.append(sum1D)
        sum1D = 0 # Need to reset this after each loop
    return temp

def Xiterations():
    for a in r_[0:np.shape(f)[1]:n]:
        for p in r_[0:np.shape(f)[0]]:
            dctmatrix[p,a:a+n] = OneDPassX(f,n,p,a)
    return dctmatrix

def OneDPassY(raw,n,p,a): # p denoes which row in the matrix to go after
    sum1D = 0
    temp = []
    for u in range(n):
        for m in range(n):
            sum1D += raw[m+a,p]*cosp(m,u,n)
        if u == 0: sum1D *= math.sqrt(1/n)
        if u > 0: sum1D *= math.sqrt(2/n)
        temp.append(sum1D)
        sum1D = 0 # Need to reset this aft5er each loop
    return temp

def Yiterations():
    for a in r_[0:np.shape(f)[0]:n]:
        for p in r_[0:np.shape(f)[1]]:
            newmatrix[a:a+n,p] = OneDPassY(dctmatrix,n,p,a)
    return newmatrix

Xiterations()
plt.figure()
plt.imshow(dctmatrix,cmap='gray',vmax = np.max(dctmatrix)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image after one 1-D pass")
newmatrix = np.zeros(np.shape(f)) # Need a new blank to deposite our values
Yiterations()
plt.figure()
plt.imshow(newmatrix,cmap='gray',vmax = np.max(dctmatrix)*0.01,vmin = 0)
plt.title( "8x8 DCTs of the image after two 1-D passes")
Quant = np.array([
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
])

# Convolve nxn sections across entire image to return    
# As factor variable increases, the image will compress even more
factor = 4
for a in r_[0:np.shape(f)[0]:n]:
    for b in r_[0:np.shape(f)[1]:n]:
        newmatrix[a:a+n,b:b+n] = newmatrix[a:a+n,b:b+n]/Quant*factor

# Let's take a look at how much it compressed the DCT...
# Display entire DCT
plt.figure()
plt.imshow(newmatrix,cmap='gray',vmax = np.max(newmatrix)*0.01,vmin = 0)
plt.title("8x8 DCTs of the image")
def OneDInvPassX(F,n,p,a): # p denoes which row in the matrix to go after
    sum1D = 0
    temp = []
    
    for m in range(n):  # range(n):
        val1 = 0
        for u in range(n):   #range(n):   #range(a,a+n)
            if u == 0: val1 = math.sqrt(1/n)
            if u > 0: val1 = math.sqrt(2/n)
            sum1D += F[p,u+a]*val1*cosp(m,u,n)
        temp.append(sum1D)
        sum1D = 0 # Need to reset this after each loop
    return temp

def InvXiterations(matrix):
    for a in r_[0:np.shape(f)[1]:n]:
        for p in r_[0:np.shape(f)[0]]:
            f2[p,a:a+n] = OneDInvPassX(matrix,n,p,a)
    return f2

def OneDInvPassY(F,n,p,a): # p denoes which row in the matrix to go after
    sum1D = 0
    temp = []
    
    for m in range(n):
        val1 = 0
        for u in range(n):   #range(a,a+n)
            if u == 0: val1 = math.sqrt(1/n)
            if u > 0: val1 = math.sqrt(2/n)
            sum1D += F[u+a,p]*val1*cosp(m,u,n)
        temp.append(sum1D)
        sum1D = 0 # Need to reset this after each loop
    return temp

def InvYiterations(matrix):
    for a in r_[0:np.shape(newmatrix2)[0]:n]:
        for p in r_[0:np.shape(newmatrix2)[1]]:
            newmatrix2[a:a+n,p] = OneDInvPassY(matrix,n,p,a)
    return newmatrix2
InvXiterations(newmatrix)
plt.figure()
plt.imshow(f2,cmap='gray',vmax = np.max(f2)*0.01,vmin = 0)
plt.title("8x8 IDCTs of the image after one 1D pass")
newmatrix2 = np.zeros(np.shape(f)) # Need new matrix to feed values into
InvYiterations(f2)
plt.figure()
plt.imshow(newmatrix,cmap='gray',vmax = np.max(f2)*0.01,vmin = 0)
plt.title("8x8 IDCTs of the image after two 1D passes")

newmatrix2 = newmatrix2 + 128

plt.imshow(newmatrix2, cmap=plt.cm.gray)
plt.axis('off')
plt.title("Compressed Image")
plt.show()

plt.imshow(f, cmap=plt.cm.gray)
plt.axis('off')
plt.title("Original Image")
plt.show()