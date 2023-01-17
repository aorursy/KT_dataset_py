#lets code some stuff
import numpy as np
#scalar
sc=69
sc
#vectors
vec = np.array([5,7,-8,22,1,3])
vec
type(vec)
type(sc)
#matrices 
mat = np.array([[1,2,3],[4,5,6]])
mat
type(mat)
mat.shape
vec.shape
vec.reshape(3,2)
vec.reshape(6,1)
#convert the vector to matrix
vec = vec.reshape(2,3)
#merge both of them to create a tensor
tensor = np.array([vec,mat])
tensor.shape
tensor
vec.shape

mat.shape
add = vec+mat
sub = vec-mat
add

sub
sub-222
add-222
#.T the matrix to see the magic.That's all
(add-222).T

#.T the matrix to see the magic.That's all
(sub-222).T

#lets get the vector to it's original shape for dot product
#you cannot reshape that we need to ravel that
vec1 = vec.ravel()
vec2 = vec.ravel()*2

vec1.shape
type(vec)
#multipy by scalar
np.dot(vec1,3)
#multiply 2 vectors = a scalar
np.dot(vec1,vec2)