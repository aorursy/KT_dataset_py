# the first step of using numpy is to tell python to use it

import numpy as np
print(np.cos(np.pi))

print(np.sqrt(1.21))

print(np.log(np.exp(5.2)))
# we can create numpy arrays by converting lists

# this is a vector

vec = np.array([1,2,3])

print(vec)

# we can create matrices by converting lists of lists

mat = np.array([[1,2,1],[4,5,9],[1,8,9]])

print('')

print(mat)

print('')

print(mat.T)
# there are lots of other ways to create numpy arrays

vec2 = np.arange(0,15)

print(vec2)

print('')

vec3 = np.arange(3,21,6)

print(vec3)



vec4 = np.linspace(0,5,10)

print(vec4)

print('')

print(vec4.reshape(5,2))

vec4_reshaped = vec4.reshape(5,2)

print(vec4_reshaped)

print(vec4)
mat2 = np.zeros([5,3])

print(mat2)

mat3 = np.ones((3,5))

print('')

print(mat3)

mat4 = np.eye(5)

print('')

print(mat4)
# we can +-*/ arrays together if they're the right size

vec5 = np.arange(1,6)

vec6 = np.arange(3,8)

print(vec5)

print(vec6)

print(vec5+vec6)

print(vec5*vec6)

print(1/vec5)

print(np.sqrt(vec6))
# we can do matrix multiplication

print(mat)

print('')

print(vec)

print()

product = np.matmul(mat,vec)

print(product)
print(np.linalg.solve(mat,product))

print('')

print(np.linalg.inv(mat))
# we can find the unique values in an array

vec7 = np.array(['blue','red','orange','purple','purple','orange','Red',6])

print(vec7)

print(np.unique(vec7))
# we can also use numpy to generate samples of a random variable

rand_mat = np.random.rand(5,5) # uniform random variable

print(rand_mat)

rand_mat2 = np.random.randn(10,5) # standard normal random variable

print('')

print(rand_mat2)
# we can also use numpy for statistical tools on arrays

print(np.mean(rand_mat))

print(np.std(rand_mat2))
print(np.min(rand_mat))

print(np.max(rand_mat2))
# break here for next video!
# how do we access entries in a numpy vector

rand_vec = np.random.randn(19)

print(rand_vec)

print(rand_vec[6])
# we can access multiple entries at once using :

print(rand_vec[4:9])
# we can also access multiple non-consecutive entries using np.arange

print(np.arange(0,15,3))

print(rand_vec[np.arange(0,15,3)])
# what about matrices

print(rand_mat)

print(rand_mat[1][2])

print(rand_mat[1,2])

print(rand_mat[0:2,1:3])
# let's change some values in an array!

print(rand_vec)

rand_vec[3:5] = 4

print('')

print(rand_vec)

rand_vec[3:5] = [1,2]

print('')

print(rand_vec)
print(rand_mat)

rand_mat[1:3,3:5] = 0

print('')

print(rand_mat)
sub_mat = rand_mat[0:2,0:3]

print(sub_mat)

sub_mat[:] = 3

print(sub_mat)

print(rand_mat)
sub_mat2 = rand_mat[0:2,0:3].copy()

sub_mat2[:] = 99

print(sub_mat2)

print(rand_mat)

# break here for next video
# we can also access entries with logicals

rand_vec = np.random.randn(15)



print(rand_vec)

print(rand_vec>0)

print(rand_vec[rand_vec>0])
print(rand_mat2)

print(rand_mat2[rand_mat2>0])


print(rand_vec)

print('')

rand_vec[rand_vec>0.5] = -5

print(rand_vec)
# let's save some arrays on the disk for use later!

np.save('saved_file_name',rand_mat2)

np.savez('zipped_file_name',rand_mat=rand_mat,rand_mat2=rand_mat2)
# now let's load it

loaded_vec = np.load('saved_file_name.npy')

loaded_zip = np.load('zipped_file_name.npz')



print(loaded_vec)

print('')

print(loaded_zip)
print(loaded_zip['rand_mat'])

print('')

print(loaded_zip['rand_mat2'])



new_array  = loaded_zip['rand_mat']

print(new_array)
# we can also save/load as text files...but only single variables

np.savetxt('text_file_name.txt',rand_mat,delimiter=',')

rand_mat_txt = np.loadtxt('text_file_name.txt',delimiter=',')

print(rand_mat)

print('')

print(rand_mat_txt)