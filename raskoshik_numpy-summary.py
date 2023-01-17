import numpy as np
# Univariate Array Creation 
lst = [1,2,6,10]
array = np.array(lst)
display(lst)

# Mulivariate Array Creation
nested_lst = [[10,23,1],[20,11,6]]
array_nested = np.array(nested_lst)
display(array_nested)

# Transformation into array
data = ['1','2','3']
array = np.asarray(data, dtype = np.float32 )
display(array)

# Array with only ones creation
ones = np.ones((3,3))
display(ones)

# Array with only zeroes creation
zeros = np.zeros((3,3))
display(zeros)

# Empty Array Creation
empty = np.empty((3,3))
display(empty)

# Emty_like
array = np.arange(5)
empty = np.empty_like(array)
display(empty)

# Zeroes_like
array = np.ones((2,2))
zeroes = np.zeros_like(array)
display(zeroes)

# Identity
identity = np.identity(3)
display(identity)

# Eye
eye = np.eye(3,3)
display(eye)
# Initial 3D array
arr = np.random.randint(0,100,(3,4,3))

display(arr.ndim,
        arr.shape,
        arr.ravel(),
        arr.reshape(3,-1,2))
array = np.arange(10)
print('Original array:'+str(array))

# Slicing Demonstration
display(array[5:])
display(array[5:8])

# Important note!
array_slice = array[1:4]
print('Original slice :'+str(array_slice))
array_slice[1:] = 1000
print('Modified slice: '+str(array_slice))
# Slice is not copy. All changes in the sice will affect original array
print('Original array:'+str(array))
# 2D Array
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Simple Selection
print(arr[2])
print(arr[1,1])
print(arr[1][1])
print('\n')

# 2D Array Slicing 
print(arr[1:,1:])
print('\n')
print(arr[:,1:])
# 3D Array. It has dimension 2x2x3
arr = np.array( [ [[1,2,3],[4,5,6]],
                [[7,8,9],[10,11,12]] ])
print(arr)
print(arr.shape)
print(arr[0][0][1])
print(arr[1][0][1])
names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
data = np.random.randn(names.shape[0],4)

print(names == 'Bob')
print('\n')
condition = names == 'Bob'
# Now, according to above condition we will select rows with True condition 
selection = data[condition]
print(selection)
print('\n')
new_sel = selection[0,:] 
new_sel = 0
print(new_sel)
print('\n')
print('Original Slice: '+str(selection))
# Set values below zero to zero 
data[data<0]= 0
data
arr = np.arange(32).reshape(8,4)
arr
# Rows Selection 

print(arr[[0,2,4,6]])
print('\n')

print(arr[ [1,5,7,2],[0,3,1,2] ]) 
print('\n')

print(arr[[1,5,7,2]][:,[0,3,1,2]])
print('\n')

print(arr[np.ix_([1,5,7,2],[0,3,1,2])])
arr = np.arange(12).reshape(2,6)
print(arr)
print('\n')
print(arr.T)
matrx_1 = np.arange(15).reshape(3,5)
matrx_2 = np.random.randn(5,3)
display(matrx_1)
display(matrx_2)
arr = np.arange(16).reshape((2,2,4))
arr
# Arrays Creation 
arr_1 = np.arange(8).reshape(2,4)
arr_2 = np.arange(30,38).reshape(2,4)

# Concatenate along rows (axis = 0)
con_row_arr = np.concatenate([arr_1, arr_2])

# Concatenate along columns (axis = 1)
con_col_arr = np.concatenate([arr_1, arr_2],axis=1)

# vstack
v_stacked_arr = np.vstack([arr_1, arr_2])

# hstack 
h_stacked_arr = np.hstack([arr_1, arr_2])

# dstack
d_stacked_arr = np.dstack([arr_1, arr_2])

display(con_row_arr,con_col_arr,
        v_stacked_arr,h_stacked_arr,d_stacked_arr)
# vsplit
upper, lower = np.vsplit(v_stacked_arr,2)
print(f'Upper array is:\n {upper}\n Lower array is:\n {lower}')
print('\n')

# hsplit
left, right = np.hsplit(v_stacked_arr,2)
print(f'Upper array is:\n {left}\n Lower array is:\n {right}')
print('\n')

# Can split not only by a number but by necessary sequence
arr = np.arange(10).reshape(2,5)
left, centred, right = np.hsplit(arr,[2,4])
print(f'Left array is:\n {left}\n Centred array is:\n{centred}\n Right array is:\n{right}')
print('\n')

# dsplit 
inner, outer = np.dsplit(d_stacked_arr,2)
print(f'Inner array is:\n {inner}\n Outer array is:\n {outer}')
# Create a row vector from an array using reshape
arr = np.array([1,2,3])
row_vec = arr.reshape(1,3)
display(arr.shape,row_vec.shape)

# Create a row vector from an array using new axis
row_vec = arr[np.newaxis,:]
display(row_vec.shape)

# Create a column vector from an array 
col_vec = arr.reshape(3,1)
display(col_vec.shape)

# Create a column vector from an array  using new axis
col_vec = arr[:,np.newaxis]
display(col_vec.shape)
# Reduce
arr = np.arange(1,11)
print(np.multiply.reduce(arr))

# accumulate
print(np.add.accumulate(arr))

# sum, prod, cumsum, cumprod
display(np.sum(arr),
        np.prod(arr),
        np.cumsum(arr),
        np.cumproduct(arr))
arr_1 = np.arange(4)
arr_2 = np.arange(4)[:,np.newaxis]
brdcs_arr = arr_1 + arr_2
print(f'First array shape: {arr_1.shape}\nSecond array shape:{arr_2.shape}'+'\n'+
      f'Broadcasting result:\n{brdcs_arr}\nFinal Shape:{brdcs_arr.shape}')
arr = np.random.randint(0,101,10)

indx_1 = [5,4,0,1] # the first shape
indx_2 = np.random.randint(0,6,(3,2)) # the second shape

display(arr[indx_1],arr[indx_2])
arr = np.random.randint(0,101,(3,4))
# The first shape
rows_1 = [0,2,1,2,0]
cols_1 = [1,3,2,1,3]
# The second shape
rows_2 = np.random.randint(0,3,(3,3))
cols_2 = np.random.randint(0,4,(3,3))
display(arr[rows_1,cols_1],arr[rows_2,cols_2])
indx = 2
arr_indx = [0,2]
display(arr,arr[2,arr_indx])
x = np.random.rand(100)

bins = np.linspace(-5,5,20)
counts = np.zeros_like(bins)

i = np.searchsorted(bins,x)
i

