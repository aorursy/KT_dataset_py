# Import torch and other required modules
import torch
import numpy as np
# Example 1 - Define a 2 by 2 matrix, either from a list or a np array
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor(np.array([[5., 6.],[7., 8.]]))
print(x,"\n",y)
# Example 2 - Tensors can be accessed using regular Python indexing and slicing mechanisms
print(x[0][0],x[0][1],x[1][0],x[1][1])
# Example 3 - As mentioned before, torch.tensor is an allias of the default tensor type "torch.FloatTensor", but the elements inside are adapted to the input type
print(x.dtype)
print(y.dtype)
# Example 4 - Matrixes need to have acceptable shapes (this means, each column  need to have the same length, each row need to have the same length), otherwise it breaks
torch.tensor([[1, 2], [3, 4, 5]])
# Example 1 - Defining a matrix with zeros and default data type
my_tensor = torch.tensor(())  # This creates an tensor of size 0
my_tensor.new_zeros((2,2))
# Example 2 - Creating a matrix with zeros of different data type (int8)
my_tensor_int8 = torch.tensor((),dtype=torch.int8)
my_tensor_int8= my_tensor_int8.new_zeros((2,2))
print(f"new_zeros inherits the data type from the original tensor. Therefore, my_tensor is data type {my_tensor.dtype} and my_tensor_int8 is {my_tensor_int8.dtype}")
# Example 3 - Creating a zeros matrix without input, this is the fastest way
torch.zeros(3, 3)
""" Example 4 - In order to use new_zeros, we need to have an PyTorch object, otherwise, we get an exception
Note that lists and other regular Python data types do not include the new_zeros attribute (aka function)
"""
a = (1,2,3,4)
a.new_zeros((2,3))
# Example 1 - Use fill_diagonal and fill with value 1
my_diagonal_matrix = torch.zeros(10, 10)
my_diagonal_matrix.fill_diagonal_(1)
# Example 2 - The matrix does not need to be squared, for example it also works with a matrix 2x5, but then of course, the diagonal looks weird
torch.zeros(2,5).fill_diagonal_(1)

# Example 3 - Interestingly, if we try to fill the diagonal with a data type which do not match to the original data type of the matrix, this is not breaking
my_tensor_int8.fill_diagonal_(2.234)  # 2.234 is a float, but the resulting matrix keeps the original torch.int8 data types and fills the diagonal with the rounded int8
# Example 1 - The function torch.random.get_rng_state() generates a random number
another_tensor = torch.zeros(1, 2, 3)
print("Originally, the dimensions were ",another_tensor.size())

print("But after permuting, the dimensions are ", another_tensor.permute(2, 0, 1).size())
# Example 2 - permute will easily break if the input is of a mismatching size to the dimensions
another_tensor.permute(2, 0, 1, 3)

# Example 3 - permute does NOT allow to repeat dimensions, this is preventing typos probably
another_tensor.permute(1, 1, 1)
# Example 1 - Lets take the tensor x and expand it to see what happens
print(f"The tensor x is of the size {x.size()} and contents: \n {x}")
print(f"After expanding with 2 new columns we get a matrix of the size {x.expand(2,2,2).size()} and contents: \n {x.expand(2,2,2)}")
# Example 2 - We can also expand a simple vector (one-dimensional array), so that we can easier see what happens
my_vector = torch.tensor([[6.5],[3.26],[8.9]])
print(my_vector)
print(my_vector.expand(3,6))
# Example 3 - We can break the code if the expanded size of the tensor does not match the existing size in one of the dimensions
print(f"We cannot expand a 2x2 matrix with additional 2x2 {x.expand(2,4)}")
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()
