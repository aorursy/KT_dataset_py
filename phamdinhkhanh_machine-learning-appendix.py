import numpy as np



x = np.array([1,2,3])

y = np.array([3,4,5])

np.dot(x, y.T)
z = x * y

print(z)
# Khởi tạo ma trận A kích thước 2 x 3

A = np.array([[2, 1, 1],

              [1, 2, 1]])



# Khởi tạo ma trận B kích thước 3 x 2

B = np.array([[1, 2],

              [2, 2],

              [1, 0]])



# Tích dot product của 2 ma trận A x B là một ma trận kích thước 2 x 2

np.dot(A, B)
# Khởi tạo ma trận A và B có cùng kích thước 2 x 3

A = np.array([[2, 1, 1],

              [1, 2, 1]])



B = np.array([[1, 2, 2],

              [2, 2, 1]])



# Tích element-wise hoặc hadamard của 2 ma trận A, B là một ma trận cùng kích thước 2 x 3

A * B
# Khởi tạo ma trận đơn vị kích thước 3 x 3

np.identity(3)
# Tính ma trận nghịch đảo của ma trận A trong python

A = np.array([[1, 2],

              [1, 3]])

np.linalg.pinv(A)
# Tạo ma trận chuyển vị của ma trận A kích thước 2 x 3

A = np.array([[1, 2, 3],

              [2, 3, 4]])



A.T