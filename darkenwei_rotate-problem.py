# Tuple method
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

#rotate 90 deg clockwise = rotate around diagonal axis then reverse the whole matrix 

rotated = [list(reversed(col)) for col in zip(*matrix)]   # zip(*matrix) gets a collection of (1,4,7),(2,5,8) and (3,6,9), then we just reverse each tuple.

for row in rotated:
    print(row)
#     print(*row)


#Numpy method

import numpy as np

matrix = [[1, 2, 3], [4,5,6,], [7,8,9]]
rotated = np.rot90(matrix, k=1, axes=(1,0)).tolist()

for row in rotated:
    print(row)
