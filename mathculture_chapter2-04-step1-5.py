def vector_sum(x, y):

    assert len(x) == len(y)

    return [_x + _y for _x, _y in zip(x, y)]
x = [1, 2, 3]

y = [8, 1, 2]

answer = vector_sum(x, y)

print(answer)  # => [9, 3, 5]
def matrix_sum(X, Y):

    assert len(X) == len(Y)

    return [vector_sum(x_v, y_v) for x_v, y_v in zip(X, Y) ]



X = [[1, 2, 3],

     [4, 5, 6]]

Y = [[8, 1, 2],

     [-1, 0, -2]]

answer = matrix_sum(X, Y)

print(answer)  # => [[9, 3, 5], [3, 5, 4]]
def dot(x_v, y_v):

    assert len(x_v) == len(y_v)

    return sum([x * y for x, y in zip(x_v, y_v)])
def matrix_vector_product(X, y):

    return [dot(x_v, y) for x_v in X]
X = [[1, 2, 3],

     [4, 5, 6]]

y = [8, 1, 2]

answer = matrix_vector_product(X, y)

print(answer)  # => [16, 49]
def trans(Y):

    return [[Y[i][j] for i in range(len(Y))] for j in range(len(Y[0]))]
def matrix_product(X, Y):

    Yt = trans(Y)

    return [matrix_vector_product(X, y) for y in Yt]



X = [[1, 2, 3],

     [4, 5, 6]]

Y = [[8, 1],

     [-1, 0],

     [0, 1]]

answer = trans(matrix_product(X, Y))

print(answer)  # => [[6, 4], [27, 10]]
def matrix_transpose(X):

    return trans(X)



X = [[1, 2, 3],

     [4, 5, 6]]

answer = matrix_transpose(X)

print(answer)  # => [[1, 4], [2, 5], [3, 6]]
