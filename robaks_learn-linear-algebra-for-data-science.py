import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
def plotVectors(vecs, cols, alpha=1):
    """
    Plot set of vectors.

    Parameters
    ----------
    vecs : array-like
        Coordinates of the vectors to plot. Each vectors is in an array. For
        instance: [[1, 3], [2, 2]] can be used to plot 2 vectors.
    cols : array-like
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    alpha : float
        Opacity of vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure of the vectors
    """
    plt.figure()
    plt.axvline(x=0, color='#A9A9A9', zorder=0)
    plt.axhline(y=0, color='#A9A9A9', zorder=0)

    for i in range(len(vecs)):
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]],
                   [x[1]],
                   [x[2]],
                   [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=cols[i],
                  alpha=alpha)

orange = '#FF9A13'
blue = '#1190FF'
plotVectors([[1,3], [2,1]], [orange, blue])
plt.xlim(0,5)
plt.ylim(0,5)
# We will start by creating a vector. This is just a 1-dimensional array:
x = np.array([1,2,3,4])
x
# The array() function can also create 2-dimensional arrays with nested brackets:
A = np.array([[1,2], [3,4], [5,6]])
A
"The shape of an array (that is to say its dimensions) for A:{}, x{}".format(A.shape, x.shape)
"The number corresponds to the length of the x:{}".format(len(x))
# transpose matrix A
A_t = A.T
A_t
"The shpe of A:{}, A_t:{}".format(A.shape, A_t.shape)
B = np.array([[4,6], [2,8], [9,0]])
B
# Add matrices A and B
C = A + B
C
# Add 2 to the matrix A
A+2
#Numpy can handle operations on arrays of different shapes.
A = np.array([[1,2], [3,4], [5,6]])
B = np.array([[4], [9], [2]])
C = A + B
C
# Numpy function dot() can be used to compute the matrix product
A = np.array([[1,2], [3,4], [5,6]])
B = np.array([[2], [4]])
C = np.dot(A,B)
C
# It is equivalent to use the method dot() of Numpy arrays
C = A.dot(B)
C
# Multiplication of two matrices
A = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
B = np.array([[2,7], [1,2], [3,6]])
C = A.dot(B)
C
# A(B+C)=AB+AC
A = np.array([[2,3], [1,4], [7,6]])
B = np.array([[5], [2]])
C = np.array([[4], [3]])
np.array_equal(A.dot(B+C), A.dot(B) + A.dot(C))
# AB≠BA
A = np.array([[2,3], [6,5]])
B = np.array([[5,3], [2,2]])
np.array_equal(A.dot(B), B.dot(A))
# However vector multiplication is commutative
x = np.array([[2], [6]])
y = np.array([[5], [2]])
np.array_equal(x.T.dot(y), y.T.dot(x))
# Simplification of the matrix product
A = np.array([[2,3], [1,4], [7,6]])
B = np.array([[5,3], [2,2]])
np.array_equal(A.dot(B).T, np.dot(B.T,A.T))
np.eye(3)
# When ‘apply’ the identity matrix to a vector the result is this same vector: I*x = x
x = np.array([[2], [6], [3]])
np.eye(x.shape[0]).dot(x)
A = np.array([[3,0,2], [2,0,-2], [0,1,1]])

# note that non square matrices (matrices with more columns than rows or more rows than columns)
# don’t have inverse.
A_inv = np.linalg.inv(A)
A_inv
A_bis = A_inv.dot(A)
A_bis
A = np.array([[2,-1], [1,1]])
A_inv = np.linalg.inv(A)
b = np.array([[0], [3]])
A_inv.dot(b)
# Let’s plot them to check privious solution
x = np.arange(-10,10)
y = 2*x
y1 = -x + 3

import pylab as plt
plt.figure()
plt.plot(x,y)
plt.plot(x,y1)
plt.xlim(-1,4)
plt.ylim(-1,4)
# draw axes
plt.axvline(x=0,color='grey')
plt.axhline(y=0,color='grey')
plt.legend(('y=2x', 'y=-x+3'), loc='upper right')
plt.show()
plt.close()
# m=1, n=2: 1 equation and 2 variables
# The graphical interpretation of n=2 is that we have a 2-D space.
# So we can represent it with 2 axes. 
# As m=1, we have only one equation.
# This means that we have only one line characterizing our linear system.
x = np.arange(-10,10)
y = 2*x + 1

import pylab as plt
plt.figure()
plt.plot(x,y)
plt.xlim(-2,10)
plt.ylim(-2,10)
# draw axes
plt.axvline(x=0,color='grey')
plt.axhline(y=0,color='grey')
plt.legend(('y=2x+1'), loc='upper right')
plt.show()
plt.close()
# m=2, n=2: 2 equations and 2 unknowns
# The graphical interpretation of n=2 is that we have a 2-D space.
# So we can represent it with 2 axes. 
# As m=2, we have two equation.
# This means that we have two lines characterizing our linear system.
x = np.arange(-10,10)
y = 2*x + 1
y1 = 6*x - 2

plt.figure()
plt.plot(x,y)
plt.plot(x,y1)
plt.xlim(-2,10)
plt.ylim(-2,10)
# draw axes
plt.axvline(x=0,color='grey')
plt.axhline(y=0,color='grey')
plt.legend(('y=2x+1', 'y=6x-2'), loc='upper right')
plt.show()
plt.close()

# m=3, n=2: 3 equations and 2 unknowns
# The graphical interpretation of n=2 is that we have a 2-D space.
# So we can represent it with 2 axes. 
# As m=3, we have three equation.
# This means that we have three lines characterizing our linear system.
# no solution because there is no point in space that is on each of these lines.
x = np.arange(-10,10)
y = 2*x + 1
y1 = 6*x - 2
y2 = 1/10*x + 6

plt.figure()
plt.plot(x,y)
plt.plot(x,y1)
plt.plot(x,y2)
plt.xlim(-2,10)
plt.ylim(-2,10)
# draw axes
plt.axvline(x=0,color='grey')
plt.axhline(y=0,color='grey')
plt.legend(('y=2x+1', 'y=6x-2', 'y=1/10*x+6'), loc='upper right')
plt.show()
plt.close()
# vectors
# Weigths of the vectors
a = 2
b = 1
# Start and end coordinates of the vectors
u = [0,0,1,3]
v = [2,6,2,1]

orange = '#FF9A13'
blue = '#1190FF'
plt.quiver([u[0], a*u[0], b*v[0]],
           [u[1], a*u[1], b*v[1]],
           [u[2], a*u[2], b*v[2]],
           [u[3], a*u[3], b*v[3]],
           angles='xy', scale_units='xy', scale=1, color=[orange, orange, blue])
plt.xlim(-1,8)
plt.ylim(-1,8)
# draw axes
plt.axvline(x=0,color='grey')
plt.axhline(y=0,color='grey')
plt.scatter(4,7,marker='x',s=50)
# Draw the name of the vectors
plt.text(-0.5,2,r'$\vec{u}$', color=orange, size=18)
plt.text(0.5,4.5,r'$\vec{u}$', color=orange, size=18)
plt.text(2.5,7,r'$\vec{v}$', color=blue, size=18)
plt.show()
plt.close()
# m=2, n=2: 2 equations and 2 unknowns
# The graphical interpretation of n=2 is that we have a 2-D space.
# So we can represent it with 2 axes. 
# As m=2, we have two equation.
# This means that we have two lines characterizing our linear system.
x = np.arange(-10,10)
y = 0.5*x + 1
y1 = -x + 4

plt.figure()
plt.plot(x,y)
plt.plot(x,y1)
plt.xlim(-2,10)
plt.ylim(-2,10)
# draw axes
plt.axvline(x=0,color='grey')
plt.axhline(y=0,color='grey')
plt.show()
plt.close()
# To talk in term of the column figure we can reach the point of coordinates (−1,4)
# if we add two times the vector u⃗  and two times the vector v⃗.
u = [0,0,0.5,1]
u_bis = [u[2],u[3],u[2],u[3]]
v = [2*u[2],2*u[3],-1,1]
v_bis = [2*u[2]-1,2*u[3]+1,v[2],v[3]]

orange = '#FF9A13'
blue = '#1190FF'
plt.quiver([u[0], u_bis[0], v[0], v_bis[0]],
          [u[1], u_bis[1], v[1], v_bis[1]],
          [u[2], u_bis[2], v[2], v_bis[2]],
          [u[3], u_bis[3], v[3], v_bis[3]],
          angles='xy', scale_units='xy', scale=1, color=[blue, blue, orange, orange])
# plt.rc('text', usetex=True)
plt.xlim(-1.5, 2)
plt.ylim(-0.5, 4.5)
# draw axes
plt.axvline(x=0, color='#A9A9A9')
plt.axhline(y=0, color='#A9A9A9')
plt.scatter(-1,4,marker='x',s=50)
plt.text(0, 0.5, r'$\vec{u}$', color=blue, size=18)
plt.text(0.5, 1.5, r'$\vec{u}$', color=blue, size=18)
plt.text(0.5, 2.7, r'$\vec{v}$', color=orange, size=18)
plt.text(-0.8, 3, r'$\vec{v}$', color=orange, size=18)
plt.show()
plt.close()
# The triangle inequity
# The norm of the sum of some vectors is less than or equal to the sum of the norms of these vectors.
u = np.array([1, 6])
v = np.array([4, 2])

# ||u+v|| <= ||u|| + ||v||
"{} <= {} + {}".format(np.linalg.norm(u+v), np.linalg.norm(u), np.linalg.norm(v))
# The Euclidean norm (L2 norm)
# In this case, the vector is in a 2-dimensional space but this stands also for more dimensions.
np.linalg.norm([3, 4])
# The squared Euclidean norm is widely used in machine learning partly
# because it can be calculated with the vector operation (transpose matrix x)*x.
# There can be performance gain due to the optimization
x = np.array([[2], [5], [3], [3]])
# (transpose matrix x)*x == ||x||2
"{} == {}".format(x.T.dot(x), np.linalg.norm(x)**2)
# Matrix norms
A = np.array([[1,2], [6,4], [3,2]])
np.linalg.norm(A)
# Diagonal matrices
# A matrix A(i,j) is diagonal if its entries are all zeros except on the diagonal (when i=j).
# The diagonal matrix can be denoted diag(v) where v is the vector containing the diagonal values.
# The Numpy function diag() can be used to create square diagonal matrices
v = np.array([2,4,3,1])
np.diag(v)
# The mutliplication between a diagonal matrix and a vector is thus
# just a ponderation of each element of the vector by v
v = np.array([2,4,3,1])
x = np.array([3,2,2,7])
D = np.diag(v)
D.dot(x)
# Non square matrices have the same properties
v = np.array([2,4,3])
x = np.array([3,2,2])
D = np.diag(v)
np.vstack([D, [0,0,0]])
D.dot(x)
# The invert of a square diagonal matrix exists if all entries of the diagonal are non-zeros.
# If it is the case, the invert is easy to find.
# Also, the inverse doen’t exist if the matrix is non-square.
v = np.array([2,4,3,1])
D = np.diag(v)
D_inv = np.linalg.inv(D)
D_inv
# Symmetric matrices
# The matrix A is symmetric if it is equal to its transpose
# This concerns only square matrices.
A = np.array([[2, 4, -1], [4, -8, 0], [-1, 0, 3]])
A.T
# Unit vectors
# A unit vector is a vector of length equal to 1. It can be denoted by a letter with a hat: ^
# Orthogonal vectors
# Two orthogonal vectors are separated by a 90° angle.
# The dot product of two orthogonal vectors gives 0.
x = [0,0,2,2]
y = [0,0,2,-2]

plt.quiver([x[0], y[0]],
          [x[1], y[1]],
          [x[2], y[2]],
          [x[3], y[3]],
          angles='xy', scale_units='xy', scale=1)
plt.xlim(-2,4)
plt.ylim(-3,3)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.text(1, 1.5, r'$\vec{u}$', size=18)
plt.text(1.5, -1, r'$\vec{v}$', size=18)

plt.show()
plt.close()

# x = [2,2], y = [2,-2], xT * y = 0
A = np.array([[-1, 3], [2, -2]])
v = np.array([[2], [1]])
Av = A.dot(v)
print(Av)

plotVectors([v.flatten(), Av.flatten()], cols=['#1190FF', '#FF9A13'])
plt.ylim(-1, 4)
plt.xlim(-1, 4)
# v is a eigenvector of A if v and Av are in the same direction (the vectors Av and v are parallel).
# The output vector is just a scaled version of the input vector.
# This scalling factor is λ which is called the eigenvalue of A.
# eigenvector [[1], [1]], eigenvalue λ = 6
A = np.array([[5,1], [3,3]])
v = np.array([[1], [1]])
Av = A.dot(v)

orange = '#FF9A13'
blue = '#1190FF'

plotVectors([Av.flatten(), v.flatten()], cols=[blue, orange])
plt.ylim(-1, 7)
plt.xlim(-1, 7)
# another eigenvector [[1], [-3]], eigenvalue λ = 2
A = np.array([[5,1], [3,3]])
v = np.array([[1], [-3]])
Av = A.dot(v)

orange = '#FF9A13'
blue = '#1190FF'

plotVectors([Av.flatten(), v.flatten()], cols=[blue, orange])
plt.ylim(-8, 2)
plt.xlim(-1, 4)
# Find eigenvalues and eigenvectors
A = np.array([[5, 1], [3, 3]])

# eigenvectors and eigenvalues
# the first array corresponds to the eigenvalues and the second to the eigenvectors concatenated in columns
np.linalg.eig(A)
# Concatenating eigenvalues and eigenvectors
V = np.array([[1, 1], [1, -3]])
V_inv = np.linalg.inv(V)
lambdas = np.diag([6,2])
V.dot(lambdas).dot(V_inv)
# symmetric matrices
A = np.array([[6, 2], [2, 3]])
eigVals, eigVecs = np.linalg.eig(A)
eigVals = np.diag(eigVals)
eigVecs.dot(eigVals).dot(eigVecs.T)
# You can see a matrix as a specific linear transformation.
# When you apply this matrix to a vector or to another matrix you will apply this linear transformation to it.
# To represent the linear transformation associated with matrices
# we can also draw the unit circle and see how a matrix can transform it

x = np.linspace(-1, 1, 100000)
y = np.sqrt(1-(x**2))
# after matrix transformation
x1 = np.linspace(-3, 3, 100000)
y1 = 2*np.sqrt(1-((x1/3)**2))

plt.plot(x, y, sns.color_palette().as_hex()[0])
plt.plot(x, -y, sns.color_palette().as_hex()[0])
plt.plot(x1, y1, sns.color_palette().as_hex()[1])
plt.plot(x1, -y1, sns.color_palette().as_hex()[1])
plt.xlim(-4,4)
plt.ylim(-4,4)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.show()
plt.close()

# Note that these examples used diagonal matrices (all zeros except the diagonal).
# The general rule is that the transformation associated with diagonal matrices imply
# only a rescaling of each coordinate without rotation.
# This is a first element to understand the SVD.
# Matrices that are not diagonal can produce a rotation.
# Since it is easier to think about angles when we talk about rotation.
x = np.linspace(-3, 3, 100000)
y = 2*np.sqrt(1-((x/3)**2))

x1 = x*np.cos(np.radians(45)) - y*np.sin(np.radians(45))
y1 = x*np.sin(np.radians(45)) + y*np.cos(np.radians(45))

x1_neg = x*np.cos(np.radians(45)) - -y*np.sin(np.radians(45))
y1_neg = x*np.sin(np.radians(45)) + -y*np.cos(np.radians(45))

u1 = [-2*np.sin(np.radians(45)), 2*np.cos(np.radians(45))]
v1 = [3*np.cos(np.radians(45)), 3*np.sin(np.radians(45))]

plotVectors([u1, v1], cols=['#FF9A13', '#FF9A13'])

plt.plot(x, y, '#1190FF')
plt.plot(x, -y, '#1190FF')

plt.plot(x1, y1, '#FF9A13')
plt.plot(x1_neg, y1_neg, '#FF9A13')

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
def matrixToPlot(matrix, vectorsCol=['#FF9A13', '#1190FF']):
    """
    Modify the unit circle and basis vector by applying a matrix.
    Visualize the effect of the matrix in 2D.

    Parameters
    ----------
    matrix : array-like
        2D matrix to apply to the unit circle.
    vectorsCol : HEX color code
        Color of the basis vectors

    Returns:

    fig : instance of matplotlib.figure.Figure
        The figure containing modified unit circle and basis vectors.
    """
    # Unit circle
    x = np.linspace(-1, 1, 100000)
    y = np.sqrt(1-(x**2))

    # Modified unit circle (separate negative and positive parts)
    x1 = matrix[0,0]*x + matrix[0,1]*y
    y1 = matrix[1,0]*x + matrix[1,1]*y
    x1_neg = matrix[0,0]*x - matrix[0,1]*y
    y1_neg = matrix[1,0]*x - matrix[1,1]*y

    # Vectors
    u1 = [matrix[0,0],matrix[1,0]]
    v1 = [matrix[0,1],matrix[1,1]]

    plotVectors([u1, v1], cols=[vectorsCol[0], vectorsCol[1]])

    plt.plot(x1, y1, 'g', alpha=0.5)
    plt.plot(x1_neg, y1_neg, 'g', alpha=0.5)

A = np.array([[3,7],[5,2]])

print('Unit circle:')
matrixToPlot(np.array([[1,0],[0,1]]))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

print('Unit circle transformed by A:')
matrixToPlot(A)
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.show()
# compute the SVD of A:
U, D, V = np.linalg.svd(A)
print('U: {}'.format(U))
print('D: {}'.format(D))
print('V: {}'.format(V))
# We can now look at the sub-transformations by looking at the effect of the matrices
# U, D and V in the reverse order
print('Unit circle:')
matrixToPlot(np.array([[1,0],[0,1]]))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

print('First rotation:')
matrixToPlot(V)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.show()

print('Scaling:')
matrixToPlot(np.diag(D).dot(V))
plt.xlim(-9, 9)
plt.ylim(-9, 9)
plt.show()

print('Second rotation:')
matrixToPlot(U.dot(np.diag(D)).dot(V))
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.show()
A = np.array([[7,2], [3,4], [5,2]])
U, D, V = np.linalg.svd(A)

D_plus = np.zeros((A.shape[0], A.shape[1])).T
D_plus[:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))

A_plus = V.T.dot(D_plus).dot(U.T)
print("pseudoinverse(A+=VD+UT): {}".format(A_plus))
# check with the pinv() function from Numpy that the pseudoinverse is correct
print("pseudoinverse(pinv()): {}".format(np.linalg.pinv(A)))
# overdetermined system of equation
# no solution
x1 = np.linspace(-5, 5, 1000)
x2_1 = -2*x1 + 2
x2_2 = 4*x1 + 8
x2_3 = -x1 + 2

plt.plot(x1, x2_1)
plt.plot(x1, x2_2)
plt.plot(x1, x2_3)
plt.xlim(-2, 1)
plt.ylim(-1, 5)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

print('overdetermined system of equation. no solution')
plt.show()

# pseudoinverse
A = np.array([[-2, -1], [4, -1], [-1, -1]])
A_plus = np.linalg.pinv(A)
print('pseudoinverse: {}'.format(A_plus))

# we can find x (x = A_plus * B)
B = np.array([[-2], [-8], [-2]])
x = A_plus.dot(B)
print('solution: {}'.format(x))

# plot this point along with the equations lines
plt.plot(x1, x2_1)
plt.plot(x1, x2_2)
plt.plot(x1, x2_3)
plt.xlim(-2, 1)
plt.ylim(-1, 5)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')
plt.scatter(x[0], x[1])

print('solution point')
plt.show()
# This method can also be used to fit a line to a set of points.
A = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [3, 1], [4, 1]])
b = np.array([[2], [4], [0], [2], [5], [3]])
coefs = np.linalg.pinv(A).dot(b)
print('coefs(mx+b=y, m=coefs[0], b=coefs[1]): {}'.format(coefs))

x = np.linspace(-1,5,1000)
y = coefs[0]*x+coefs[1]

plt.plot(A[:,0], b, '*')
plt.plot(x,y)
plt.xlim(-1,6)
plt.ylim(-1,6)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

print('result')
plt.show()
# Fitting a line to a set of data points
np.random.seed(123)
x = 5*np.random.rand(100)
y = 2*x + 1 + np.random.rand(100)

x = x.reshape(100, 1)
y = y.reshape(100, 1)
A = np.hstack((x, np.ones(np.shape(x))))
print('A: {}'.format(A))
print('Shape A: {}'.format(np.shape(A)))
print('A_plus: {}'.format(np.linalg.pinv(A)))
print('Shape A_plus: {}'.format(np.shape(np.linalg.pinv(A))))
coefs = np.linalg.pinv(A).dot(y)

x_line = np.linspace(0,5,1000)
y_line = coefs[0]*x_line+coefs[1]

plt.plot(x,y, '*')
plt.plot(x_line, y_line)
plt.show()
# trace is the sum of all values in the diagonal of a square matrix
A = np.array([[2, 9, 8], [4, 7, 1], [8, 2, 5]])
print('Trace: {}'.format(np.trace(A)))

# transposition of a matrix doesn’t change the diagonal
print('norm A = sqrt( trace(A*A_tran) )')
print('{} = {}'.format(np.linalg.norm(A), np.sqrt(np.trace(A.dot(A.T)))))

A = np.array([[4, 12], [7, 6]])
B = np.array([[1, -3], [4, 3]])
C = np.array([[6, 6], [2, 5]])
print('Trace(ABC) = Trace(CAB) = Trace(BCA)')
print('{} = {} = {}'.format(np.trace(A.dot(B).dot(C)), np.trace(C.dot(A).dot(B)), np.trace(B.dot(C).dot(A))))
i = [0, 1]
j = [1, 0]

plotVectors([i, j], [['#1190FF'], ['#FF9A13']])
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.show()

print('Apply A on our two unit vectors i and j')
A = np.array([[2, 0], [0, 2]])
plotVectors([A.dot(i), A.dot(j)], [['#1190FF'], ['#FF9A13']])
plt.xlim(-0.5, 5)
plt.ylim(-0.5, 5)
plt.show()

print('determinant: {}'.format(np.linalg.det(A)))
# negative determinant
i = [0, 1]
j = [1, 0]

plotVectors([i, j], [['#1190FF'], ['#FF9A13']])
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.show()

print('Apply A on our two unit vectors i and j')
A = np.array([[-2, 0], [0, 2]])
plotVectors([A.dot(i), A.dot(j)], [['#1190FF'], ['#FF9A13']])
plt.xlim(-3, 1)
plt.ylim(-1, 3)
plt.show()

print('determinant: {}'.format(np.linalg.det(A)))
points = np.array([[1, 3], [2, 2], [3, 1], [4, 7], [5, 4]])
C = np.array([[-1, 0], [0, 1]])

newPoints = points.dot(C)

print('mirrored the initial shape')
plt.figure()
plt.plot(points[:, 0], points[:, 1])
plt.plot(newPoints[:, 0], newPoints[:, 1])
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')
plt.show()
np.random.seed(123)
x = 5*np.random.rand(100)
y = 2*x + 1 + np.random.rand(100)

x = x.reshape(100, 1)
y = y.reshape(100, 1)

print('dataset with correlated features')
X = np.hstack([x,y])
plt.plot(X[:,0], X[:,1], '*')
plt.show()

# function that substract the mean of each column to each data point of this column
def centerData(X):
    X = X.copy()
    X -= np.mean(X, axis = 0)
    return X

print('let’s center our data X around 0 for both dimensions')
X_centered = centerData(X)
plt.plot(X_centered[:,0], X_centered[:,1], '*')
plt.show()

eigVals, eigVecs = np.linalg.eig(X_centered.T.dot(X_centered))

orange = '#FF9A13'
blue = '#1190FF'
plotVectors(eigVecs.T, [orange, blue])
plt.plot(X_centered[:,0], X_centered[:,1], '*')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

X_new = eigVecs.T.dot(X_centered.T)
plt.plot(eigVecs.T.dot(X_centered.T)[0, :], eigVecs.T.dot(X_centered.T)[1, :], '*')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()