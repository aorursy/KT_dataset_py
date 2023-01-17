# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
z1 = 4+3j # Forma binómica

z1
z2 = complex(1,7) # Usando la función complex

z2
type(z1)

type(z2)
z1.real # Parte real
z2.imag # Parte imaginaria
z1.conjugate()
z2.conjugate()
import cmath

abs(z1) # El valor absoluto de un número complejo no existe, estamos haciendo el módulo
cmath.phase(z2) # Obtenemos el argumento principal de Z2
z1 + z2 # Suma de números complejos
5 * z2 # Producto por un escalar
z1 * z2 # Producto de números complejos
z1=2+3j

z2=1+1j



z1*z2
z1 = 1+1j

z2 = 1-1j



z1*z2
from sympy import *

init_printing(use_latex='mathjax')

x = symbols('x')

y = symbols('y')
expand((x**2 + x + 1) * (x-1))
expand((x + 1)**2)
expand((x+1)*(x-1))
# Matrices

import numpy as np # linear algebra



# Exercise 1

A = np.array([[0,1,-2],[2,3,1],[1,-1,5]])

B = np.array([[1,-2,2,1],[2,-2,2,-2],[-1,2,1,2]])

C = np.array([[2],[0],[1],[-4]])



#print(A.dot(B)) error, can't operate because dimension

print(B.dot(C))

print()

print(B.transpose())

print()

print(B.transpose().dot(A))

print()

print(C.transpose().dot(B.transpose()))
import numpy as np # linear algebra



# Exercise 2

A = np.array([[0,1],[0,1]])

B = np.array([[-1,-1],[0,0]])



A_plus_B_square = np.linalg.matrix_power((A+B),2)

A_square = np.linalg.matrix_power(A,2)

B_square = np.linalg.matrix_power(B,2)

AB_twice = 2*(A+B)



print(A_plus_B_square == A_square + AB_twice + B_square) # it's false

print()



A_plus_B_cube = np.linalg.matrix_power((A+B),3)

A_cube = np.linalg.matrix_power(A,3)

B_cube = np.linalg.matrix_power(B,3)



print (A_plus_B_cube == A_cube + 3*A_square*B + 3*A*B_square + B_cube) # it's true

# Exercise 1 -> np.linalg.solve(A, b) way

import numpy as np

A = np.array([[10,2,-1,1,0,10],[-1,-3,0,0,-1,5],[0,-1,3,-1,0,0],[17,1,0,3,5,-15],[0,-10,0,-5,3,0],[-3,1,1,1,-2,2]])

b = np.array([0,5,5,4,-21,11])

AB = np.array([[10,2,-1,1,0,10,0],[-1,-3,0,0,-1,5,5],[0,-1,3,-1,0,0,5],[17,1,0,3,5,-15,4],[0,-10,0,-5,3,0,-21],[-3,1,1,1,-2,2,11]])
np.linalg.matrix_rank(A) == np.linalg.matrix_rank(AB) == np.linalg.matrix_rank(A) == 6
x = np.linalg.solve(A, b)

x
# Exercise 1 -> linsolve way



from sympy import *

from sympy.solvers.solveset import linsolve

x1,x2,x3,x4,x5,x6 = symbols('x1,x2,x3,x4,x5,x6')



linsolve([10*x1 + 2*x2 -1*x3 + 1*x4 + 0*x5 + 10*x6 + 0, 

          -1*x1 - 3*x2 + 0*x3 + 0*x4 - 1*x5 + 5*x6 - 5, 

          0*x1 - 1*x2 + 3*x3 - 1*x4 + 0*x5 + 0*x6 - 5, 

          17*x1 + 1*x2 + 0*x3 + 3*x4 + 5*x5 - 15*x6 - 4, 

          0*x1 - 10*x2 + 0*x3 - 5*x4 + 3*x5 + 0*x6 + 21, 

          -3*x1 + 1*x2 + 1*x3 + 1*x4 - 2*x5 + 2*x6 - 11], 

         (x1,x2,x3,x4,x5,x6))
# Exercise 2 -> linsolve way



from sympy import *

from sympy.solvers.solveset import linsolve

x1,x2,x3,x4,x5,x6,x7 = symbols('x1,x2,x3,x4,x5,x6,x7')



linsolve([-2*x1 + 2*x2 -1*x3 + 1*x4 + 0*x5 + 4*x6 + 0*x7 -5,

          -1*x1 - 3*x2 + 0*x3 + 0*x4 - 1*x5 + 5*x6 - 2*x7 - 5,

          0*x1 - 1*x2 + 3*x3 - 1*x4 + 0*x5 + 0*x6 + 0*x7 - 5,

          0*x1 + 1*x2 + 0*x3 + 3*x4 - 2*x5 + 1*x6 + 4*x7 - 0,

          0*x1 - 10*x2 + 0*x3 - 5*x4 + 3*x5 + 0*x6 + 0*x7 + 21,

          -3*x1 + 1*x2 + 1*x3 + 1*x4 - 2*x5 + 2*x6 + 0*x7 - 11], 

         (x1,x2,x3,x4,x5,x6,x7))