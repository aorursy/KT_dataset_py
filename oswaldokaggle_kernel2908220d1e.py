# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





import sympy
alpha, alpha_star, gamma, gamma_star, iden, q = sympy.symbols('alpha alpha_star gamma gamma_star iden q')

# TODO: Find out how to define q as a variable living in a subset

eq1 = sympy.Eq(alpha_star*alpha + gamma_star*gamma, iden)

eq2 = sympy.Eq(alpha*alpha_star + q**2*gamma*gamma_star, iden)

eq3 = sympy.Eq(gamma*gamma_star,gamma_star*gamma)

eq4 = sympy.Eq(q*gamma*alpha,alpha*gamma)

eq5 = sympy.Eq(q*gamma_star*alpha,alpha*gamma_star)

eq6 = sympy.Eq(q*alpha_star*gamma_star,gamma_star*alpha_star)

eq7 = sympy.Eq(q*alpha_star*gamma,gamma*alpha_star)



x, y= sympy.symbols('x y')

A = sympy.tensor.array.Array([[1,2],[3,4]])

B = sympy.tensor.array.Array([[x,y]])

sympy.tensor.array.tensorproduct(A,B)
tensorproduct = sympy.tensor.array.tensorproduct

m = 1

k = 1

l = 0

sympy.simplify(

    (tensorproduct(alpha,alpha)-q*tensorproduct(gamma_star,gamma))**m*(tensorproduct(gamma,alpha)+tensorproduct(alpha_star,gamma))**k*(tensorproduct(gamma_star,alpha_star)+tensorproduct(alpha,gamma_star))**l

)
sympy.expand(

    (tensorproduct(alpha,alpha)-q*tensorproduct(gamma_star,gamma))**m

    *(tensorproduct(gamma,alpha)+tensorproduct(alpha_star,gamma))**k

    *(tensorproduct(gamma_star,alpha_star)+tensorproduct(alpha,gamma_star))**l

)