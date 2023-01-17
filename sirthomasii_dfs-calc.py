import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sympy import symbols, Eq, solve

import math



hbar = 6.582119569e-16

a, b = symbols('a b')





m1_0 = 5/2

m2_0 = -5/2

f_0 =  0



m1_1 = 5/2

m2_1 = -3/2

f_1 =  300e6



m1_2 = -3/2

m2_2 = 1/2

f_2 =  733e6



I_n = 5/2

J = 62



E_1 = (f_0-f_1)

E_2 = (f_0-f_2)



F1_0 = m1_0 + J

F2_0 = m2_0 + J



C1_0 = F1_0*(F1_0+1)-I_n*(I_n+1)-J*(J+1)

C2_0 = F2_0*(F2_0+1)-I_n*(I_n+1)-J*(J+1)



F1_1 = m1_1 + J

F2_1 = m2_1 + J



C1_1 = F1_1*(F1_1+1)-I_n*(I_n+1)-J*(J+1)

C2_1 = F2_1*(F2_1+1)-I_n*(I_n+1)-J*(J+1)



F1_2 = m1_2 + J

F2_2 = m2_2 + J



C1_2 = F1_2*(F1_2+1)-I_n*(I_n+1)-J*(J+1)

C2_2 = F2_2*(F2_2+1)-I_n*(I_n+1)-J*(J+1)





eq1 = Eq(((((a/2)*C1_0)+(b*(((3/4)*C1_0*(C1_0+1)-I_n*(I_n+1)*J*(J+1)))/(2*I_n*(2*I_n-1)*J*(2*J-1)))) \

         + (((a/2)*C2_0)+(b*(((3/4)*C2_0*(C2_0+1)-I_n*(I_n+1)*J*(J+1)))/(2*I_n*(2*I_n-1)*J*(2*J-1))))) \

         + \

         ((((a/2)*C1_1)+(b*(((3/4)*C1_1*(C1_1+1)-I_n*(I_n+1)*J*(J+1)))/(2*I_n*(2*I_n-1)*J*(2*J-1))))\

         +(((a/2)*C2_1)+(b*(((3/4)*C2_1*(C2_1+1)-I_n*(I_n+1)*J*(J+1)))/(2*I_n*(2*I_n-1)*J*(2*J-1))))) \

         - E_1 )





eq2 = Eq(((((a/2)*C1_0)+b*(((3/4)*C1_0*(C1_0+1)-I_n*(I_n+1)*J*(J+1))/(2*I_n*(2*I_n-1)*J*(2*J-1)))) \

         + (((a/2)*C2_0)+b*(((3/4)*C2_0*(C2_0+1)-I_n*(I_n+1)*J*(J+1))/(2*I_n*(2*I_n-1)*J*(2*J-1))))) \

         + \

         ((((a/2)*C1_2)+b*(((3/4)*C1_2*(C1_2+1)-I_n*(I_n+1)*J*(J+1))/(2*I_n*(2*I_n-1)*J*(2*J-1))))\

         +(((a/2)*C2_2)+b*(((3/4)*C2_2*(C2_2+1)-I_n*(I_n+1)*J*(J+1))/(2*I_n*(2*I_n-1)*J*(2*J-1))))) \

         - E_2 )





sol_dict = solve((eq1,eq2), (a, b))

print("a coefficient: ", '{:.2e}'.format(sol_dict[a]))

print("b coefficient: ", '{:.2e}'.format(sol_dict[b]))
import math

hbar = 6.582119569e-16



#-------ground state

f_0 = 0

f_1 = 190e6



#-------

f_2 = 1815e6

f_3 = 2015e6



delta_E0 = abs(f_0-f_1)

delta_E1 = abs(f_0-f_2)





print("Energy_split F2-F1: ", '{:.2e}'.format(delta_E0), "")

print("Energy_split F2-F1: ", '{:.2e}'.format(delta_E1), "")