import numpy as np
import pandas as pd
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
from math import exp as e
# 팩토리얼 함수
def factorial( n ):
    if n == 0:
        return 1
    fact = 1
    for i in range( 1, n+1 ):  
        fact *= i  # fact = 1 * 2 * 3 * ... * n-1 * n
    return fact
 
# 테일러 급수 함수    
def taylor_mthd(f, a, b, N, v_ini):
    
    h = (b-a)/float(N)                  # h = step-size
    t[0], w[0] = v_ini                  # t&w's initial values
    
    for i in range(1,N+1):              # apply Euler's method
        T = 0
        for j in range(len(f)):
            h_factor = h**(j)/float(factorial(j+1))  # j는 order
            T += h_factor * f[j](t[i-1], w[i-1])     # T(4)면 j는 3. f'''
        w[i] = w[i-1] + h * T
    return w
# initial value
a, b = 0.0, 3.0
N = 15
h = (b-a)/N
IV = (0.0, 1.0)
t = np.arange(a, b+h, h)   # t value
w = np.zeros((N+1,))       # w value


# differential equation
f   = lambda t, y: y - t**2.0 + 2
df  = lambda t, y: y - t**2.0 + 2 - 2*t
ddf = lambda t, y: y - t**2.0 - 2*t
dddf = lambda t, y: y - t**2.0 - 2*t

# numeritic value
w2 = taylor_mthd( [ f, df ], a, b, N, IV )
w3 = taylor_mthd( [ f, df, ddf ], a, b, N, IV )
w4 = taylor_mthd( [ f, df, ddf, dddf ], a, b, N, IV )
# analytic value(real value)
y = []

for i in t:
    w = e(i)*(e(-i)*i**2 + 2*e(-i)*i + 1)
    y.append(w)

y = np.array(y)
# 시각화: 그래프 그리기

plt.plot(t, w2, c='yellow', label = '2nd order')
plt.plot(t, w3, c='orange', label = '3rd order')
plt.plot(t, w4, c='pink', label = '4th order')
plt.plot(t, y,  c='green', label = 'exact')

plt.title("Taylor's Method Example, N="+str(N))
plt.xlabel('t') 
plt.ylabel('y(t)')
plt.legend(loc='upper left')
plt.show()
error = y - w4
error
plt.plot(t, error,  c='red', label = 'error')
plt.legend(loc='upper left')
plt.show()