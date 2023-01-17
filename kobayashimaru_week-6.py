from IPython.display import display, Math, Latex
display(Math(r'We\, have\,  initial\, condition: '))

display(Math(r'T\,  = \,1'))

display(Math(r'\xi_{1}\,  = \, 0.05'))

display(Math(r'S(0)\, = \, 50'))

display(Math(r'And\, we\, have\, to\, find\, Asian\, Option\, Price\, for\, various\, m,\, \xi_{2}, K'))
display(Math(r'S(t)\, = \, S(0)\, exp(\,(\xi_{1}\, - \, \dfrac{1}{2}\xi_{2}^{2})t\, +\, \xi_{2}W(t)\,), \, where\, W(t)\, is\, Wiener\, Process'))
display(Math(r'C_{T}\, = \, (\dfrac{1}{T} \int_{0}^{T}S(t)dt \, - \, K)^{+}, where\, x^{+} = x\, \, if\, x > 0'))
display(Math(r'discounted\, payoff\, =\, exp(-\xi_{1}T)\, C_{T}(T)'))
display(Math(r'E(\, exp(-\xi_{1}T)\, C_{T}(T) \,)'))
import numpy as np
#==============Asian Option Price==================



def discounted_payoff(S0, xi1, xi2, T, m, K, rep):

    

    def Wiener_proccess(m, T):

        W = np.zeros(1)

        i = 0

        for i in range(m - 1):

            W = np.append(W, W[len(W) - 1] + (np.sqrt(T / m)) * 

                np.random.normal(loc = 0, scale = 1, size = 1))

        return(W)

    

    i = 0

    Ct_array = []

    

    for i in range(rep):

        t = np.linspace(0, T, num = m + 1)

        t = t[1:]

        w = Wiener_proccess(m, T)

        St = S0 * np.exp((xi1 - 0.5 * xi2**2 ) * t + xi2 * w)

        

        Ct = np.mean(St) / T - K

        if Ct <= 0:

            Ct = 0

            

        Ct_array.append(Ct)

    

    Ct_array = np.array(Ct_array)

            

    discounted_payoff_array = np.exp(-xi1 * T) * Ct_array

    

    expected_discounted_payoff = np.mean(discounted_payoff_array)

    return(expected_discounted_payoff)

#==============Asian Option Price==================
#==============Asian Option Price==================

T = 1

xi1 = 0.05

S0 = 50



m = np.array([16, 64])

xi2 = np.array([0.1, 0.3])

K = np.array([45, 50, 55])



for i in range(len(m)):

    for j in range(len(xi2)):

        for k in range(len(K)):

            print("The expected discounted payoff when ", 

                  "T = 1, xi1 = 0.05, S0 = 50, "

                  "m = ", m[i], ",", 

                  "xi2 = ", xi2[j], ",", 

                  "K = ", K[k], 

                  " is ", discounted_payoff(S0, xi1, xi2[i], T, m[j], K[k], rep = 10000))

            

#==============Asian Option Price==================
display(Math(r'f_{1}(x_{1}, x_{2}, x_{3})\, = \, 3x_{1} - cos(x_{2}x_{3}) - \dfrac{1}{2}\, =\, 0'))

display(Math(r'f_{2}(x_{1}, x_{2}, x_{3})\, = \, x_{1}^{2} - 81(x_{2} + 0.1)^{2} + sin(x_{3}) + 1.06 =\, 0'))

display(Math(r'f_{3}(x_{1}, x_{2}, x_{3})\, = \, exp(-x_{1}x_{2}) + 20x_{3} + \dfrac{1}{3}(10 \pi - 3) =\, 0'))
display(Math(r'\nabla{f_{1}}^T = (3, \,\, x_{3}sin(x_{2}x_{3}), \,\,  x_{2}sin(x_{2}x{3}))'))

display(Math(r'\nabla{f_{2}}^T = (2x_{1}, \,\, -162(x_{2} + 0.1), \,\,  cos(x_{3}))'))

display(Math(r'\nabla{f_{3}}^T = (-x_{2}exp(-x_{1}x_{2}), \,\, -x_{1}exp(-x_{1}x_{2}), \,\,  20)'))
display(Math(r'x_{n+1} \, = \, x_{n} - J(x_{n})^{-1}f(x_{n})\, \, ,'))

display(Math(r'where\, J(x_{n}) = \begin{bmatrix} \nabla{f_{1}}^T\\ \nabla{f_{2}}^T \\ \nabla{f_{3}}^T \end{bmatrix},\,  f(x) = \begin{bmatrix} f_{1}(x_{1}, x_{2}, x_{3}) \\ f_{2}(x_{1}, x_{2}, x_{3}) \\ f_{3}(x_{1}, x_{2}, x_{3}) \end{bmatrix}'))
from numpy.linalg import inv


#==============Newton's Method==================



def three_dim_Newton_method(f1, f2, f3, grad_f1, grad_f2, grad_f3, 

                            x0, tol):

    

    def f(x): return(np.matrix([f1(x), f2(x), f3(x)]))

    

    X_array = [x0]

    

    

    #=========I make x1 from x0 in order to keep the loop going======

    i = 1 

    J = np.matrix([grad_f1(X_array[i - 1]), 

                   grad_f2(X_array[i - 1]),

                   grad_f3(X_array[i - 1])])

    inv_J = inv(J)

    x_update = np.transpose(np.asmatrix(X_array[i - 1])) - np.matmul(inv_J, np.transpose(f(X_array[i - 1])))

    x_update = np.asarray(np.transpose(x_update))

    X_array.append(x_update[0])

    i += 1

    #=========I make x1 from x0 in order to keep the loop going======



    def dist(x, y): 

        return(np.sqrt(sum((x - y) ** 2)))

    

    i = 1

    

    while(dist(X_array[i - 1], X_array[i]) >= tol):

      J = np.matrix([grad_f1(X_array[i]), 

                     grad_f2(X_array[i]),

                     grad_f3(X_array[i])])

      inv_J = inv(J)

      x_update = np.transpose(np.asmatrix(X_array[i])) - np.matmul(inv_J, np.transpose(f(X_array[i])))

      x_update = np.asarray(np.transpose(x_update))

      X_array.append(x_update[0])

      i += 1

    

    return(X_array)



#==============Newton's Method==================



  

def f1(x): return(3*x[0] - np.cos(x[1] * x[2]) - 0.5)

def f2(x): return(x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + np.sin(x[2]) + 1.06)

def f3(x): return(np.exp(-x[0] * x[1]) + 20*x[2] + (10 * np.pi - 3) / 3)



def grad_f1(x): return(np.array([3, x[2] * np.sin(x[1] * x[2]), x[1] * np.sin(x[1] * x[2])]))

def grad_f2(x): return(np.array([2 * x[1], -162 * (x[1] + 0.1), np.cos(x[2])]))

def grad_f3(x): return(np.array([-x[1] * np.exp(-x[0] * x[1]), -x[0] * np.exp(-x[0] * x[1]), 20]))

#============================x0 = (0.1, 0.1, -0.1)==============================



sol_array_1 = three_dim_Newton_method(f1, f2, f3, grad_f1, grad_f2, grad_f3,

                                      x0 = np.array([0.1, 0.1, -0.1]), tol = 10 ** (-6))    

print("The solution of the system is ", sol_array_1[len(sol_array_1) - 1])



#============================x0 = (0.1, 0.1, -0.1)===============================
#============================x0 = (1, 1, -1)==============================



sol_array_2 = three_dim_Newton_method(f1, f2, f3, grad_f1, grad_f2, grad_f3,

                                      x0 = np.array([1, 1, -1]), tol = 10 ** (-6))    



print("The solution of the system is ", sol_array_2[len(sol_array_2) - 1])



#============================x0 = (1, 1, -1)==============================