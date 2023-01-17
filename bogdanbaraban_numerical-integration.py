import math

import numpy as np

def f1(x):

    return x**3+4*x**2+16

def cos2x(x):

    return np.cos(2*x)
class NumericalIntegration:

    

    def loss(self,a,b,m,n,param,h=None):

        """

        Arguments:

        a,b -- bonds of a segment

        m -- max(abs(f(n)(x))), x[a;b], f(n) - n-th derivative over x

        n -- number of segments

        param -- specifies loss

        h -- interval between consecutive subsegments

        """

        

        if not h:

            h = (b-a)/n

        if param == "riemann":

            return h**2*m*(b-a)/24

        elif param == "trapezoidal":

            return h**2 * m *(b-a)/12

        else: #simpson

            return h**4*m*(b-a)/2880

    

    def riemann(self,a,b,f,n,m=None):

        """

        Arguments:

         a,b -- bounds of a segment

         f -- input function

         n -- number of segments

        Returns:

         res -- integral value on a certain segment

        """

        h = (b - a)/n

        x = np.linspace(a,b,n+1)

        x_mid = (x[:-1] + x[1:])/2 # searching for middle point 



        if m is not None:

            return np.sum(f(x_mid)*h), self.loss(a,b,m,n,"riemann",h)

        return np.sum(f(x_mid)*h)

    

    

    def trapezoidal(self,a,b,f,n,m=None):

        """

        Arguments:

        ----------

        a,b -- bounds of a segment

        f -- function

        n -- number of segments, has to be even

        m -- max(f''(x)) on [a;b]

        Returns:

        res -- integral value on a segment

        loss -- error rate

        """



        h = (b-a)/n    

        X = [a+j*h for j in range(n+1)]

        res = sum(f(X[i])+f(X[i+1]) for i in range(n))*0.5*h

        if m is not None:

            return res, self.loss(a,b,m,n,"trapezoidal",h)

        return res  

    

    

    def simpson(self,a,b,f,n,m=None): 

        """

        Arguments:

        ----------

        a,b -- bounds of a segment

        f -- function

        Returns:

        --------

        integral value at on a certain segment

        """

        h = (b-a)/n

        res = h/6 * sum(f(a+h*i) + 4*f(a+h*(i+0.5)) + f(a+h*(i+1)) for i in range(n))  

        if m:

            return res, self.loss(a,b,m,n,"simpson",h)

        return res
n = NumericalIntegration()

print(n.riemann(0,math.pi/4,cos2x,1,4))

print(n.trapezoidal(0,math.pi/4,cos2x,10,4))

print(n.simpson(0,math.pi/4,cos2x,1,16))