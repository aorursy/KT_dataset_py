import sys

print(sys.version)
from IPython.display import YouTubeVideo



YouTubeVideo('Q81RR3yKn30', width=800, height=300)

"""

Generate a 10x10 Hilbert matrix

"""



from scipy.linalg import hilbert



x = hilbert(10)

x
#Type your code here-
"""

Multiply Hilbert transposed matrix with the original one

"""



import numpy as np



np.matrix(x).T * np.matrix(x)
#Type your code here-

"""

Calculate the correlation coefficient of the Hilbert matrix 

"""



import pandas as pd



pd.DataFrame(x, columns=['x%d'%i for i in range(1,11)]).corr()
"""

Define linear distribution function and the actual parameters

"""



from scipy.optimize import leastsq

import numpy as np

from scipy.linalg import hilbert



x = hilbert(10) # Generate 10x10 Hilbert Matrix

np.random.seed(10) # Random number seed will guarantee the same random number generated every time

w = np.random.randint(2,10,10) # Generate w randomly

y_temp = np.matrix(x) * np.matrix(w).T # Calculate y

y = np.array(y_temp.T)[0] #Convert y to 1D row vector



print("Actual Parameters w: ", w)

print("Actual Values y: ", y)
"""

Least squares linear fitting

"""



func=lambda p,x: np.dot(x, p) # Function

err_func = lambda p, x, y: func(p, x)-y # Loss function 

p_init=np.random.randint(1,2,10) # Init all parameters as 1



parameters = leastsq(err_func, p_init, args=(x, y)) # LS method

print("Fitted Parameters w: ",parameters[0])
YouTubeVideo('NGf0voTMlcs', width=800, height=300)
"""

Ridge regression fitting

"""



from sklearn.linear_model import Ridge



ridge_model = Ridge(fit_intercept=False) # Without intercept

ridge_model.fit(x, y)
ridge_model.coef_ # Print parameters
"""

Different alpha to get different fitting results

"""



alphas = np.linspace(-3,2,20)



coefs = []

for a in alphas:

    ridge = Ridge(alpha=a, fit_intercept=False)

    ridge.fit(x, y)

    coefs.append(ridge.coef_)
"""

Plot the results with different alphas

"""



from matplotlib import pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

%matplotlib inline



plt.plot(alphas, coefs) # Plot w-alpha

plt.scatter(np.linspace(0,0,10), parameters[0]) # Add w of OLS in the figure

plt.xlabel('alpha')

plt.ylabel('w')

plt.title('Ridge Regression')
"""

LASSO regression fitting and plot

"""



from sklearn.linear_model import Lasso



alphas = np.linspace(-2,2,10)

lasso_coefs = []



for a in alphas:

    lasso = Lasso(alpha=a, fit_intercept=False)

    lasso.fit(x, y)

    lasso_coefs.append(lasso.coef_)

    

plt.plot(alphas, lasso_coefs) # plot w-alpha

plt.scatter(np.linspace(0,0,10), parameters[0]) # Add w of OLS in the figure

plt.xlabel('alpha')

plt.ylabel('w')

plt.title('Lasso Regression')