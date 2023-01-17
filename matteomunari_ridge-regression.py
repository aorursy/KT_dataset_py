import numpy as np



def noisy_f(x):

    return x**2 - 16*x + np.random.randint(10)

    

X = np.linspace(0,10,num=100).reshape(-1,1)

y = np.array(list(map(noisy_f,X)))
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge

from sklearn.pipeline import make_pipeline



poly1 = make_pipeline(PolynomialFeatures(2),Ridge(alpha=1))

poly1.fit(X,y)



prediction1 = poly1.predict(X)



poly2 = make_pipeline(PolynomialFeatures(2),Ridge(alpha=25))

poly2.fit(X,y)



prediction2 = poly2.predict(X)



poly3 = make_pipeline(PolynomialFeatures(2),Ridge(alpha=100))

poly3.fit(X,y)



prediction3 = poly3.predict(X)
import matplotlib.pyplot as plt



plt.plot(X,y,'.r',label='training data')

plt.plot(X,prediction1,'b',label='alpha=1')

plt.plot(X,prediction2,'g',label='alpha=25')

plt.plot(X,prediction3,'y',label='alpha=100')

plt.legend()