#import these
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import seaborn as sns
import pandas as pd 
import numpy as np
from sklearn import linear_model

#NoW suppose we have a dataset
x=[1,2,3,4,5]
y=[5,7,9,11,13]


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
img = mpimg.imread('../input/linear-regression-analysis/cost_function.png')
imgplot = plt.imshow(img)
#Let's plot it
plt.scatter(x,y,marker='o',color='r',label='skitscat')
plt.xlabel('X')
plt.ylabel('Y')
#plt.grid(False,color='k')
plt.title('Scatter')
plt.legend()
plt.show()
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
img = mpimg.imread('../input/linear-regression-analysis/learningrate.png')
imgplot = plt.imshow(img)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
img = mpimg.imread('../input/linear-regression-analysis/MSEvsb.png')
imgplot = plt.imshow(img)
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
%matplotlib inline 
img = mpimg.imread('../input/linear-regression-analysis/learningrate.png') 
imgplot = plt.imshow(img)
#now let's create a gradient descent function
n=len(x)
x=np.array(x)
y=np.array(y)
learningrate=0.08
def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=10000
    
    for i in range(iterations):
        y_predic=m_curr*x + b_curr
        cost=(1/n)*sum([val**2 for val in (y-y_predic)])
        md=-(2/n)*sum(x*(y-y_predic))
        bd=-(2/n)*sum(y-y_predic)
        m_curr=m_curr-learningrate*md
        b_curr=b_curr-learningrate*bd
        print(cost,m_curr,b_curr)
gradient_descent(x,y)
        
    
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
img = mpimg.imread('../input/linear-regression-analysis/Learningr.png')
imgplot = plt.imshow(img)
reg=linear_model.LinearRegression()

df=pd.DataFrame(columns=['x','y'])
df['x']=x
df['y']=y
df.head()


df[['x']]
df['y']
reg.fit(df[['x']],df.y)
reg.predict([[2.5]])

print(reg.coef_,reg.intercept_)
homeprices=pd.read_csv('../input/linear-regression-analysis/homeprices.csv')
homeprices
df=homeprices
df.fillna(3.0,inplace=True)
df
#You need to fill the nan values with exploratory data analytics
#WE are using 3.0 here

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
reg.predict([[3000,2000,3.0]])