from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
boston = datasets.load_boston()
boston.keys()
df = pd.DataFrame(data = boston.data)
df.columns = boston.feature_names
df.head()
df['Target'] = boston.target
df = df.rename(columns = {'Target':'Price'})
corr = df.corr()
print(boston.DESCR)
corr['Price'].sort_values(ascending = False)
corr_values = corr['Price'].abs().sort_values(ascending = False)
corr_values
from sklearn import preprocessing

x_RM = preprocessing.scale(df['RM'])
x_LSTAT = preprocessing.scale(df['LSTAT'])
y = preprocessing.scale(df['Price'])
from pylab import rcParams
rcParams['figure.figsize'] = 12,8
plt.scatter(y, x_RM, s=5, label = 'RM')
plt.scatter(y, x_LSTAT, s=5, label = 'LSTAT')
plt.legend(fontsize=15)
plt.xlabel('Average number of rooms & Low status population', fontsize=15)
plt.ylabel('Price', fontsize=15)
plt.legend()
plt.show()
x = np.c_[np.ones(x_RM.shape[0]),x_RM, x_LSTAT]
# Parameters required for Gradient Descent
alpha = 0.0001   #learning rate
m = y.size  #no. of samples
np.random.seed(10)
theta = np.random.rand(3)  #initializing theta with some random values
def gradient_descent(x, y, m, theta, alpha):
    cost_list = []   #to record all cost values to this list
    theta_list = []  #to record all theta_0 and theta_1 values to this list 
    prediction_list = []
    run = True
    cost_list.append(1e10)    #we append some large value to the cost list
    i=0
    while run:
        prediction = np.dot(x, theta)   #predicted y values theta_0*x0+theta_1*x1
        prediction_list.append(prediction)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)   #  (1/2m)*sum[(error)^2]
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   # alpha * (1/m) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:   #checking if the change in cost function is less than 10^(-9)
            run = False

        i+=1
    cost_list.pop(0)   # Remove the large number we added in the begining 
    return prediction_list, cost_list, theta_list
prediction_list, cost_list, theta_list = gradient_descent(x, y, m, theta, alpha)
theta = theta_list[-1]
plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost_list)
plt.show()
yp = theta[0] +theta[1]*x[:,1] + theta[2]*x[:,2]
MSE_equ = ((yp-y)**2).mean()  #Using yp from equation of hyperplane
MSE_GD = ((prediction_list[-1]-y)**2).mean()  #From Gradient Descent


print('Mean Square Error using equation of hyperplane : {}'.format(round(MSE_equ,3)))
print('Mean Square Error from Gradient Descent prediction : {}'.format(round(MSE_GD,3)))

from sklearn.linear_model import LinearRegression
ys = df['Price']
xs = np.c_[df['RM'],df['LSTAT']]
ys.shape, xs.shape
xs = preprocessing.scale(xs)
ys = preprocessing.scale(ys)
lm = LinearRegression()

#Fitting the model
lm = lm.fit(xs,ys)
pred = lm.predict(xs)
pred.shape
intercept = lm.intercept_
Theta_0 = lm.coef_[0]
Theta_1 = lm.coef_[1]

print('Intercept : {}'.format(round(intercept,3)))
print('Theta_0 : {}'.format(round(Theta_0,4)))
print('Theta_1 : {}'.format(round(Theta_1,4)))
print('Intercept : {}'.format(round(theta[0],3)))
print('Theta_0 : {}'.format(round(theta[1],4)))
print('Theta_1 : {}'.format(round(theta[2],4)))
r2_sk = lm.score(xs,ys)
print('R square from sci-kit learn: {}'.format(round(r2_sk,4)))
r2 = 1 - (sum((y - prediction_list[-1])**2)) / (sum((y - y.mean())**2))
print('R square doing from the scratch: {}'.format(round(r2,4)))
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D



# Create a figure and a 3D Axes
fig = plt.figure(figsize=(12,10))
ax = Axes3D(fig)
ax.set_xlabel('Rooms', fontsize = 15)
ax.set_ylabel('Population', fontsize = 15)
ax.set_zlabel('Price', fontsize = 15)

plt.close()
def init():
    ax.scatter(xs[:,0], xs[:,1], ys, c='C6', marker='o', alpha=0.6) 
    x0, x1 = np.meshgrid(xs[:,0], xs[:,1])
    yp = Theta_0 * x0 + Theta_1 * x1
    ax.plot_wireframe(x0,x1,yp, rcount=200,ccount=200, linewidth = 0.5,color='C9', alpha=0.5)
    ax.legend(fontsize=15, labels = ['Data points', 'Hyperplane'])
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,


# Animate

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)

# plt.legend(fontsize=15, labels = [''])
anim.save('animation.gif', writer='imagemagick', fps = 30)
plt.close()
#Display the animation...
import io
import base64
from IPython.display import HTML

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
# Function for getting the 2D view

def plot_view(elev_given, azim_given):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Scatter plot
    ax.scatter(xs[:,0], xs[:,1], ys, c='C6', marker='o', alpha=0.6, label='Data points')

    #Plane 

    x0, x1 = np.meshgrid(xs[:,0], xs[:,1])
    yp = Theta_0 * x0 + Theta_1 * x1
    ax.plot_wireframe(x0,x1,yp, rcount=200,ccount=200, linewidth = 0.5, color='C9', alpha=0.5, label='Hyperplane')

    ax.set_xlabel('Rooms', fontsize = 15)
    ax.set_ylabel('Population', fontsize = 15)
    ax.set_zlabel('Price', fontsize = 15)
    plt.legend(fontsize=15)
    ax.view_init(elev=elev_given, azim=azim_given)
    
    

plot_view(-23,91)
plt.show()
plot_view(158,-172)
plt.show()
x_ZN = df['ZN']
xs = np.c_[df['RM'],df['ZN']]
xs = preprocessing.scale(xs)
lm = lm.fit(xs,ys)
lm.score(xs,ys)
xsingle = preprocessing.scale(df['RM'])
xsingle = xsingle.reshape(-1,1)
lm = lm.fit(xsingle,ys)
lm.score(xsingle,ys)
print('R square from sci-kit learn using single feature: {}'.format(round(lm.score(xsingle,ys),4)))
# Adjusted R square : 
1 - (1-r2_sk)*(df.shape[0]-1)/(df.shape[0]-2-1)