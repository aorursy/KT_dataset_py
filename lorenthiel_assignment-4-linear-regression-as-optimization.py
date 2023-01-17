import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/weights_heights.csv', index_col='Index')
data.plot(y='Height', kind='hist', 
           color='red',  title='Height (inch.) distribution');
data.head()
data.plot(y='Weight', kind='hist', 
           color='green',  title='Weight (pounds) distribution')
def make_bmi(height_inch, weight_pound):
    METER_TO_INCH, KILO_TO_POUND = 39.37, 2.20462
    return (weight_pound / KILO_TO_POUND) / \
           (height_inch / METER_TO_INCH) ** 2
data['BMI'] = data.apply(lambda row: make_bmi(row['Height'], 
                                              row['Weight']), axis=1)
sns.pairplot(data)
def weight_category(weight):
    if weight < 120:
        return 1
    elif weight >= 150:
        return 3
    return 2

data['weight_cat'] = data['Weight'].apply(weight_category)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.boxplot(data=data, y="Height", x="weight_cat")
data.plot(kind='scatter', title='Height vs Weight', x='Height', y = 'Weight')
def mse(data, w0, w1):
    y = data.Height
    x = data.Weight
    return np.array((y - (w0 + w1 * x)) ** 2).sum()
mse(data, 1, 1)
plt.plot(data.Weight, data.Height, 'bo')
x = np.linspace(75,175, 2)
plt.plot(x, x*0.05+60, color='green' )
plt.plot(x, x*0.16+50, color='red' )
x = np.linspace(0,1,101)
error = list(map(lambda w: mse(data, 50, w), x))
#print(np.array(error))

print(mse(data, 50, 0.14))
plt.plot(x, error)
import scipy 

def f(x):
    return mse(data, 50, x)

opt = scipy.optimize.minimize_scalar(f, bounds=(-5, 5), method='bounded')
opt.x
plt.plot(data.Weight, data.Height, 'bo')
x = np.linspace(75,175, 2)
plt.plot(x, x*opt.x+50, color='green' )
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d') # get current axis

# Create NumPy arrays with data points on X and Y axes.
# Use meshgrid method creating matrix of coordinates
# By vectors of coordinates. Set needed function Z(x, y).
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
print(Z)
# Finally use *plot_surface* method of type object
# Axes3DSubplot. Add titles to axes.
surf = ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d') # get current axis

x = np.linspace(0,1,101)
y = np.linspace(0,100,101)

z = []
for a in x:
    tmp = []
    for b in y:
        tmp.append(mse(data,b,a))
    z.append(tmp)

X, Y = np.meshgrid(x, y)
Z = np.array(z)
surf = ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
#pairs = [[w0, w1] for w0 in y for w1 in x]  
#error = list(map(lambda w: (w[0], w[1], mse(data, w[0], w[1])), pairs))
#error
# Your code here
# Your code here

import scipy 

def f(params):
    y,x = params
    return mse(data, y, x)

bnds = ((-100, 100), (-5, 5))
opt = scipy.optimize.minimize(f, (0,0), bounds = bnds, method='L-BFGS-B')
opt.x
plt.plot(data.Weight, data.Height, 'bo')
x = np.linspace(75,175, 2)
plt.plot(x, x*0.08200637+57.57179162, color='green' )