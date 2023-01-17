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
           color='green',  title='Weight distribution');
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
    else:
        return 2
    
data['weight_cat'] = data['Weight'].apply(weight_category)
data.head()
sns.boxplot(x = 'weight_cat', y = 'Height', data= data)
data.plot(x = 'Weight', y = 'Height', kind= 'scatter', title= 'Dependencies between height and weight')
def sq_error(w0, w1, x, y):
    n = len(x)
    error = 0
    for i in range(n):
        error += (y[i] - (w0 + w1 * x[i]))**2
    return error
# y = w_0 + w_1 * x
x = np.linspace(70, 170)
y1 = 60 + 0.05 * x
y2 = 50 + 0.16 * x
data.plot(x = 'Weight', y = 'Height', kind= 'scatter', title= 'Dependencies between height and weight')
plt.plot(x, y1, color = 'green')
plt.plot(x , y2, color = 'red')
w0 = 50
x = np.array(data['Height'])
y = np.array(data['Weight'])
w1 = np.linspace(-0.5, 2.5)
plt.plot(w1, sq_error(w0, w1, x, y))
plt.title('Error function dependecy of w1 when w0 = 50')
plt.xlabel('w1')
plt.ylabel('error function value')
plt.show()
from scipy.optimize import minimize_scalar
w0 = 50
x = np.array(data['Height'])
y = np.array(data['Weight'])
minimize_scalar(lambda w1: sq_error(w0,w1,x,y),bounds=(-5,5))
# w1_opt = 1.1351597092091679
x = np.linspace(70, 170)
y3 = 50 + 1.1 * x
data.plot(x = 'Weight', y = 'Height', kind= 'scatter', title= 'Dependencies between height and weight')
plt.plot(x, y3)
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

# Finally use *plot_surface* method of type object
# Axes3DSubplot. Add titles to axes.
surf = ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d') # get current axis

x = np.array(data['Height'])
y = np.array(data['Weight'])

W0 = np.arange(-5, 5, 0.25)
W1 = np.arange(-5, 5, 0.25)
W0, W1 = np.meshgrid(W0, W1)
Z = sq_error(W0, W1, x, y)

# Finally use *plot_surface* method of type object
# Axes3DSubplot. Add titles to axes.
surf = ax.plot_surface(W0, W1, Z)
ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Error')
plt.show()
#from scipy.optimize import minimize
#x = np.array(data['Height'])
#y = np.array(data['Weight'])
#minimize(sq_error, [0,0], args=(2,), bounds=[[-100,100], [-5,5]], method = 'L-BFGS-B')