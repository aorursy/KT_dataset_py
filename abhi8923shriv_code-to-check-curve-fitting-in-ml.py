# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

# Scikit-Learn for fitting models

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



# For plotting 

import matplotlib

import matplotlib.pyplot as plt
# We define a curve, in this case a sine curve to serve as our process that generates the data. As the real-world is never perfectly clean however, we also need to add some noise into the observations. This is done by adding a small random number to each value.



#Set the random seed for reproducible results

np.random.seed(42)



#generating function representing a process in real life

def true_gen(x):

    y = np.sin(1.2 * x * np.pi) 

    return(y)



# x values and y value with a small amount of random noise

x = np.sort(np.random.rand(120))

y = true_gen(x) + 0.1 * np.random.randn(len(x))#basicallyfor each value of pure sin(x), we are adding a small random value 
# Training and Testing



# Random indices for creating training and testing sets

random_ind = np.random.choice(list(range(120)), size = 120, replace=False)

x_t = x[random_ind]#for the elements in the array x, random elements are chosen based on the index deterimined by random_ind

y_t = y[random_ind]



# Training and testing observations

train = x_t[:int(0.7 * len(x))] #we are now choosing 70% of data as training and the remaining below as test.

test = x_t[int(0.7 * len(x)):]



y_train = y_t[:int(0.7 * len(y))]

y_test = y_t[int(0.7 * len(y)):]



# Model the true curve. As you might have noticed, here too we are calling the true_gen method. But here the difference is..

#...the input is a continuous values of x (determined by np.linspace, we gives uniformly spaced values) for we are getting

#..the sin(x) value, and thus fit_poly(train, y_train, test, y_test, degrees = 1, plot='test')this time plotting the sine curve itself. (above we just created 'points' following sine curve)

x_linspace = np.linspace(0, 1, 1000)

y_true = true_gen(x_linspace)
# Visualization



# Visualize observations and true curve

plt.plot(train, y_train, 'ko', label = 'Train'); 

plt.plot(test, y_test, 'ro', label = 'Test')

plt.plot(x_linspace, y_true, 'b-', linewidth = 2, label = 'True function')

plt.legend()

plt.xlabel('x'); plt.ylabel('y'); plt.title('Data');
# Visualization



# Visualize observations and true curve

plt.plot(train, y_train, 'ko', label = 'Train'); 

plt.plot(test, y_test, 'ro', label = 'Test')

plt.plot(x_linspace, y_true, 'b-', linewidth = 4, label = 'True function')

plt.legend()

plt.xlabel('x'); plt.ylabel('y'); plt.title('Data');
def fit_poly(train, y_train, test, y_test, degrees, plot='train', return_scores=False):

    

    # Create a polynomial transformation of features.For example for degree 2, (x,x**2) i.e. 2,2**2 => (2,4)

    # so a point in 1-D is converted to a point in 2-D, and so on for even higher degree. 

    features = PolynomialFeatures(degree=degrees, include_bias=False)

    

    # Reshape training features for use in scikit-learn and transform features

    train = train.reshape((-1, 1))

    train_trans = features.fit_transform(train)

    

    # Create the linear regression model and train

    model = LinearRegression()

    model.fit(train_trans, y_train) 

    

    # Train set predictions and error

    train_predictions = model.predict(train_trans)

    training_error = mean_squared_error(y_train, train_predictions) # Format test features

    test = test.reshape((-1, 1))

    test_trans = features.fit_transform(test)

    

    # Test set predictions and error

    test_predictions = model.predict(test_trans)

    testing_error = mean_squared_error(y_test, test_predictions)

    

    # Find the model curve and the true curve

    x_curve = np.linspace(0, 1, 100)

    x_curve = x_curve.reshape((-1, 1))

    x_curve_trans = features.fit_transform(x_curve)

    

    # Model curve

    model_curve = model.predict(x_curve_trans)

    

    # True curve

    y_true_curve = true_gen(x_curve[:, 0])

    

    # Plot observations, true function, and model predicted function

    if plot == 'train':

        plt.plot(train[:, 0], y_train, 'ko', label = 'Observations')

        plt.plot(x_curve[:, 0], y_true_curve, linewidth = 4, label = 'True Function')

        plt.plot(x_curve[:, 0], model_curve, linewidth = 4, label = 'Model Function')

        plt.xlabel('x'); plt.ylabel('y')

        plt.legend()

        plt.ylim(-1, 1.5); plt.xlim(0, 1)

        plt.title('{} Degree Model on Training Data'.format(degrees))

        plt.show()

        

    elif plot == 'test':

        # Plot the test observations and test predictions

        plt.plot(test, y_test, 'o', label = 'Test Observations')

        plt.plot(x_curve[:, 0], y_true_curve, 'b-', linewidth = 2, label = 'True Function')

        plt.plot(test, test_predictions, 'ro', label = 'Test Predictions')

        plt.ylim(-1, 1.5); plt.xlim(0, 1)

        plt.legend(), plt.xlabel('x'), plt.ylabel('y'); plt.title('{} Degree Model on Testing Data'.format(degrees)), plt.show();

    

     # Return the metrics

    if return_scores:

        return training_error, testing_error

fit_poly(train, y_train, test, y_test, degrees = 1, plot='train')
# The model predictions for the testing data are shown compared to the true function and testing data points



fit_poly(train, y_train, test, y_test, degrees = 1, plot='test')
fit_poly(train, y_train, test, y_test, degrees = 25, plot='train')
fit_poly(train, y_train, test, y_test, degrees = 25, plot='test')
fit_poly(train, y_train, test, y_test, degrees = 8, plot='train')
fit_poly(train, y_train, test, y_test, degrees = 8, plot='test')
fit_poly(train, y_train, test, y_test, degrees = 22, plot='train')
fit_poly(train, y_train, test, y_test, degrees = 22, plot='test')
fit_poly(train, y_train, test, y_test, degrees = 190, plot='train')
fit_poly(train, y_train, test, y_test, degrees = 190, plot='test')
# Range of model degrees to evaluate

degrees = [int(x) for x in np.linspace(1, 40, 40)]



# Results dataframe

results = pd.DataFrame(0, columns = ['train_error', 'test_error'], index = degrees)



# Try each value of degrees for the model and record results

for degree in degrees:

    degree_results = fit_poly(train, y_train, test, y_test, degree, plot=False, return_scores=True)

    results.loc[degree, 'train_error'] = degree_results[0]

    results.loc[degree, 'test_error'] = degree_results[1]
print('Training Errors\n')

train_eval = results.sort_values('train_error').reset_index(level=0).rename(columns={'index': 'degrees'})

train_eval.loc[:,['degrees', 'train_error']] .head(10)
print('Testing Errors\n')

train_eval = results.sort_values('test_error').reset_index(level=0).rename(columns={'index': 'degrees'})

train_eval.loc[:,['degrees', 'test_error']] .head(10)
print('Testing Errors\n')

train_eval = results.sort_values('test_error').reset_index(level=0).rename(columns={'index': 'degrees'})

train_eval.loc[:,['degrees', 'test_error']] .head(40)
#plotting both the train and test against the model complexity

plt.plot(results.index, results['train_error'], 'b', ms=6, label = 'Training Error')

plt.plot(results.index, results['test_error'], 'r', ms=6, label = 'Testing Error')

plt.legend(loc=2)

plt.xlabel('Degrees')

plt.ylabel('Mean Square Error')

plt.title('Training and Testing Curves');

plt.ylim(0, 0.05) 

plt.show()



print('\nMinimum Training Error occurs at {} degrees.'.format(int(np.argmin(results['train_error'].values))))

print('Minimum Testing Error occurs at {} degrees.\n'.format(int(np.argmin(results['test_error'].values))))
#plotting both the train and test against the model complexity

plt.plot(results.index, results['train_error'], 'b', ms=10, label = 'Training Error')

plt.plot(results.index, results['test_error'], 'r', ms=10, label = 'Testing Error')

plt.legend(loc=2)

plt.xlabel('Degrees')

plt.ylabel('Mean Square Error')

plt.title('Training and Testing Curves');

plt.ylim(0, 0.10) 

plt.show()



print('\nMinimum Training Error occurs at {} degrees.'.format(int(np.argmin(results['train_error'].values))))

print('Minimum Testing Error occurs at {} degrees.\n'.format(int(np.argmin(results['test_error'].values))))
#plotting both the train and test against the model complexity

plt.plot(results.index, results['train_error'], 'b', ms=100, label = 'Training Error')

plt.plot(results.index, results['test_error'], 'r', ms=100, label = 'Testing Error')

plt.legend(loc=5)

plt.xlabel('Degrees')

plt.ylabel('Mean Square Error')

plt.title('Training and Testing Curves');

plt.ylim(0, 0.10) 

plt.show()



print('\nMinimum Training Error occurs at {} degrees.'.format(int(np.argmin(results['train_error'].values))))

print('Minimum Testing Error occurs at {} degrees.\n'.format(int(np.argmin(results['test_error'].values))))