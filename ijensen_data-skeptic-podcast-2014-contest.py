import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

        

import matplotlib.pyplot as plt

import scipy.optimize as opt



df = pd.DataFrame([['2014-06-21', 0, 130],

                   ['2014-07-31', 40, 2313],

                   ['2014-08-30', 70, 6146],

                   ['2014-10-04', 105, 13400]], columns = ['dates', 'dateNums', 'downloads'])



plt.plot(df.dateNums, df.downloads)

plt.scatter(df.dateNums, df.downloads, color='red')

plt.show()
def func(x, A, B, C, D):

    x = np.float128(x) # perform calulation in 128bit to prevent overflow

    return np.float64(A * (x + np.log(1+np.exp(B*(C-x)))/B) + D)
import warnings



from scipy.optimize import differential_evolution



xData = df.dateNums

yData = df.downloads

    

def sumOfSquaredError(parameterTuple):

    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm

    return np.sum((yData - func(xData, *parameterTuple)) ** 2)



def generate_Initial_Parameters():    

    parameterBounds = []

    

    # parameter bounds are selected with trial-and-error

    parameterBounds.append([0, 100]) # parameter bounds for A

    parameterBounds.append([0, 10]) # parameter bounds for B

    parameterBounds.append([0, 10]) # parameter bounds for C

    parameterBounds.append([-100, 10000]) # parameter bounds for D



    # "seed" the numpy random number generator for repeatable results

    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3, maxiter=10000)

    return result.x



# generate initial parameter values

initialParameters = generate_Initial_Parameters()



# print initial parameter values

print("Initial Parameters:")

for name, param in zip(['A','B','C','D'], initialParameters):

    print(name, ": ", param)
from scipy.optimize import curve_fit

# curve fit the test data

fittedParameters, pcov = curve_fit(func, xData, yData, initialParameters)



print("Fitted Parameters")

for name, param in zip(['A','B','C','D'], fittedParameters):

    print(name, ": ", param)
df.plot.scatter(1, 2, color='red')

plt.plot(range(-50, 150), func(range(-50,150), *fittedParameters))

plt.show()
# find intersection

intersection = opt.fsolve(lambda x : func(x, *fittedParameters) - 27182, 0)

print("The model predicts the number will be reached", float(intersection), "days after 2014-06-21")



plt.plot(range(200), func(range(200), *fittedParameters), label='model')

plt.plot(range(200), [27182]*200, label='27182 downloads')

plt.scatter([intersection], func(intersection, *fittedParameters), color='red')

plt.legend()

plt.show()
import datetime

start_date = datetime.datetime.strptime("2014-06-21", "%Y-%m-%d")

end_date = start_date + datetime.timedelta(days=float(intersection))



print("The crossover will occur at", end_date)