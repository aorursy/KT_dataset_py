import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from   sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")
# Load data GlobalTemperatures
colnames    = ['dt', 'LandAverageTemperature', 'LandAverageTemperatureUncertainty']
newnames    = ['dt', 'at', 'atu']
datatypes   = {'dt': 'str','at':'float32','atu':'float32'}
temperature = pd.read_csv("../input/GlobalTemperatures.csv", 
                            usecols = colnames, 
                            dtype = datatypes)
temperature.columns = newnames
temperature = temperature[pd.notnull(temperature['at'])]
temperature['dt'] = temperature['dt'].map(lambda x: int(x.split('-')[0]))
group = temperature.groupby('dt').mean()


plt.figure(figsize=(8,6))
plt.scatter(group.index, group['at'], s=40, c='darkblue', alpha=0.5, linewidths=0, label='Mean Temperature')
plt.legend(loc='upper left')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Mean of temperature by years')
plt.savefig('temperature.png')
def addpolynomialfeatures(subX, x):
    subX['x2'] = x**2
    subX['x3'] = x**3
    subX['x4'] = x**.5
    subX['x5'] = np.sin(x)
    subX['x6'] = np.cos(x)
    subX['x8'] = np.log(x)
X         = pd.DataFrame(group.index)
addpolynomialfeatures(X, X['dt'])
y         = pd.DataFrame(group['at'])
X.index   = X['dt']
Xy        = pd.concat([X,y], axis=1)
regresor  = linear_model.LinearRegression()
regresor2 = linear_model.BayesianRidge(compute_score=True)
regresor2.fit(X, y)
predict2   = regresor2.predict(X)
regresor.fit(X,y,5000)
predict   = regresor.predict(X)
print('Coefficients: \n', regresor.coef_)
print("Mean of error: %.2f" % np.mean((predict - y) ** 2))
plt.figure(figsize=(8,6))
plt.scatter(group.index, group['at'], s=40, c='darkblue', alpha=0.5, linewidths=0, label='Expected Output')
for i in range(50):
    sample = Xy.sample(n=40)
    X_test = sample[Xy.columns[:-1]]
    y_test = sample[Xy.columns[-1]]
    regresor.fit(X_test,y_test)
    predict   = regresor.predict(X)
    plt.plot(group.index, predict, c='red', alpha=0.1, linewidth=2.)
plt.legend(loc='upper left')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Mean of temperature by years')
plt.savefig('temperature.png')
plt.figure(figsize=(8,6))
plt.scatter(group.index, group['at'], s=40, c='darkblue', alpha=0.5, linewidths=0, label='Expected Output')
for i in range(50):
    sample = Xy.sample(n=30)
    X_test = sample[Xy.columns[:-1]]
    y_test = sample[Xy.columns[-1]]
    regresor2.fit(X_test,y_test)
    predict2   = regresor2.predict(X)
    plt.plot(group.index, predict2, c='red', alpha=0.1, linewidth=3.)
plt.legend(loc='upper left')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Mean of temperature by years')
plt.savefig('temperature.png')