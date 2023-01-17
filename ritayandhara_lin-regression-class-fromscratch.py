# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import statistics

plt.style.use("fivethirtyeight")



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')
data.sample(10)
data.info()
data.describe()
class LinearRegression:

    def __init__(self,fit_intercept=True,normalize=False):

        self.numerator=0

        self.denominator=0

        self.m=0

        self.c=0

        self.fit_intercept = fit_intercept

        self.normalize = normalize

        self.var = 0

        self.X_mean = 0

                

    def fit(self,X,y):

        

        # Normalizing input 

        if (self.fit_intercept == True) & (self.normalize == True):

            self.var = statistics.variance(X)

            self.X_mean=np.mean(X)

            for i in range(len(X)):

                X[i] = (X[i] - self.X_mean) / self.var

        

        # Mean of the input and output

        X_mean=np.mean(X)

        y_mean=np.mean(y)

        

        # Calculating slope

        for i in range(len(X)):

            self.numerator+=(X[i]-X_mean)*(y[i]-y_mean)

            self.denominator+=(X[i]-X_mean)**2

            

        self.m+=self.numerator/self.denominator

        

        # Calculating intercept

        if self.fit_intercept == True:

            self.c += y_mean - (self.m*X_mean)

        else:

            self.c = 0

            

        return self

    

    def predict(self,X):

        if (self.fit_intercept == True) & (self.normalize == True):

            return self.m * ((X-self.X_mean)/self.var) + self.c

        return self.m*X+self.c
# Extracting Input and Label and splitting train/test data



X = np.array(data.iloc[:,0].values)

y = np.array(data.iloc[:,1].values)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)
# Training with fit intercept and normalizing input values



lr = LinearRegression(fit_intercept=True, normalize=True)

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

y_pred
print("R2 Score = ", r2_score(y_test,y_pred))
def extended(ax, x, y, **args):

    xlim = ax.get_xlim()

    ylim = ax.get_ylim()



    x_ext = np.linspace(xlim[0], xlim[1], 100)

    p = np.polyfit(x, y , deg=1)

    y_ext = np.poly1d(p)(x_ext)

    ax.plot(x_ext, y_ext, **args)

    ax.set_xlim(xlim)

    ax.set_ylim(ylim)

    return ax



plt.figure(figsize=(10,10))

ax = plt.subplot()

ax.scatter(X,y)

ax = extended(ax, X_test, y_pred,  color="r", lw=3, label="Predicted Line of Best Fit (extended)")

ax.plot(X_test,y_pred,color='blue', lw=3, label="Predicted Line of Best Fit")

plt.xlabel("Years of Experience")

plt.ylabel("Salary earned")

plt.title("Plotting Line of Best Fit")

ax.legend()

plt.show()