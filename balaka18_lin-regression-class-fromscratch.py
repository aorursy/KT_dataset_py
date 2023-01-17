# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as ml

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('ggplot')



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
test_data = pd.read_csv('/kaggle/input/years-of-experience-and-salary-dataset/Salary_Data.csv')

print(test_data.shape)

test_data.head(10)
test_data.info()
test_data.describe()
class LinearRegression():

    def __init__(self,fit_intercept=True):

        self.numerator = 0

        self.denominator = 0

        self.b0 = 0

        self.b1= 0

        self.fit_intercept = fit_intercept

    def fit(self,datax,datay):

        # Mean of the input and output

        meanx = np.mean(datax)

        meany = np.mean(datay)

        

        # Total number of values

        N = len(datax)      # datax must be an ndarray or a list.

        

        # Formula to calculate b1 and b0

        '''Basic formula of the best fit line :

            y = b0 + b1*x, where

            

            b0 = intercept

            b1 = slope

            

            Now, b1 = [(Xi - X_bar)*(Yi - Y_bar)]/[(Xi - X_bar)^2]

                 b0 = Y_bar - (b1*X_bar)

        '''

        for i in range(N):

            self.numerator += (datax[i] - meanx)*(datay[i] - meany)

            self.denominator += (datax[i] - meanx)**2

        self.b1 += self.numerator/self.denominator

        if self.fit_intercept == True:

            self.b0 += meany - (self.b1*meanx)

        else:

            self.b0 = 0 

        

        return self

    def predict(self,testx):

        y = self.b1*testx + self.b0

        return y
X = np.array(test_data.iloc[:,0].values)

y = np.array(test_data.iloc[:,1].values)

trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.3,random_state=5)



lr = LinearRegression(fit_intercept=True)

lr.fit(trainx,trainy)

y_pred = lr.predict(testx)

y_pred
print("R2 Score for this model = ", r2_score(testy,y_pred))
plt.figure(figsize=(10,5))

plt.plot(testx,y_pred,color='blue',label="Best Fit Line")

plt.scatter(X,y,color='red',label="Data points")

plt.xlabel("YearsExperience")

plt.ylabel("Salary")

plt.title("PLOTTING THE BEST FIT CURVE")

plt.show()