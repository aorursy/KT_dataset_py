# Import the libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



import seaborn as sns    # plot tools



# Take a look in the Stocks directory to select one dataset from a company



import os

print(os.listdir("../input/Data/"))



# Any results you write to the current directory are saved as output.
# Directory of the dataset 

filename = '../input/Data/Stocks/googl.us.txt'



# Read the file

Prgoo = pd.read_csv(filename,sep=',',index_col='Date')



# Prices is the predict value and initial the independet variable (y)

prices = Prgoo['Close'].tolist()

initial = (Prgoo['Open']).tolist()

 

#Convert to 1d Vector

prices = np.reshape(prices, (len(prices), 1))

initial = np.reshape(initial, (len(initial), 1))



Prgoo.head(5)

Prgoo[['Open']].plot()

plt.title('Google Open Price')

plt.show()



Prgoo[['Close']].plot()

plt.title('Google Close Price')

plt.show()
plt.subplots(figsize=(8,6))

sns.heatmap(Prgoo.corr(),annot=True, linewidth=.5,)
sns.distplot(Prgoo['Open'], hist = False, kde = True, kde_kws = {'linewidth': 5},label='Open',) 

sns.distplot(Prgoo['Close'], hist = False, kde = True, kde_kws = {'linewidth': 3},label='Close') 



plt.legend(prop={'size': 10}, title = 'Types',loc= 'best')

plt.title('Density Plot the Open and Close of the stock prices')

plt.xlabel('Prices')

plt.ylabel('Density')
#Splitting the dataset into the Training set and Test set

xtrain, xtest, ytrain, ytest = train_test_split(initial, prices, test_size=0.33, random_state=42)

regressor = LinearRegression()

regressor.fit(xtrain, ytrain)

 

#Train Set Graph

print('Train-set /','R2 score:',r2_score(ytrain,regressor.predict(xtrain)))

plt.scatter(xtrain, ytrain, color='red', label= 'Actual Price') #plotting the initial datapoints

plt.plot(xtrain, regressor.predict(xtrain), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression

plt.title('Linear Regression price| Open vs. Close')

plt.legend()

plt.xlabel('Prices')

plt.show()

 

#Test Set Graph

print('Test-set/','R2 score:',r2_score(ytest,regressor.predict(xtest)))

plt.scatter(xtest, ytest, color='red', label= 'Actual Price') #plotting the initial datapoints

plt.plot(xtest, regressor.predict(xtest), color='blue', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression

plt.title('Linear Regression price| Open vs. Close')

plt.legend()

plt.xlabel('Prices')

plt.show()