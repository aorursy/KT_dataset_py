import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

IceCream = pd.read_csv("../input/IceCreamData.csv") #Upload the dataset called 'IceCream.csv' attached on the same folder to the programme and display it on the screen.
IceCream
IceCream.describe() #Obtaining general information about the dataset. The number of elements on the dataset, the mean, standard deviation, etc.
type(IceCream) #Showing the type of object we are working with.
IceCream.info() #Obtaining more general information about the dataset like the memory that it occupies.
sns.jointplot(x = 'Temperature', y = 'Revenue', data = IceCream) #Plotting Revenue per day as a function of the emperature in that given day.
sns.pairplot(IceCream) #Other method of visualization for Temperature against Revenue and viceversa.
sns.lmplot( x = 'Temperature', y = 'Revenue', data = IceCream) #Selecting and visualizing the Revenue against the Temperature and plotting a line of best fit to check how the graph tends to behave.
x = IceCream[['Temperature']] #Defining 'x' as all the information on the Temperature column.
y = IceCream[['Revenue']] #Defining 'y' as all the information on the Revenue column.
from sklearn.model_selection import train_test_split #Importing machine learning library.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25) #Defyning what percentage of values from x and y are for training and for testing the model. Assigning 25% of values to testing porpoises.
x_train #Values from x used to train the machine learning model. Randomly organized. 75%
x_test  #Values from x used to test the machine learning model. Randomly organized. 25%
y_train  #Values from y used to train the machine learning model. Randomly organized. 75%
y_test  #Values from y used to train the machine learning model. Randomly organized. 25%
from sklearn.linear_model import LinearRegression #Import the Linear Regression library as our machine learning model.
regressor = LinearRegression(fit_intercept = True) #Intercept values from the graph accepted. Neededd for improvement on accuracy.
regressor.fit(x_train, y_train) #Select the trained data and fit it on the Linear Regression model.
print('Linear model coefficient (m)', regressor.coef_) #Print the linear coeficient (m) of this dataset. Corresponds to the mathematical equation y = mx + b.
print('Linear model coefficient (b)', regressor.intercept_) #Print the linear coeficient (b) of this dataset. Corresponds to the mathematical equation y = mx + b.
x_train.shape #Print the shape of the trained x data.
y_train.shape #Print the shape of the trained y data.
x_test.shape #Print the shape of the tested x data.
y_test.shape #Print the shape of the tested y data.
plt.scatter(x_train, y_train, color = 'red') #Plot a graph with the trained data for the Revenue against the Temperature in red.
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #Add a line of best fit for this condition.
plt.ylabel('Revenue[$]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Vs. Temperature [Training Data]')
plt.scatter(x_test, y_test, color = 'red') #Plot a graph with the tested data for the Revenue against the Temperature in red.
plt.plot(x_test, regressor.predict(x_test), color = 'blue') #Add a line of best fit for this condition.
plt.ylabel('Revenue[$]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Vs. Temperature [Testing Data]')
Sample_Test = np.array([[35]]) #Set the temperature to 35 degrees as an example.
y_predict_test = regressor.predict(Sample_Test) #Use the data trained to predict the revenue when at 35 decgrees Celsius and check if it satisfies the Revenue Vs. Temperature [Testing Data] graph.
y_predict_test
degrees = float(input("Enter today's temprature in Cº: ")) #Allow the user to enter any number as the temperature.
Today_T = np.array([[degrees]]) #Upload the number to the programme.
predict_revenue = regressor.predict(Today_T) #Use the data trained to predict the revenue when at the user's chosen temperature in decgrees Celsius.
print('Expected benefits= ', predict_revenue, '$') 