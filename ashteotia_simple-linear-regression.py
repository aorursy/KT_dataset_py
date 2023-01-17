#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
%matplotlib inline
print('Libraries Imported')
#Import dataset
dirty_training_set = pd.read_csv('../input/train.csv')
dirty_test_set = pd.read_csv('../input/test.csv')
print('Dataset Imported')

training_set = dirty_training_set.dropna() 
test_set = dirty_test_set.dropna()

print(training_set.describe())

x = training_set.iloc[:, :-1].values #features independent variables
y = training_set.iloc[:, 1].values #dependent variable 

x_test = test_set.iloc[:, :-1].values #features independent variables
y_test = test_set.iloc[:, 1].values #dependent variable 

#Lets check correlation between these 2 variables.
plt.scatter(x,y)
plt.xlabel('X-Independent Variable')
plt.ylabel('Y-Dependent Variable')
plt.title('Correlation Between X and Y')
plt.show()
#After Visualizing scatter plot we can see that there is a ositive correlation between X and Y.
#We found that there is positive corelation between them.
#but we need to know how strong is that correlation? We'll use Pearson's r to find it out.
#From scatter plot we concluded that there is positive linear corelation, So our Person's r
#should be (0 < pearson'r >= 1)
x1 = x.ravel()
y1 = x.ravel()
linr = st.linregress(x1, y1)
print(linr)
#Here we can see that rvalue = 1.0, now we can conclude that we have strong positive correlation,
#as our Pearson's r is 1 so no need to calculate the R-Squared value because it will be same in this
#case
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
print('prediction complete')
#Predicting the test set results
y_pred = regressor.predict(x_test)
print(regressor.score(x_test, y_test)*100,'% Prediction Accuracy')
#Prediction vs Actual Test Results
plt.plot(y_pred, '.', y_test, 'x')
plt.xlabel('Independent X')
plt.ylabel('Dependent Y')
plt.title('Prediction Result')
plt.show()
