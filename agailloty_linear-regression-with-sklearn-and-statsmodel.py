# Let's import our libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Since we are going to mae lot of visualization, let's set some visualization parameters in order to have same plots size
plt.rcParams['figure.figsize'] = [12,6]
sns.set_style('darkgrid')
house = pd.read_excel('../input/Maison.xlsx') ## Reading the data
house.head(3)
# Since the columns are in french, in order to make them more readable, let's translate them into English
house = house.rename(index = str, columns = {'PRIX':'price','SUPERFICIE': 'area','CHAMBRES': 'rooms', 
                         'SDB': 'bathroom', 'ETAGES': 'floors','ALLEE': 'driveway',
                         'SALLEJEU':'game_room', 'CAVE': 'cellar', 
                         'GAZ': 'gas', 'AIR':'air', 'GARAGES': 'garage', 'SITUATION': 'situation'})
house.head()
# yay, we have our columns name changed
# Let's now do some visualizations. That's m favourite part 
# Let's see ig we have a linear relation between price and area
sns.scatterplot(house['area'], house['price'], house['gas'], palette = 'viridis')
# We can see some linear trend but as we move along, the dispersion goes wide. We'll fix that later
# Now let's build our model, we will build it both with scikit-learn and with statsmodel
import warnings
warnings.filterwarnings('ignore')
sns.distplot(house['price'])
sns.distplot(house['area'])
# Import the libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# We now instatiate a Linear Regression object
lm = LinearRegression()
# let's do the split of the dataset
house.columns
X = house[['area', 'rooms', 'bathroom', 'floors', 'driveway', 'game_room',
       'cellar', 'gas', 'air', 'garage', 'situation']]
y = house['price']
# I copy this code directly from the function documentation
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)
## Let's chec the head of some of these splits
X_test.head()
# We see that they are randomly selected
# Now let's build the model using sklearn
lm.fit(X_test,y_test)
# Now let's look at the coefficients
print(lm.coef_)
# it would be nicer if we can put them together in a dataframe
coef = pd.DataFrame(lm.coef_, X.columns, columns = ['Coefficients'])
coef
# here we have the coefficients. We can interpret them as follow: "A unit increase of the area (meter square) equals to
# to an increase of the price of $ 3.54 "
# But if we want to to dig more into the statistics, then we should consider using statsmodels whichs gives us more results.
import statsmodels.api as sm
# Unlike sklearn that adds an intercept to our data for the best fit, statsmodel doesn't. We need to add it ourselves
# Remember, we want to predict the price based off our features.
# X represents our predictor variables, and y our predicted variable.
# We need now to add manually the intercepts
X_endog = sm.add_constant(X_test)
res = sm.OLS(y_test, X_endog)
res.fit().summary()
predictions = lm.predict(X_test)
# To check the quality of our model, let's plot it
sns.scatterplot(y_test, predictions)
# We want to know what is the distribution of the residuals. 
sns.distplot(y_test-predictions)
# Evaluation metrics
# Mean Absolute Error (MAE)
# Mean Squared Error (MSE)
# Root Mean Squared Error(RMSE)
import numpy as np
from sklearn import metrics

print('MAE :', metrics.mean_absolute_error(y_test, predictions))
print('MSE :', metrics.mean_squared_error(y_test, predictions))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
