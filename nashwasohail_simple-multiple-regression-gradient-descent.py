# importing libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from math import sqrt
# Load Dataset
df_house = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
# View the dataset
df_house.head()
# Total Rows and Columns
df_house.shape
# Checking Datatypes
df_house.dtypes
df_house = df_house.drop(['id','date'],axis =1)
df_house.head()
df_house.isnull().sum().sort_values(ascending = False)
min_ = min(df_house['price'])
max_ = max(df_house['price'])
x = np.linspace(min_,max_,100)
mean = np.mean(df_house['price'])
std = np.std(df_house['price'])

# For Histogram
plt.hist(df_house['price'], bins=20, density=True, alpha=0.3, color='b')
y = norm.pdf(x,mean,std)

# For normal curve
plt.plot(x,y, color='red')


plt.show()
df_house['price'].describe()
sns.boxplot(df_house['price'])
correlation_matrix = df_house.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix)
# Price & Sqft Living
df_house.plot(x='sqft_living',y='price',style = 'o')
plt.title('Sqft_Living Vs Price')
df_house.boxplot(column = ['price'],by='bedrooms')
df_house.plot(x='lat',y='price',style = 'o')
plt.title('lat Vs Price')
from sklearn.model_selection import train_test_split
train, test = train_test_split(df_house, test_size=0.20)
train.shape
test.shape
X_train_simple = train['sqft_living'].values.reshape(-1,1)
X_test_simple = test['sqft_living'].values.reshape(-1,1)

y_train_simple = train['price'].values.reshape(-1,1)
y_test_simple = test['price'].values.reshape(-1,1)
model_s = LinearRegression()
model_s.fit(X_train_simple,y_train_simple)

print('Intercept: ', model_s.intercept_)

print('Sqft_living Coefficient: ', model_s.coef_)
# Making Predictions
pred_simple = model_s.predict(X_test_simple)
pred_simple
# RSS
RSS_simple = np.sum((y_test_simple - pred_simple)**2)
print("RSS_simple: ", RSS_simple)
plt.plot(test['sqft_living'],test['price'],'.',
        test['sqft_living'], pred_simple,'-')
sns.residplot('sqft_living','price', data = test, color = 'red')
cov = pd.DataFrame.cov(df_house[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms', 
                                 'sqft_basement','waterfront','floors']])
print(cov)


sns.heatmap(cov,fmt='g')
plt.show()
# Separating Attributes and Labels
X_train = train[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms']].values
X_test = test[['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'bedrooms']].values
y_train = train['price'].values.reshape(-1,1)
y_test = test['price'].values.reshape(-1,1)
# Fitting Regression Model
model1 = LinearRegression()
model1.fit(X_train,y_train)

print('Intercept: ', model1.intercept_)

print('Coefficients: ', model1.coef_)

df1 = pd.DataFrame(model1.coef_, columns = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view',
                                            'bedrooms'])
print(df1)
# Making the predictions od model 1
pred1 = model1.predict(X_test)
pred1
# RSS of model 1
RSS_1 = np.sum((y_test - pred1)**2)
print("RSS_1: ", RSS_1)
# Separating Attributes and Labels
X_train = train[['sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms', 'sqft_above', 'sqft_living15','sqft_basement'
                 ,'lat']].values
X_test = test[['sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms','sqft_above', 'sqft_living15','sqft_basement'
              ,'lat']].values

y_train = train['price'].values.reshape(-1,1)
y_test = test['price'].values.reshape(-1,1)
# Fitting a Regression Model 2
model2 = LinearRegression()
model2.fit(X_train,y_train)

print('Intercept: ', model2.intercept_)

print('Coefficients: ', model2.coef_)

df2 = pd.DataFrame(model2.coef_, columns = ['sqft_living', 'grade', 'bathrooms', 'view', 'bedrooms', 
                                           'sqft_above', 'sqft_living15','sqft_basement','lat'])
print(df2)
# Making Predictions of Model 2
pred2 = model2.predict(X_test)
pred2
# RSS of model 2
RSS_2 = np.sum((y_test - pred2)**2)
print("RSS_2: ", np.sum((y_test - pred2)**2))
# Separating Attributes and Labels
X_train = train[['sqft_living', 'grade', 'bathrooms', 'bedrooms','view','lat']].values
X_test = test[['sqft_living', 'grade', 'bathrooms', 'bedrooms','view','lat']].values

# Fiting a Regression Model
model3 = LinearRegression()
model3.fit(X_train,y_train)

print('Intercept: ', model3.intercept_)

print('Coefficients: ', model3.coef_)

df3 = pd.DataFrame(model3.coef_, columns = ['sqft_living', 'grade', 'bathrooms', 'bedrooms', 'view','lat'])
print(df3)
# Predicting the value
pred3 = model3.predict(X_test)
pred3
# RSS
RSS_3 = np.sum((y_test - pred3)**2)
print("RSS_3: ", np.sum((y_test - pred3)**2))
RSS = pd.DataFrame(np.array([[RSS_1,RSS_2,RSS_3]]),columns = ['RSS_1','RSS_2','RSS_3'])
print(RSS)
def get_data(data,features, output):
    data['constant'] = 1
    features = ['constant'] + features

    features_new = data[features]
    feature_matrix = np.asarray(features_new)
    
    output_data = data[output]
    output_array = output_data.to_numpy()
    return(feature_matrix,output_array)
def prediction(feature_matrix, weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)
def feature_derivative(errors,feature):
    derivative = 2*(np.dot(errors,feature))
    return(derivative)
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predictions = prediction(feature_matrix,weights)
        
        errors = predictions - output
        
        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative = feature_derivative(errors,feature_matrix[:,i])
            
            gradient_sum_squares = derivative**2 + gradient_sum_squares
            
            # subtract the step size times the derivative from the current weight
            weights = weights - step_size*(derivative)
            
            gradient_magnitude = sqrt(gradient_sum_squares)
            if gradient_magnitude < tolerance:
                converged = True
    return(weights)  
(feature_matrix,output_array) = get_data(train, ['sqft_living'],'price')
initial_weights = np.array([-47000., 1.])
predictions = prediction(feature_matrix, initial_weights)
errors = output_array - predictions
step_size = 7e-12
tolerance = 2.5e7
weights = regression_gradient_descent(feature_matrix, output_array, initial_weights, step_size, tolerance)
print(weights)
# Using test data to compute RSS
(feature_matrix_test,output_array_test) = get_data(test, ['sqft_living'],'price')

prediction_test = prediction(feature_matrix_test, weights)
errors = output_array_test - prediction_test
RSS_simple_GD = np.sum((errors)**2)
print("RSS_simple_GD: ", np.sum((errors)**2))
print("RSS using Simple Regression: ", RSS_simple)

