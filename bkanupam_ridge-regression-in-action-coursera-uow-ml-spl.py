import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# dictionary with dataset column names and their corresponding data types
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 
              'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

os.chdir('/kaggle/input/polynomialregression')
sales_df = pd.read_csv('kc_house_data.csv', dtype = dtype_dict)
sales_df.sort_values(by=['sqft_living', 'price'], inplace=True)
sales_df.head()
sqft_living = sales_df.loc[:, 'sqft_living'].values.reshape(-1, 1)
price = sales_df.loc[:, 'price'].values.reshape(-1, 1)    
from sklearn.preprocessing import PolynomialFeatures

def create_polynomial_features(degree_n, input_features):
    polynomial_features = PolynomialFeatures(degree=degree_n)
    return polynomial_features.fit_transform(input_features)    
poly1_data = create_polynomial_features(1, sqft_living)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

deg_color_map = {1: 'red', 2: 'green', 3: 'blue', 4: 'cyan', 5: 'grey', 6: 'gold', 7: 'lavender', 8: 'lime', 9: 'magenta', 15: 'coral'}

def regression_degree_n(degree_n, data, regressor):
    # data : data frame containing both input features and target variable
    # returns : The coefficient of degree 1 i.e. theta1
    input_features = data.loc[:, 'sqft_living'].values.reshape(-1, 1)
    y = data.loc[:, 'price'].values.reshape(-1, 1)    
    input_features_degree_n = create_polynomial_features(degree_n, input_features)    
    regressor.fit(input_features_degree_n, y)
    predicted_y = regressor.predict(input_features_degree_n)        
    rmse_deg = np.sqrt(mean_squared_error(y, predicted_y))
    r2_deg = r2_score(y, predicted_y)
    plt.plot(input_features, predicted_y, color=deg_color_map[degree_n], label='degree {}'.format(degree_n))            
    print('For model complexity of polynomial degree {}:'.format(degree_n))
    print('The learned coefficients = {}'.format(regressor.coef_))
    print('The root mean squared error = {}, r2_score = {} '.format(rmse_deg, r2_deg, degree_n))
    return regressor.coef_[0][1]

import matplotlib.pyplot as plt

def plot_sqftliving_price(sqft_living, price):
    fig, ax = plt.subplots(figsize=(12,6))    
    plt.plot(sqft_living, price, 'o', mfc='cyan', mec='orange')
    xrange = np.linspace(0, 14000, 15)
    ax.set_xticks(xrange)
    plt.xlabel('sqft_living')
    plt.ylabel('price')    

plot_sqftliving_price(sqft_living, price)
linear_regressor = LinearRegression(normalize=True)
regression_degree_n(1, sales_df, linear_regressor)    
regression_degree_n(15, sales_df, linear_regressor)    
plt.legend()
plt.title('Polynomial regression')    
plt.show()
from sklearn.linear_model import Ridge

l2_small_penalty = 1e-5
ridge_regressor = Ridge(alpha=l2_small_penalty, normalize=True)
plot_sqftliving_price(sqft_living, price)
regression_degree_n(15, sales_df, ridge_regressor)    
plt.legend()
plt.title('Polynomial regression of degree 15 with l2 penalty')    
plt.show()
# dtype_dict same as above
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
l2_smaller_penalty=1e-9
coeff_deg1_list = []
ridge_regressor_smaller = Ridge(alpha=l2_smaller_penalty, normalize=True)

def fit_and_plot_model(data, regressor, coef_deg1, label):
    plot_sqftliving_price(sqft_living, price)
    coef_deg1_1 = regression_degree_n(15, data, regressor)    
    coef_deg1.append(coef_deg1_1)    
    plt.legend()
    plt.title('Polynomial regression of degree 15 with l2 penalty ({})'.format(label))    
    plt.show()
    
fit_and_plot_model(set_1, ridge_regressor_smaller, coeff_deg1_list, 'set 1')    
fit_and_plot_model(set_2, ridge_regressor_smaller, coeff_deg1_list, 'set 2')    
fit_and_plot_model(set_3, ridge_regressor_smaller, coeff_deg1_list, 'set 3')    
fit_and_plot_model(set_4, ridge_regressor_smaller, coeff_deg1_list, 'set 4')    
sorted_coef_deg1 = sorted(coeff_deg1_list)
print(sorted_coef_deg1)
l2_big_penalty = 1.23e2
coeff_deg1_list_2 = []
ridge_regressor_big = Ridge(alpha=l2_big_penalty, normalize=True)

fit_and_plot_model(set_1, ridge_regressor_big, coeff_deg1_list_2, 'set 1')    
fit_and_plot_model(set_2, ridge_regressor_big, coeff_deg1_list_2, 'set 2')    
fit_and_plot_model(set_3, ridge_regressor_big, coeff_deg1_list_2, 'set 3')   
fit_and_plot_model(set_4, ridge_regressor_big, coeff_deg1_list_2, 'set 4')   
sorted_coef_deg1_2 = sorted(coeff_deg1_list_2)
print(sorted_coef_deg1_2)
train_valid_shuffled = pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
print(len(train_valid_shuffled))
print(len(test))
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in range(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print(i, (round(start, 0), round(end, 0)))
train_valid_shuffled[0:10] # rows 0 to 9
n = len(train_valid_shuffled)
k = 10

def get_segment_indices(num_segments, num_elements):
    k_segments = []
    for i in range(num_segments):
        start = (num_elements*i)/num_segments
        end = (num_elements*(i+1))/num_segments-1
        k_segments.append((int(round(start,0)), int(round(end,0))))
    return k_segments

k_segments = get_segment_indices(k, n)
start_4 = k_segments[3][0]    
end_4 = k_segments[3][1]
validation4 = train_valid_shuffled[start_4:end_4]
print(int(round(validation4['price'].mean(), 0)))
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print(first_two.append(last_two))
pre_validation4 = train_valid_shuffled[0: start_4]
post_validation4 = train_valid_shuffled[end_4+1: n]
train4 = pre_validation4.append(post_validation4)
print(len(validation4))
print(len(train4))
print(int(round(train4['price'].mean(), 0)))
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list, degree_n):
    n = len(data)
    k_segments = get_segment_indices(k, n)
    k_validation_errors = []
    # generate polynomial features from the given input features
    X = data.loc[:, features_list].values.reshape(-1, 1)
    y = data.loc[:, output_name].values.reshape(-1, 1)    
    poly_X = create_polynomial_features(degree_n, X)    
    poly_data = np.concatenate((poly_X, y), axis=1)
    for i, segment in enumerate(k_segments):
        start_i = k_segments[i][0]    
        end_i = k_segments[i][1]
        validation_i = poly_data[start_i:end_i]
        pre_validation_i = poly_data[0: start_i]
        post_validation_i = poly_data[end_i+1: n]
        train_i = np.concatenate((pre_validation_i, post_validation_i), axis=0)
        regressor = Ridge(alpha=l2_penalty, normalize=True)
        # exclude the last column which is the output variable y to get X
        train_X = train_i[:, 0:-1]
        valid_X = validation_i[:, 0:-1]
        # the last column is y
        train_y = train_i[:, -1]
        valid_y = validation_i[:, -1]    
        regressor.fit(train_X, train_y)
        valid_predicted_y = regressor.predict(valid_X)        
        rss = np.sum((valid_y - valid_predicted_y)**2)
        k_validation_errors.append(rss)
    return np.mean(k_validation_errors)

result = k_fold_cross_validation(10, 1000, train_valid_shuffled, 'price', ['sqft_living'], 15)
print(result)
penalty_rss_list = []
for penalty in np.logspace(3, 9, num=13):
    result = k_fold_cross_validation(10, penalty, train_valid_shuffled, 'price', ['sqft_living'], 15)
    penalty_rss_list.append((penalty, result))
    print('penalty:{} --> rss:{}'.format(penalty, result))

sorted_penalty_rss_list = sorted(penalty_rss_list, key=lambda item:item[1])
print(sorted_penalty_rss_list[0])
regressor = Ridge(alpha=sorted_penalty_rss_list[0][0], normalize=True)
X = sales_df.loc[:, 'sqft_living'].values.reshape(-1, 1)
y = sales_df.loc[:, 'price'].values.reshape(-1, 1)    
poly15_X = create_polynomial_features(15, X)    
regressor.fit(poly15_X, y)
# make predictions on the test dataset
X_test = test.loc[:, 'sqft_living'].values.reshape(-1, 1)
poly15_X_test = create_polynomial_features(15, X_test)    
y_test = test.loc[:, 'price'].values.reshape(-1, 1)    
predicted_y_test = regressor.predict(poly15_X_test)        
rss_test = np.sum((y_test - predicted_y_test)**2)
print(rss_test)
