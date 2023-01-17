import pandas as pd

import numpy as np



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error





import seaborn as sns

# import xgboost as xgb



import matplotlib.pyplot as plt





import statsmodels.api as sm



from statsmodels.stats.outliers_influence import variance_inflation_factor



import warnings as ws

ws.defaultaction = "ignore"
# airbnb_data = pd.read_csv('AB_NYC_2019.csv')

# from google.colab import files

# uploaded = files.upload()

# import io

data = pd.read_csv('../input/insurance/insurance.csv')
data
print('Data shape: ',data.shape, '\n')

print('*******************************')

print('Data means:\n',data.mean(), '\n')

print('*******************************')

print('Data features count:\n',data.count(), '\n')

print('*******************************')

print('Data Info about null vals:\n',data.info(), '\n')

print('*******************************')

print('Data Features null vals:\n',data.isnull().sum(), '\n')
# Insurance charges histogram (How good is its skew value?)

plt.figure(figsize=(10, 8))

plt.hist(data['charges'], bins = 50 ,color='#3f4c6b', ec='#606c88')

plt.title('Insurance charges in $ vs Nr. of people', fontsize=18)

plt.ylabel('Nr. of people', fontsize=14)

plt.xlabel('Prices in $', fontsize=14)

plt.show()
# Changing "sex" feature to 0s and 1s => 0s: female; 1s: male

data['sex'] = data.sex.replace({"female" :0, "male" : 1 })



# Changing"smoker" features to 0s and 1s => 0s: no; 1s: yes

data['smoker'] = data.smoker.replace({"yes": 1, "no" : 0 })



data['region'] = data.region.replace({"southeast": 0, "southwest" : 1,

                                     "northeast":2, "northwest":3})

# Extracting relevant data and ignoring repetitive correlations

mask = np.zeros_like(data.corr())

triangle_indices = np.triu_indices_from(mask)

mask[triangle_indices] = True

data.corr()
# Correlations value graph

plt.figure(figsize=(10, 8))



sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={"size":14})



#Analysis: We can clearly notice that there is a strong correlation between the age and the charges
sns.set_context('talk')

sns.set_style('darkgrid')

g = sns.FacetGrid(data, row="smoker", col="sex", margin_titles=True, height=5, )

g.map(sns.regplot, "bmi", "charges", color="#12c2e9", x_jitter=.1, line_kws={"color": "#f64f59"})
region_charges = sns.catplot(x="region", y='charges', data=data, legend_out = False,

            height=8, hue="sex", kind='bar', palette=["#f64f59", "#12c2e9"]);



# region_charges.set_title('Region vs. Charges by gender')

leg = region_charges.axes.flat[0].get_legend()

region_charges.set(xlabel='Regions', ylabel='Charges', 

                   title='Regions vs. Insurance Charges')



region_charges.set_xticklabels(['Southeast','Southwest','Northeast','Northwest'])





leg.set_title('Gender')

new_labels = ['Felmale', 'Male']

for t, l in zip(leg.texts, new_labels): t.set_text(l)

plt.show()







child_charges = sns.catplot(x="children", y='charges', data=data, height=8, legend_out = False,

           kind='bar', palette=["#aa4b6b", "#3b8d99"]);



child_charges.set(xlabel='# of Children', ylabel='Charges', 

                   title='Nr. of Children vs. Insurance Charges')



charges = data['charges']

features = data.drop(['charges'], axis=1) #Dropping charges collumn



X_train, X_test, y_train, y_test = train_test_split(features, 

                                                    charges, 

                                                    test_size= 0.2, 

                                                    random_state=42)



regression = LinearRegression()

model = regression.fit(X_train, y_train)

prediction = regression.predict(X_test)



print('Test Data r-Squared score: ', regression.score(X_test, y_test))

print('Train Data r-Squared score: ', regression.score(X_train, y_train))

print(X_train, y_train)



pd.DataFrame(data=regression.coef_, index=X_train.columns, columns=['coef'])



# Pre-transformation skew val

pre_trans = round(data['charges'].skew(), 3)

print('Pre-transformation skew val: ', pre_trans)

sns.distplot(data['charges'])

plt.title(f'Original Charges with skew {pre_trans}')

plt.show()
# Post-transformation skew val

post_trans = round(np.log(data['charges'].skew()), 3)

print('Post-transformation skew val: ', post_trans)



y_log = np.log(data['charges'])

sns.distplot(y_log)

plt.title(f'Log Charges with skew {post_trans}')
# Apply the transformation.

log_charges = np.log(data['charges'])



transformed_data = data.drop('charges', axis=1)





X_train, X_test, y_train, y_test = train_test_split(transformed_data, 

                                                    log_charges, 

                                                    test_size= 0.2, 

                                                    random_state=42)



regression_t = LinearRegression()

model_t = regression_t.fit(X_train, y_train)

prediction_t = regression_t.predict(X_test)



pd.DataFrame(data=regression_t.coef_, index=X_train.columns, columns=['coef'])



plt.scatter(y_test, prediction_t)

plt.plot(y_test, y_test, color='red')

rmse = np.sqrt(mean_squared_error(y_test, prediction_t))





print('Intercept: ', regression_t.intercept_)

print('Coef: ', regression_t.coef_)

print('rmse: ', rmse)

print('Test Data r-Squared score: ', regression_t.score(X_test, y_test))

print('Train Data r-Squared score: ', regression_t.score(X_train, y_train))





x_include_const = sm.add_constant(X_train) #Adding an intercept



model = sm.OLS(y_train, x_include_const) 

results = model.fit()





# Graph of Actual vs. Predicted Prices

plt.figure(figsize=(10, 8))

corr = round(y_train.corr(results.fittedvalues), 2)

plt.scatter(x=y_train, y=results.fittedvalues, c='black', alpha=0.6)

plt.plot(y_train, y_train, color='cyan')



plt.xlabel('Actual log prices $y _i$', fontsize=14)

plt.ylabel('Predicted log prices $\hat y _i$', fontsize=14)

plt.title(f'Actual vs Predicted log prices $y _i$ vs $\hat y _i$ (Corr: {corr})', 

          fontsize=18)



plt.show()





pd.DataFrame({'Coef' : results.params, 

             'P-values' : round(results.pvalues, 3)})

#Hence, all the features are statistically significant
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR



from sklearn.model_selection import cross_val_score
##new training model using RandomForest Regression

charges = data['charges']



transformed_data = data.drop(['charges', 'region'], axis=1)

transformed_data
X_train, X_test, y_train, y_test = train_test_split(transformed_data, log_charges, test_size = 0.2)
randomForest = RandomForestRegressor(n_estimators=150, random_state=12)



model = randomForest.fit(X_train, y_train)
model.score(X_test, y_test)
model.score(X_train, y_train)
# model2.score(X_test, y_test)
decTree = DecisionTreeRegressor()

model2 = decTree.fit(X_train, y_train)
model2.score(X_test, y_test)
model2.score(X_train, y_train)
# Apply the transformation.

log_charges = np.log(data['charges'])



transformed_data = data.drop('charges', axis=1)





X_train, X_test, y_train, y_test = train_test_split(transformed_data, 

                                                    log_charges, 

                                                    test_size= 0.2, 

                                                    random_state=42)



print(X_train.shape)

print(y_train.shape)

y_train = np.array(y_train).reshape(-1, 1)

y_train.shape
def estimate_coefficients(x, y): 

    # size of the dataset OR number of observations/points 

    n = np.size(x) 

  

    # mean of x and y

    # Since we are using numpy just calling mean on numpy is sufficient 

    mean_x, mean_y = np.mean(x), np.mean(y) 

  

    # calculating cross-deviation and deviation about x 

    SS_xy = np.sum(y*x - n*mean_y*mean_x) 

    SS_xx = np.sum(x*x - n*mean_x*mean_x) 

  

    # calculating regression coefficients 

    b_1 = SS_xy / SS_xx 

    b_0 = mean_y - b_1*mean_x 

  

    return(b_0, b_1)



    # x,y are the location of points on graph

    # color of the points change it to red blue orange play around







def plot_regression_line(x, y, b): 

    # plotting the points as per dataset on a graph

    plt.scatter(x, y, color = "m",marker = "o", s = 30) 



    # predicted response vector 

    y_pred = b[0] + b[1]*x 

  

    # plotting the regression line

    plt.plot(x, y_pred, color = "g")

  

    # putting labels for x and y axis

    plt.xlabel('Size') 

    plt.ylabel('Cost') 

  

    # function to show plotted graph

    plt.show()

    

# b = estimate_coefficients(X_train, y_train) 

# print("Estimated coefficients:\nb_0 = {} \nb_1 = {}".format(b[0], b[1])) 

  

# plotting regression line 

estimate_coefficients(X_train, y_train)
# Function mse(y, y_hat)



def mse(y, y_hat):

    #calc = (1/y.size) * sum((y - y_hat)**2)

    mse_calc = np.average((y - y_hat)**2, axis=0)

    return mse_calc
# # x values, y values, array of theta parameters (theta0 at index 0 and theta1 at index 1)

# def grad(x, y, thetas):

#     n = y.size

#     theta0_slope = (-2/n) * sum(y - thetas[0] - thetas[1]*x)

#     theta1_slope = (-2/n) * sum((y - thetas[0] - thetas[1]*x)*x)

    

# #     return np.array([theta0_slope[0], theta1_slope[0]])

# #     return np.append(arr=theta0_slope, values= theta1_slope)

#     return np.concatenate((theta0_slope, theta1_slope), axis = 0)

def grad(x, y, thetas):

    #use equation

    n = y.size

    theta0_slope = (-2/n) * sum(np.double(y) - np.double(thetas[0]) - np.double(thetas[1]*x))

    theta1_slope = (-2/n) * sum((np.double(y) - np.double(thetas[0]) - np.double(thetas[1]*x))*np.double(x))

    

#     return np.array([theta0_slope[0], theta1_slope[0]])

#     return np.append(arr=theta0_slope, values= theta1_slope)

    return np.concatenate((theta0_slope, theta1_slope), axis = 0)
multiplier = 0.01

thetas = np.array([1.9, 1.9])

mse_vals = mse(y_train, thetas[0]+thetas[1]*X_train)

print(thetas[0])
x_5 = np.array([(1,2,3, 4,5,6, 7), (1,2,3, 4,5,6, 7), (1,2,3, 4,5,6, 7), (1,2,3, 4,5,6, 7), (1,2,3, 4,5,6, 7), (1,2,3, 4,5,6, 7), (1,2,3, 4,5,6, 7)])

y_5 = np.array([1.7, 2.4, 3.5, 3.0, 6.1, 9.4, 8.2]).reshape(7, 1)

grad(x_5, y_5, thetas)
print(X_train.dtypes)

# print(y_train.dtype)
grad(X_train, y_train, thetas)
print(y_train.shape)

print(X_train.shape)
multiplier = 0.01

thetas = np.array([2.9, 2.9,2.9,2.9,2.9,2.9])



#Collect data point for scatter plot

plot_vals = thetas.reshape(1, 6)

mse_vals = mse(y_train, thetas[0]+thetas[1]*X_train)



for i in range(1000):

    thetas = thetas - multiplier * grad(X_train, y_train, thetas)

    plot_vals = np.concatenate((plot_vals, thetas.reshape(1, 2)), axis=0)

    

    mse_vals = np.append(arr=mse_vals, 

                         values=mse(y_train, thetas[0] + thetas[1]*X_train))

#Results

print('Min occurs at Theta 0 :', thetas[0])

print('Min occurs at Theta 1 :', thetas[1])

print('MSE is:', mse(y_train, thetas[0] + thetas[1]*X_train))
# # Quick linear regression

# regr = LinearRegression()

# regr.fit(x_1, y_1)



# print('Theta 0: ', regr.intercept_[0])

# print('Theta 1: ', regr.coef_[0][0])
# # y_hat = theta0 + theta1*x

# y_hat = -0.06339843177975801 + 1.497577113036996*x_1

# print('Estimated values y_hat are: \n', y_hat)

# print('In comparison, the actual y values are \n', y_1)