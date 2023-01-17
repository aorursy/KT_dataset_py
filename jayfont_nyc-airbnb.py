import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, TheilSenRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from scipy import stats

import tensorflow as tf

from tensorflow.keras.utils import normalize

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
dataset = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")  

dataset.head()
dataset.info()
dataset.describe()
sns.boxplot(dataset['price'])
# See all unique neighborhoods

all_boroughs = dataset['neighbourhood_group'].unique()



# Isolate neighborhood and price

neigh = dataset[['neighbourhood_group', 'price']]



# Isolate Each Neighborhood

def get_borough_price_data(df, name):

    

    query = "neighbourhood_group == '" + name + "'"

    df = neigh.query(query)

    mean_price = np.mean(df['price'])

    return df, mean_price



[bk, bk_avg_price] = get_borough_price_data(neigh, "Brooklyn")

[m, m_avg_price] = get_borough_price_data(neigh, "Manhattan")

[bx, bx_avg_price] = get_borough_price_data(neigh, "Bronx")

[q, q_avg_price] = get_borough_price_data(neigh, "Queens")

[s, s_avg_price] = get_borough_price_data(neigh, "Staten Island")



# Concatenate

avg_prices_by_borough = [bk_avg_price, m_avg_price, q_avg_price, s_avg_price, bx_avg_price]
# Plot Average Price by Borough

fig = plt.figure()

bar = fig.subplots()

bar.bar(all_boroughs, avg_prices_by_borough)

bar.set_title("Average Prices of Nightly AirBnb Rental by Neighborhood")

bar.set_xlabel("Neighborhoods")

bar.set_ylabel("Price ($)");
# Remove non-numerical features

data = dataset.select_dtypes(exclude='object')

data = data.drop(columns=['id', 'host_id'])

unscored_data = data.dropna()

data.head()

cols = data.columns



# Remove some outliers with z-scoring

z_scores = np.abs(stats.zscore(unscored_data)) # calculate z scores

print("Max before removal: " + str(data['price'].max()))

data = unscored_data[(z_scores < 2).all(axis=1)] # filter for only values with z scores less than 2

sns.boxplot(data['price'])

print("Max after removal: " + str(data['price'].max()))

data.describe()
# Scale data

scaler = StandardScaler()

scaler.fit(data)

scaled_data = scaler.transform(data)



# Estimating required number of components

digits = [i for i in range(1,9)]

explained_var = []

for i in digits:

    pca = PCA()

    explained_var.append(pca.fit(scaled_data).explained_variance_ratio_.cumsum()[i-1])



plt.plot(digits, explained_var)

plt.title("Cumulated Variance by Number of Components")

plt.xlabel("# of Components")

plt.ylabel("Variance")



# Fit PCA, than transform data

pca = PCA(n_components=7) # Can be any number

pca.fit(scaled_data) # Fit pca model

pca_data = pca.transform(scaled_data) # scale data to two components



fig = plt.figure(figsize=(8,8))

p = fig.subplots()

p.scatter(pca_data[:,0], pca_data[:,1], c=data['price']) # visualize pca
print("Explained variance ratio: {}".format(pca.explained_variance_ratio_))

print("Explained Variance: {}".format(pca.explained_variance_ratio_.cumsum()))



# Plot components by each feature

components = pd.DataFrame(pca.components_, columns = cols)

sns.heatmap(components)
# Isolate desired data

reg_data = data[['latitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'price']]



# Set x to all features except price and set y to price

x_train, x_test, y_train, y_test = train_test_split(reg_data.drop('price', axis=1), reg_data['price'], test_size=0.3, random_state=101)

x_train = np.array(x_train)

x_test = np.array(x_test)

y_train = np.array(y_train)

y_test = np.array(y_test)



# Scale data separately for x and y

x_scale = StandardScaler().fit(x_train)

x_train = x_scale.transform(x_train)

x_test = x_scale.transform(x_test)



y_scale = StandardScaler().fit(y_train.reshape(-1,1))

y_train = y_scale.transform(y_train.reshape(-1,1))

y_test = y_scale.transform(y_test.reshape(-1,1))



# Create Model and train with scaled data

lm = LinearRegression()

lm.fit(x_train, y_train)

pred = lm.predict(x_test)



fig = plt.figure(figsize=(9,10))

l, res = fig.subplots(2)



# Unscale data to get results

unscaled_predictions = y_scale.inverse_transform(pred)

unscaled_test_values = y_scale.inverse_transform(y_test)



l.scatter(unscaled_test_values, unscaled_predictions)

l.set_xlabel("Test Values")

l.set_ylabel("Predicted Values")



res.scatter(unscaled_test_values, unscaled_test_values-unscaled_predictions)

res.set_title("Residuals vs Predicted Values")

res.set_xlabel("Test Values")

res.set_ylabel("Residuals")
mean_from_set = data['price'].mean()

mean_from_model = y_scale.inverse_transform(pred).mean()

print("Actual mean price: {}\nPredicted mean price: {}".format(mean_from_set,mean_from_model))

mse = metrics.mean_squared_error(y_test, pred)

print("MSE: " + str(mse))

print("RMSE: " + str(np.sqrt(mse)))
def test_against_samples(num, data, model, x_scale, y_scale):

    test_listing = data.sample(num)



    # Isolate same features from testing

    test = test_listing[['latitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month']]



    # Scale testing data, then use model to predict

    test_pred = model.predict(x_scale.transform(test))



    # Unscale predictions to get final results

    try:

        test_pred = y_scale.inverse_transform(test_pred).transpose()

    except ValueError:

        test_pred = y_scale.inverse_transform(test_pred.reshape(-1,1)).transpose()

    

    return test_pred, test_listing
def plot_samples(test_listing, test_pred, model_name):

    

    bar_x1 = [i-0.5 for i in range(1,n_tests+1)]

    bar_x2 = [i+0.5 for i in bar_x1]

    

    try:

        vals = []

        for space in test_pred:

            for val in space:

                vals.append(val)



        fig = plt.figure(figsize=(8,8))

        ax = fig.subplots()

        ax.bar(bar_x1, test_listing['price'], width=0.3, color='navy')

        ax.bar(bar_x2, vals, width=0.3, color='r')

        ax.set_xlabel("Trials")

        ax.set_ylabel("Prices")

        ax.set_title("Bar Plot of Real Prices (Navy) vs Predicted Prices (Red) for {} Model".format(model_name))

    except: 

        bar_x1 = [i-0.5 for i in range(1,n_tests+1)]

        bar_x2 = [i+0.5 for i in bar_x1]



        fig = plt.figure(figsize=(8,8))

        ax = fig.subplots()

        ax.bar(bar_x1, test_listing['price'], width=0.3, color='navy')

        ax.bar(bar_x2, test_pred, width=0.3, color='r')

        ax.set_xlabel("Trials")

        ax.set_ylabel("Prices")

        ax.set_title("Real Prices (Navy) vs Predicted Prices (Red) for {} Model".format(model_name))

        

    fig = plt.figure(figsize=(10,10))

    ax = fig.subplots()

    ax.plot(range(1,n_tests+1), test_listing['price'], color='navy')

    ax.plot(range(1,n_tests+1), test_pred.reshape(-1,1), color='red')

    ax.set_title("Line Plot of Real Prices (Navy) vs Predicted Prices (Red) for {} Model".format(model_name))

    ax.set_ylabel("Prices")

    ax.set_xlabel("Trials")
# Testing against samples



n_tests = 5000



(test_pred, test_listing) = test_against_samples(n_tests, data, lm, x_scale, y_scale)



plot_samples(test_listing, test_pred, 'LinearRegression')
# Trying another Regression Model

ridge = Ridge()

ridge.fit(x_train, y_train)

ridge_pred = ridge.predict(x_test)



fig = plt.figure(figsize=(9,10))

reg, res = fig.subplots(2)



unscaled_predictions_r = y_scale.inverse_transform(ridge_pred)

unscaled_test_values = y_scale.inverse_transform(y_test)



reg.scatter(unscaled_test_values, unscaled_predictions_r)

reg.set_xlabel("Test Values")

reg.set_ylabel("Predicted Values")

reg.set_title("Test vs Predicted Prices")



res.scatter(unscaled_test_values, unscaled_test_values-unscaled_predictions_r)

res.set_title("Residuals vs Predicted Values")

res.set_xlabel("Test Values")

res.set_ylabel("Residuals")
mse = metrics.mean_squared_error(y_test, ridge_pred)

print("MSE: " + str(mse))

print("RMSE: " + str(np.sqrt(mse)))
# Testing against a real posting

test_pred, test_listing = test_against_samples(n_tests, data, ridge, x_scale, y_scale)

plot_samples(test_listing, test_pred, "Ridge")
mean_from_set = data['price'].mean()

mean_from_model = y_scale.inverse_transform(ridge_pred).mean()

print("Actual mean price: {}\nPredicted mean price: {}".format(mean_from_set,mean_from_model))
# Trying a Theil Sen Regression, which may be resilient to outliers

ts = TheilSenRegressor().fit(x_train, np.ravel(y_train))

ts_pred = ts.predict(x_test)



fig = plt.figure(figsize=(9,10))

reg, res = fig.subplots(2)



unscaled_predictions_ts = y_scale.inverse_transform(ts_pred)

unscaled_test_values = y_scale.inverse_transform(np.ravel(y_test))



reg.scatter(unscaled_test_values, unscaled_predictions_ts)

reg.set_xlabel("Test Values")

reg.set_ylabel("Predicted Values")

reg.set_title("Test vs Predicted Prices")



res.scatter(unscaled_test_values, unscaled_test_values-unscaled_predictions_ts)

res.set_title("Residuals vs Predicted Values")

res.set_xlabel("Test Values")

res.set_ylabel("Residuals")
mse = metrics.mean_squared_error(y_test, ts_pred)

print("MSE: " + str(mse))

print("RMSE: " + str(np.sqrt(mse)))
mse = metrics.mean_squared_error(y_test, ts_pred)

print("MSE: " + str(mse))

print("RMSE: " + str(np.sqrt(mse)))
# Testing against a real posting

test_pred, test_listing = test_against_samples(n_tests, data, ts, x_scale, y_scale)

plot_samples(test_listing, test_pred, 'Theil Sen')
mean_from_set = data['price'].mean()

mean_from_model = y_scale.inverse_transform(ts_pred).mean()

print("Actual mean price: {}\nPredicted mean price: {}".format(mean_from_set,mean_from_model))
# Trying a Random Forest

rfr = RandomForestRegressor(n_estimators=4, random_state=0)

rfr.fit(x_train, y_train)

rfr_pred = rfr.predict(x_test)



fig = plt.figure(figsize=(9,10))

reg, res = fig.subplots(2)



unscaled_predictions_rfr = y_scale.inverse_transform(rfr_pred)

unscaled_test_values = y_scale.inverse_transform(np.ravel(y_test))



reg.scatter(unscaled_test_values, unscaled_predictions_rfr)

reg.set_xlabel("Test Values")

reg.set_ylabel("Predicted Values")

reg.set_title("Test vs Predicted Prices")



res.scatter(unscaled_test_values, unscaled_test_values-unscaled_predictions_rfr)

res.set_title("Residuals vs Predicted Values")

res.set_xlabel("Test Values")

res.set_ylabel("Residuals")
mse = metrics.mean_squared_error(y_test, rfr_pred)

print("MSE: " + str(mse))

print("RMSE: " + str(np.sqrt(mse)))
# Testing against a real posting

test_pred, test_listing = test_against_samples(n_tests, data, rfr, x_scale, y_scale)

plot_samples(test_listing, test_pred, "Random Forest Regressor")



diff = []

count=0

for price in test_listing['price']:

    diff.append(np.abs(price - test_pred[count]))

    count+=1

print('mean difference between test and predicted prices: {}'.format(np.mean(diff)))
mean_from_set = data['price'].mean()

mean_from_model = y_scale.inverse_transform(rfr_pred).mean()

print("Actual mean price: {}\nPredicted mean price: {}".format(mean_from_set,mean_from_model))
# Testing random forest with a listing found on Airbnb: https://www.airbnb.com/rooms/7858468?source_impression_id=p3_1592509914_6%2FQO7PfPMYw1WBTO&guests=1&adults=1

# lat, min_nights, num_reviews, reviews_per month (not listed, num_reviews / reviews_per_month)

real_listing_values = np.array([40.763168, 1, 357, 15]) 

real_listing_values = real_listing_values.reshape(1,-1)

real_price = 100



real_pred = rfr.predict(x_scale.transform(real_listing_values))

real_pred = y_scale.inverse_transform(real_pred.reshape(-1,1))



print("Real Price of Airbnb: " + str(real_price))

print("Price Predicted by Random Forest Model: " + str(real_pred[0]))
# Saving model

import pickle

import datetime



inp = input("Do you want to save?: Y/N\nRemember to change the model number before saving\n")

if inp == "Y" or inp =='y':

    filename = "Random_Forest_Model{}.pkl".format(datetime.datetime.now())

    pickle.dump(rfr, open(filename, 'wb'))



# Saving the scalers

pickle.dump(x_scale, open('x_scale.pkl', 'wb'))

pickle.dump(y_scale, open('y_scale.pkl', 'wb'))
# Splitting data for neural network

x = reg_data.drop('price', axis=1)

y = reg_data['price']



x_train, x_test, y_train, y_test = train_test_split(x,y)
x_scale_mm = MinMaxScaler()

x_scale_mm.fit(x)

y_scale_mm = MinMaxScaler()

y_scale_mm.fit(y.to_numpy().reshape(-1,1))



x_train_s = x_scale_mm.transform(x_train)

x_test_s = x_scale_mm.transform(x_test)

y_train_s = y_scale_mm.transform(y_train.to_numpy().reshape(-1,1))

y_test_s = y_scale_mm.transform(y_test.to_numpy().reshape(-1,1))



nn = Sequential()



nn.add(Flatten())

nn.add(Dense(512))

nn.add(Activation('relu'))

nn.add(Dropout(0.4))



nn.add(Dense(128))

nn.add(Activation('relu'))

nn.add(Dropout(0.4))



nn.add(Dense(1)) # Output 1 value

nn.add(Activation('linear'))



nn.compile(optimizer='adam', loss='mse', metrics=['mae'])



nn.fit(x_train_s, y_train_s, epochs=6, batch_size=16, validation_split=0.1)
val_loss, val_error = nn.evaluate(x_test_s, y_test_s, verbose=0)

print('Validation loss: {}'.format(val_loss))

print('Mean Squared Error: {}'.format(val_error))

print('RMSE: {}'.format(np.sqrt(val_error)))
test_listings = reg_data.sample(n_tests)



# Isolate same features from testing

test = test_listings[['latitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month']]

multipred = y_scale_mm.inverse_transform(nn.predict(x_scale_mm.transform(test)))



print('Predicted mean: {}'.format(multipred.mean()))

print('Sample mean: {}'.format(test_listings['price'].mean()))





bar_x1 = [i-0.5 for i in range(1,n_tests+1)]

bar_x2 = [i+0.5 for i in bar_x1]

vals = []

for space in multipred:

    for val in space:

        vals.append(val)

fig = plt.figure(figsize=(8,8))

ax = fig.subplots()

ax.bar(bar_x1, test_listings['price'], width=0.3, color='navy')

ax.bar(bar_x2, vals, width=0.3, color='r')

ax.set_xlabel("Trials")

ax.set_ylabel("Prices")

ax.set_title("Real Prices (Navy) vs Predicted Prices (Red) for Multilayer Perceptron")



fig = plt.figure(figsize=(10,10))

ax = fig.subplots()

ax.plot(range(1,n_tests+1), test_listings['price'], color='navy')

ax.plot(range(1,n_tests+1), multipred, color='red')



diff = []

count = 0

for i in test_listings['price']:

    diff.append(np.abs(i - multipred[count]))

    count+=1

mean_diff = np.mean(diff)

print('mean difference between actual and predicted price: {}'.format(mean_diff))
pred = nn.predict(x_scale_mm.transform(real_listing_values))

print('price predicted by Multilayer Perceptron model: {}'.format(y_scale_mm.inverse_transform(pred)))