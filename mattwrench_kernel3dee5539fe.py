# Project imports

import pandas

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import sklearn

import sklearn.model_selection

import sklearn.linear_model
# Load data

data = pandas.read_csv('../input/seattle/listings.csv')
# Reduce data to only the relevant columns

print(data.columns)

all_columns = ['property_type', 'room_type', 'zipcode', 

           'bedrooms', 'beds', 'bathrooms', 'accommodates', 'guests_included', 'extra_people', 

           'review_scores_rating', 'host_response_rate', 

           'security_deposit', 'cleaning_fee','price']

reduced_data = data[all_columns]

print(reduced_data)



# See number of NaN values per column

for c in reduced_data.columns:

    print(c)

    print(reduced_data[c].value_counts(dropna = False))
# View all unique values for each column

for c in reduced_data.columns:

    print(c)

    print(reduced_data[c].unique())
# Clean the data

cleaned_data = reduced_data

print(cleaned_data.shape)

# Property type: Reduce to only apartment or house

cleaned_data = cleaned_data[(cleaned_data.property_type == 'Apartment') | (cleaned_data.property_type == 'House')]



# Zipcode: Remove mistyped values ('99\n98122')

cleaned_data = cleaned_data[cleaned_data.zipcode != '99\n98122'] 



# Remove dollar and percentage signs from data

for c in reduced_data.columns:

    cleaned_data[c] = cleaned_data[c].replace('[\$,]', '', regex=True)

    cleaned_data[c] = cleaned_data[c].replace('[\%,]', '', regex=True)

    

# Security deposity & cleaning fee: Convert NaN to 0

cleaned_data['security_deposit'].fillna(value=0)

cleaned_data['cleaning_fee'].fillna(value=0)

    

# Remove all data w/ NaN values

saved_data = cleaned_data # Dropping all rows w/ NaN loses a lot of data; save this data so we can reuse some of these rows once the 

cleaned_data = cleaned_data.dropna()



# View all unique values again

for c in cleaned_data.columns:

    print(c)

    print(cleaned_data[c].unique())

print(cleaned_data.shape)
# Determine and display Pearson correlation coefficients for numeric data

numeric_columns = ['bedrooms', 'beds', 'bathrooms', 'accommodates', 'guests_included', 'extra_people', 

           'review_scores_rating', 'host_response_rate', 

           'security_deposit', 'cleaning_fee','price']



# Convert string values to float values for each numeric column

for c in numeric_columns:

    cleaned_data[c] = cleaned_data[c].astype(float)

    

correlations = np.corrcoef(cleaned_data[numeric_columns].values.T)



# Create heatmap

fig, ax = plt.subplots(figsize=(16,8))

im = ax.imshow(correlations, cmap='YlOrRd')



# Set axes

ax.set_xticks(np.arange(len(numeric_columns)))

ax.set_yticks(np.arange(len(numeric_columns)))

ax.set_xticklabels(numeric_columns)

ax.set_yticklabels(numeric_columns)

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor") # Rotate x-axis



# Loop over data dimensions and create text annotations.

for i in range(len(numeric_columns)):

    for j in range(len(numeric_columns)):

        text = ax.text(j, i, '{:.2f}'.format(correlations[i, j]),ha="center", va="center", color="black")



ax.set_title("Correlation Coefficients")

plt.show()
# Use only the selected features

# Based on correlation coefficients and relevant categorical data

selected_columns = ['property_type', 'room_type', 'zipcode', 'bedrooms', 'beds', 'price']

selected_data = saved_data[selected_columns]

# Re-drop NaN values

selected_data = selected_data.dropna()

# Re-convert string values to float values for each numeric column

numeric_columns = ['bedrooms', 'beds', 'price']

for c in numeric_columns:

    selected_data[c] = selected_data[c].astype(float)

print(selected_data.shape)
# Explore the data visually

# Display bar chart for property_type

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



types = selected_data.property_type.unique()



# Calculate arithmetic mean for each property type

averages = []

for type in types:

    df = selected_data[selected_data.property_type == type]

    prices = df.price

    averages.append(np.mean(prices))

    

ax.bar(types, averages)

ax.set_title('Average Price by Property Type')

ax.set_xlabel('Property Type')

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor") # Rotate x-axis

ax.set_ylabel('Price ($)')

plt.show()
# Display bar chart for room_type

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])



types = selected_data.room_type.unique()



# Calculate arithmetic mean for each room type

averages = []

for type in types:

    df = selected_data[selected_data.room_type == type]

    prices = df.price

    averages.append(np.mean(prices))

    

ax.bar(types, averages)

ax.set_title('Average Price by Room Type')

ax.set_xlabel('Room Type')

ax.set_ylabel('Price ($)')

plt.show()
# Display scatter plot for bedrooms

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.scatter(selected_data['bedrooms'], selected_data['price'], color='r')

ax.set_xlabel('# of Bedrooms')

ax.set_ylabel('Price ($)')

ax.set_title('Price by # of Bedrooms')

plt.show()
# Display scatter plot for beds

fig=plt.figure()

ax=fig.add_axes([0,0,1,1])

ax.scatter(selected_data['beds'], selected_data['price'], color='b')

ax.set_xlabel('# of Beds')

ax.set_ylabel('Price ($)')

ax.set_title('Price by # of Beds')

plt.show()
# Display bar chart of average price per zipcode

fig = plt.figure(figsize=(16,8))



ax = fig.add_axes([0,0,1,1])

zipcodes = selected_data.zipcode.unique()



# Calculate arithmetic mean for each zipcode

averages = []

for zip in zipcodes:

    df = selected_data[selected_data.zipcode == zip]

    prices = df.price

    averages.append(np.mean(prices))

    

# Order by price

combined = pandas.DataFrame({

    'zipcode': zipcodes,

    'average': averages

})

combined = combined.sort_values(by=['average'])



ax.bar(combined['zipcode'], combined['average'])

ax.set_title('Average Price by ZIP Code')

ax.set_xlabel('ZIP Code')

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor") # Rotate x-axis

ax.set_ylabel('Price ($)')

plt.show()
# Display listing count histograms for each selected feature

def render_hist(column, name):

    plt.hist(selected_data[column])

    plt.xlabel(name)

    plt.ylabel('# of Listings')

    plt.title('# of Listings by {}'.format(name))

    plt.show()



render_hist('property_type', 'Property Type')

render_hist('room_type', 'Room Type')

#render_hist('zipcode', 'ZIP Code')

render_hist('bedrooms', 'Beds')

render_hist('beds', 'Bedrooms')



# Render ZIPs as bar chart

fig = plt.figure(figsize=(16,8))



ax = fig.add_axes([0,0,1,1])

zipcodes = selected_data.zipcode.unique()



# Calculate arithmetic mean for each zipcode

counts = []

for zip in zipcodes:

    counts.append(selected_data[selected_data.zipcode == zip].count())

    

# Order by price

combined = pandas.DataFrame({

    'zipcode': zipcodes,

    'average': averages

})

combined = combined.sort_values(by=['average'])



ax.bar(combined['zipcode'], combined['average'])

ax.set_title('# of Listings by ZIP Code')

ax.set_xlabel('ZIP Code')

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",rotation_mode="anchor") # Rotate x-axis

ax.set_ylabel('# of Listings')

plt.show()
# Convert categorical data into seperate numeric columns

# Necessary in order to perform multiple linear regression

def break_up_categories(original_column_name):

    categories = selected_data[original_column_name].unique()

    for c in categories:

        selected_data[c] = np.where(selected_data[original_column_name] == c, 1, 0)

    del selected_data[original_column_name] # Remove original categorical column once done

    

break_up_categories('zipcode')

break_up_categories('property_type')

break_up_categories('room_type')

print(selected_data)

print(selected_data.shape)
# Separate training and testing data

X = selected_data

X = X.drop(columns = 'price')

y = selected_data[['price']]



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = .20)
# Perform multiple linear regression on training data

regr = sklearn.linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Display the coefficients and b value

for i in range(regr.coef_[0].size):

    #print(X_train.columns[i], end="")

    #print(X_train.columns[i])



    #print(":", end="")

    print(regr.coef_[0][i])

print (regr.intercept_)
# Make predictions using the testing set

y_pred = regr.predict(X_test)



# Evaluate prediction effectiveness with R^2

print('Coefficient of determination: {:2f}'.format(sklearn.metrics.r2_score(y_test, y_pred)))
