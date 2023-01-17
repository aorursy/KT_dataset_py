# Predicting The Costs Of Used Cars - Random Forest



import pandas as pd

import numpy as np



dataset = pd.read_csv("/kaggle/input/used-cars-dataset-for-ml/dataset.csv")



from sklearn import preprocessing 

le=preprocessing.LabelEncoder()

dataset['Location'] = le.fit(dataset['Location']).transform(dataset['Location'])

dataset['Fuel_Type'] = le.fit(dataset['Fuel_Type']).transform(dataset['Fuel_Type'])

dataset['Transmission'] = le.fit(dataset['Transmission']).transform(dataset['Transmission'])

dataset['Owner_Type'] = le.fit(dataset['Owner_Type']).transform(dataset['Owner_Type'])

dataset['Mileage'] = dataset['Mileage'].str.replace('kmpl','').str.replace('km/kg','').str.strip().astype(float)

dataset=dataset.dropna(subset=['Mileage'])

dataset['Engine'] = dataset['Engine'].str.replace('CC','').str.strip().astype(float)

dataset=dataset.dropna(subset=['Engine'])

dataset['Power'] = dataset['Power'].str.replace('bhp','').str.replace('null','nan').str.strip().astype(float)

dataset=dataset.dropna(subset=['Power'])



# Create Company, Model & Spec

dataset['Name'] = dataset['Name'].str.replace('Land Rover','LandRover')

dataset['Name'] = dataset['Name'].str.replace('Wagon R','WagonR')

dataset['Company'] = dataset.Name.str.split().str.get(0)

dataset['Model'] = dataset.Name.str.split().str.get(1)

dataset['Spec'] = dataset['Name'].apply(lambda x: ' '.join(x.split(' ')[2:]))



dataset['Company'] = le.fit(dataset['Company']).transform(dataset['Company'])

dataset['Model'] = le.fit(dataset['Model']).transform(dataset['Model'])

dataset['Spec'] = le.fit(dataset['Spec']).transform(dataset['Spec'])



dataset['NPrice'] = dataset['Price']



dataset = dataset.drop(['Name', 'New_Price', 'Spec', 'Price'],axis=1) # Remove Unwanted Columns

dataset=dataset.dropna() # Remove Null Values





# Labels are the values we want to predict

labels = np.array(dataset['NPrice'])



# Remove the labels from the features

features= dataset.drop('NPrice', axis = 1)



# Saving feature names for later use

feature_list = list(features.columns)



# Convert to numpy array

features = np.array(features)





# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split



# Split the data into training and testing sets

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)



print('Training Features Shape:', train_features.shape)

print('Training Labels Shape:', train_labels.shape)

print('Testing Features Shape:', test_features.shape)

print('Testing Labels Shape:', test_labels.shape)





# Import the model we are using

from sklearn.ensemble import RandomForestRegressor



# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)



# Train the model on training data

rf.fit(train_features, train_labels);



# Use the forest's predict method on the test data

predictions = rf.predict(test_features)



# Calculate the absolute errors

errors = abs(predictions - test_labels)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')



# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / test_labels)



# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')


