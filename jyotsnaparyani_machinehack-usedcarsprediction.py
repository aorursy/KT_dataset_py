# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

training_set = pd.read_excel("../input/Data_Train.xlsx")

test_set = pd.read_excel("../input/Data_Test.xlsx")
training_set.head()
#checking the number of features in the Datasets

print("Number of features in the datasets :")

print("Training Set : ", len(training_set.columns))

print("Test Set : ",len(test_set.columns))
#List of feature column in the Datasets

print("Number of features in the datasets :")

print("Training Set : ", list(training_set.columns))

print("Test Set : ",list(test_set.columns))
#checking the data types of features

print("Datatypes of features in the datasets :")

print("Training Set : \n", training_set.dtypes)

print("Test Set : \n",test_set.dtypes)
#checking the number of rows in the Datasets

print("Number of rows in the datasets :")

print("Training Set : ", len(training_set))

print("Test Set : ",len(test_set))
#checking for NaNs or empty cells

print("\n\nEmpty cells or Nans in the datasets :")

print("\nTraining Set : \n",training_set.isnull().values.any())

print("\nTest Set : \n",test_set.isnull().values.any())
#checking for NaNs or empty cells by features

print("\n\nNumber of empty cells or Nans in the datasets :")

print("\nTraining Set : \n", training_set.isnull().sum())

print("\nTest Set : \n",test_set.isnull().sum())
#Exploring Categorical variables

#combining training set and test set data

all_brands = list(training_set.Name) + list(test_set.Name)

all_locations = list(training_set.Location) + list(test_set.Location)

all_fuel_types = list(training_set.Fuel_Type) + list(test_set.Fuel_Type)

all_transmissions = list(training_set.Transmission) + list(test_set.Transmission)

all_owner_types = list(training_set.Owner_Type) + list(test_set.Owner_Type)
print("\nNumber Of Unique Values In Name : \n ", len(set(all_brands)))



print("\nNumber Of Unique Values In Location : \n ", len(set(all_locations)))

print("\nThe Unique Values In Location : \n ", set(all_locations) )



print("\nNumber Of Unique Values In Fuel_Type : \n ", len(set(all_fuel_types)))

print("\nThe Unique Values In Fuel_Type : \n ", set(all_fuel_types) )



print("\nNumber Of Unique Values In Transmission : \n ", len(set(all_transmissions)))

print("\nThe Unique Values In Transmission : \n ", set(all_transmissions) )



print("\nNumber Of Unique Values In Owner_Type : \n ", len(set(all_owner_types)))

print("\nThe Unique Values In Owner_Type : \n " ,set(all_owner_types) )
#DataCleaning

#1.) Feature column : Name - Separating the column as Brand and Model

#Training Set

names = list(training_set.Name)

brand = []

model = []

for i in range(len(names)):

   try:

       brand.append(names[i].split(" ")[0].strip())

       try:

           model.append(" ".join(names[i].split(" ")[1:]).strip())

       except:

           pass

   except:

       print("ERR ! - ", names[i], "@" , i)

training_set["Brand"] =  brand

training_set["Model"] = model

training_set.drop(labels = ['Name'], axis = 1, inplace = True)
#Test Set data cleaning for Name Column

names = list(test_set.Name)

brand = []

model = []

for i in range(len(names)):

   try:

       brand.append(names[i].split(" ")[0].strip())

       try:

           model.append(" ".join(names[i].split(" ")[1:]).strip())

       except:

           pass

   except:

       print("ERR ! - ", names[i], "@" , i)

test_set["Brand"] =  brand

test_set["Model"] = model

test_set.drop(labels = ['Name'], axis = 1, inplace = True)
#Feature Column : Mileage

#""" Removing the  mileage units"""



# Training Set

training_set.Mileage = training_set.Mileage.fillna(0.0)

mileage = list(training_set.Mileage)

for i in range(len(mileage)):

    mileage[i] = str(mileage[i]).split(" ")[0].strip()

training_set['Mileage'] = mileage

training_set['Mileage'] = training_set['Mileage'].astype(float)
training_set.head()
#Feature Column : Mileage

#""" Removing the  mileage units"""



# Testing Set

test_set.Mileage = test_set.Mileage.fillna(0.0)

mileage = list(test_set.Mileage)

for i in range(len(mileage)):

    mileage[i] = str(mileage[i]).split(" ")[0].strip()

test_set['Mileage'] = mileage

test_set['Mileage'] = test_set['Mileage'].astype(float)



test_set.head()
#Feature Column : Engine

#""" Removing the  Engine units"""



# Training Set

training_set.Engine = training_set.Engine.fillna(0)

engine = list(training_set.Engine)

for i in range(len(engine)):

    engine[i] = str(engine[i]).split(" ")[0].strip()

training_set['Engine'] = engine

training_set['Engine'] = training_set['Engine'].astype(int)

training_set.head()
#Feature Column : Engine

#""" Removing the  Engine units"""



# Training Set

test_set.Engine = test_set.Engine.fillna(0)

engine = list(test_set.Engine)

for i in range(len(engine)):

    engine[i] = str(engine[i]).split(" ")[0].strip()

test_set['Engine'] = engine

test_set['Engine'] = test_set['Engine'].astype(int)
training_set.head()
#Feature Column : Power

#""" Removing the  Power units"""



# Training Set

training_set.Power = training_set.Power.fillna(0.0)

power = list(training_set.Power)

for i in range(len(power)):

    power[i] = str(power[i]).split(" ")[0].strip()

training_set['Power'] = power

training_set['Power'] = training_set['Power'].str.replace("null", "0.0", case = False) 
training_set['Power'] = training_set['Power'].astype(float)
training_set.head()
training_set.dtypes
#Feature Column : Power

#""" Removing the  Power units"""



# Test Set

test_set.Power = test_set.Power.fillna(0.0)

power = list(test_set.Power)

for i in range(len(power)):

    power[i] = str(power[i]).split(" ")[0].strip()

test_set['Power'] = power
test_set['Power'] = test_set['Power'].str.replace("null", "0.0", case = False) 

test_set['Power'] = test_set['Power'].astype(float)
test_set.dtypes
training_set.head()
#Dropping Feature/Column : New_Price as it has many NaN

training_set.drop(labels = ['New_Price'], axis = 1, inplace = True)

test_set.drop(labels = ['New_Price'], axis = 1, inplace = True)
training_set.head()

# Univariate visualisation for quantative features

features = ['Mileage', 'Engine','Power','Seats']

training_set[features].hist(figsize=(15, 8));
import seaborn as sns

import matplotlib.pyplot as plt



#multivariate visualisation for 2 vaariables Location and mileage

plt.figure(figsize=(15,16))

sns.boxplot(x = 'Location', y = 'Mileage', data = training_set) 
sns.boxplot(x = 'Fuel_Type', y = 'Power', data = training_set) 
sns.boxplot(x = 'Transmission', y = 'Mileage', data = training_set) 
training_set.head()
#Categorical vairiables visualizations

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 6))

sns.countplot(x='Fuel_Type', data=training_set, ax=axes[0]);

sns.countplot(x='Location', data=training_set, ax=axes[1]);
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

sns.countplot(x='Year', data=training_set, ax=axes[0]);

sns.countplot(x='Transmission', data=training_set, ax=axes[1]);
#correlation matrix for numerical variables

# Drop non-numerical variables

numerical = list(set(training_set.columns) - 

                 set(['Location', 'Year', 'Fuel_Type', 

                      'Transmission', 'Owner_Type', 'Brand','Model']))



# Calculate and plot with annotated correlation

corr_matrix = training_set[numerical].corr()

plt.figure(figsize=(6,6))

sns.heatmap(corr_matrix, vmin=-1,

            cmap='coolwarm',

            annot=True);

#multivariate visualisation like scatterplot for 2 variables

sns.jointplot(x='Engine', y='Price', 

              data=training_set, kind='scatter');
#multivariate visualisation

sns.jointplot(x='Engine', y='Price', 

              data=training_set, kind='scatter');
#multivariate visualisation

sns.jointplot(x='Engine', y='Power', 

              data=training_set, kind='scatter');
#multivariate visualisation

sns.jointplot(x='Engine', y='Mileage', 

              data=training_set, kind='scatter');
training_set.head()
#Quantitative vs. Categorical multivariate visualisation

sns.lmplot('Engine', 'Mileage', data=training_set, hue='Transmission', fit_reg=False);
#Categorical vs. Categorical

_, axes = plt.subplots(1, 2, sharey=True, figsize=(24, 8))



sns.countplot(x='Fuel_Type', hue='Transmission', data=training_set, ax=axes[0]);

sns.countplot(x='Location', hue='Transmission', data=training_set, ax=axes[1]);
#Contingency table : also called a cross tabulation. 

#It shows a multivariate frequency distribution of categorical variables in tabular form. 

pd.crosstab(training_set['Brand'], training_set['Transmission']).T
#'Brand', 'Model', 'Location','Fuel_Type', 'Transmission', 'Owner_Type'



all_brands = list(set(list(training_set.Brand) + list(test_set.Brand)))

all_models = list(set(list(training_set.Model) + list(test_set.Model)))

all_locations = list(set(list(training_set.Location) + list(test_set.Location)))

all_fuel_types = list(set(list(training_set.Fuel_Type) + list(test_set.Fuel_Type)))

all_transmissions = list(set(list(training_set.Transmission) + list(test_set.Transmission)))

all_owner_types = list(set(list(training_set.Owner_Type) + list(test_set.Owner_Type)))
#Initializing label encoders

from sklearn.preprocessing import LabelEncoder

le_brands = LabelEncoder()

le_models = LabelEncoder()

le_locations = LabelEncoder()

le_fuel_types = LabelEncoder()

le_transmissions = LabelEncoder()

le_owner_types = LabelEncoder()
#Fitting the categories

le_brands.fit(all_brands)

le_models.fit(all_models)

le_locations.fit(all_locations)

le_fuel_types.fit(all_fuel_types)

le_transmissions.fit(all_transmissions)

le_owner_types.fit(all_owner_types)
#Applying encoding to Training_set data

training_set['Brand'] = le_brands.transform(training_set['Brand'])

training_set['Model'] = le_models.transform(training_set['Model'])

training_set['Location'] = le_locations.transform(training_set['Location'])

training_set['Fuel_Type'] = le_fuel_types.transform(training_set['Fuel_Type'])

training_set['Transmission'] = le_transmissions.transform(training_set['Transmission'])

training_set['Owner_Type'] = le_owner_types.transform(training_set['Owner_Type'])
#Applying encoding to Test_set data

test_set['Brand'] = le_brands.transform(test_set['Brand'])

test_set['Model'] = le_models.transform(test_set['Model'])

test_set['Location'] = le_locations.transform(test_set['Location'])

test_set['Fuel_Type'] = le_fuel_types.transform(test_set['Fuel_Type'])

test_set['Transmission'] = le_transmissions.transform(test_set['Transmission'])

test_set['Owner_Type'] = le_owner_types.transform(test_set['Owner_Type'])
training_set.head()
#Re-ordering the columns

training_set = training_set[['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',

      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']]

test_set = test_set[['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',

      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']]
training_set.head()
# Dependent Variable

Y_train_data = training_set.iloc[:, -1].values



# Independent Variables

X_train_data = training_set.iloc[:,0 : -1].values



# Independent Variables for test Set

X_test = test_set.iloc[:,:].values
from sklearn.impute import SimpleImputer

import numpy as np



#Handling missing values

#Training Set Imputation

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer = imputer.fit(X_train_data[:,8:12]) 

X_train_data[:,8:12] = imputer.transform(X_train_data[:,8:12])



#Test_set Imputation

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

imputer = imputer.fit(X_test[:,8:12]) 

X_test[:,8:12] = imputer.transform(X_test[:,8:12])
from sklearn.model_selection import train_test_split



#Splitting the training set into Training and validation sets

X_train, X_val, Y_train, Y_val = train_test_split(X_train_data, Y_train_data, test_size = 0.2, random_state = 1)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



#Scaling Original Training Data

X_train_data = sc.fit_transform(X_train_data)
#Reshaping vector to array for transforming

Y_train_data = Y_train_data.reshape((len(Y_train_data), 1))
Y_train_data = sc.fit_transform(Y_train_data)

#converting back to vector

Y_train_data = Y_train_data.ravel()



X_test = sc.transform(X_test)



# Scaling Splitted training and val sets

X_train = sc.fit_transform(X_train)

X_val = sc.fit_transform(X_val)



#Reshaping vector to array for transforming

Y_train = Y_train.reshape((len(Y_train), 1)) 

Y_train = sc.fit_transform(Y_train)

#converting back to vector

Y_train = Y_train.ravel()
#Modelling And Predicting

#Calculating Accuracy With RMLSE

#We will use the Root Mean Log Squared Error (RMLSE) on the validation set for calculating 

#the accuracy as mentioned in the hackathons evaluation page.

def score(y_pred, y_true):

   error = np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean() ** 0.5

   score = 1 - error

   return score





y_true = Y_val
#Initializing Linear regressor

from sklearn.linear_model import LinearRegression

lr = LinearRegression()



#Fitting the regressor with training data

lr.fit(X_train,Y_train)



#Predicting the target(Price) for predictors in validation set X_val

Y_pred = sc.inverse_transform(lr.predict(X_val))



#Eliminating negative values in prediction for score calculation

for i in range(len(Y_pred)):

   if Y_pred[i] < 0:

       Y_pred[i] = 0



#Printing the score for validation sets

print("Linear Regression SCORE : ", score(Y_pred, y_true))



#Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators = 100)

rf.fit(X_train,Y_train)

Y_pred = sc.inverse_transform(rf.predict(X_val))



#Eliminating negative values in prediction for score calculation

for i in range(len(Y_pred)):

   if Y_pred[i] < 0:

       Y_pred[i] = 0

        

print("Random Forest classifier score : ",score(Y_pred,y_true))
#Initializing a new regressor

rf2 = RandomForestRegressor(n_estimators = 100)



#Fitting the regressor with complete training data(X_train_data,Y_train_data)

rf2.fit(X_train_data,Y_train_data)



#Predicting the target(Price) for predictors in the test data

Y_pred2 = sc.inverse_transform(rf2.predict(X_test))



#Eliminating negative values in prediction for score calculation

for i in range(len(Y_pred2)):

   if Y_pred2[i] < 0:

       Y_pred2[i] = 0



#Saving the predictions to an excel sheet

pd.DataFrame(Y_pred2, columns = ['Price']).to_excel("predictions.xlsx")