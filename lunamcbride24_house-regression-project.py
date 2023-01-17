# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split #Split the data into train and test

from sklearn.linear_model import LinearRegression #Add in our linear regression

from sklearn.preprocessing import StandardScaler #Test out scaling

from sklearn.neural_network import MLPRegressor #Add a multilayer perceptron to test regression ability

from sklearn import svm #Add a support vector machine to test regression ability

from sklearn.tree import DecisionTreeRegressor #Add a single tree regressor to test regression ability

from sklearn.ensemble import RandomForestRegressor #Add a forest regressor to test regression ability

from sklearn.ensemble import ExtraTreesRegressor #Add even more trees to test regression ability



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
house = pd.read_csv("../input/kc-housesales-data/kc_house_data.csv") #Read in the houses dataset

house.head() #Take a peek at the dataset
print(house.count())

print(house.isnull().any())
house["age"] = pd.DatetimeIndex(house['date']).year - house["yr_built"] #Get the age of the building

print(house['date'][1], " - ", house["yr_built"][1], " = ", house["age"][1]) #Print the equation as a sanity check
house.drop(columns = ["yr_renovated"], inplace = True) #Drop the year renovated field

house.head() #Take a peek at the dataset
house.drop(columns = ["id", "date"], inplace = True) #Drop the ID and Date Fields

house.head() #Take a peek at the dataset
house.drop(columns = ["sqft_living", "sqft_lot"], inplace = True) #Drop the old sqft lot and living Fields

house.head() #Take a peek at the dataset
price = np.array(house["price"].copy().astype(int)) #Set price to be the prices of the houses

price = np.log(price) #Log the price

print(price) #Print the prices
characteristics = house.drop("price", axis = 1) #Get every other feature of our dataframe except price

chara = pd.get_dummies(characteristics) #Get the dummies for easier model training

scale = StandardScaler() #Add a standard scaler to scale our data for easier use later

scale.fit(chara) #Fit the scaler with our characteristics

chara = scale.transform(chara) #Transform the data with our scaler



print(chara) #Print the scaled data
charaTrain, charaTest, priceTrain, priceTest = train_test_split(chara, price, test_size = 0.3) #Split the data into train and test

print(priceTest) #Print one of the price splits
regression = LinearRegression() #Open a linear regression model

regression.fit(charaTrain, priceTrain) #Fit the regression model
print(regression.score(charaTest, priceTest)) #Print the accuracy of the model

print(regression.coef_) #Print the model coefficients
neural = MLPRegressor(hidden_layer_sizes = (3,100), random_state=1, max_iter=500) #Build a neural network to test regression

neural.fit(charaTrain, priceTrain) #Fit the network with the train set
svr =  svm.SVR() #Get a support vector regressor to test ability

svr.fit(charaTrain, priceTrain) #Fit the regresso
tree = DecisionTreeRegressor() #Build a tree

tree.fit(charaTrain, priceTrain) #Fit the tree
forest = RandomForestRegressor() #Build a whole forest of trees

forest.fit(charaTrain, priceTrain) #Fit the forest
forestBig = ExtraTreesRegressor() #Build a more random forest

forestBig.fit(charaTrain, priceTrain) #Fit the more random forest
#Print the accuracies of all the models

print("Linear Regression Accuracy: ", regression.score(charaTest, priceTest))

print("Neural Network Accuracy: ", neural.score(charaTest, priceTest))

print("Support Vector Accuracy: ", svr.score(charaTest, priceTest))

print("Single Tree Accuracy: ", tree.score(charaTest, priceTest))

print("Random Forest Accuracy: ",forest.score(charaTest, priceTest))

print("Even more Random Forest Accuracy: ", forestBig.score(charaTest, priceTest))
attributes = characteristics.columns #Get the tested attributes

attributes = list(zip(attributes, regression.coef_)) #Zip the attributes together with their coefficient

sortAtt = sorted(attributes, key = lambda x: x[1], reverse = True) #Sort the zipped attributes by their coefficients



print("According to the Linear Regression, the most important factors for pricing are: ") #Start printing the most important labels

i=0 #Counter variable so only the top five are printed



#For each attribute in the sorted attributes

for label, coef in sortAtt:

    if i<5: #If there has not been five printed yet

        print(label) #Print the label as an important factor

    i += 1 #Increase i by 1
attributes = characteristics.columns #Get the tested attributes

attributes = list(zip(attributes, forest.feature_importances_)) #Zip the attributes together with their coefficient

sortAtt = sorted(attributes, key = lambda x: x[1], reverse = True) #Sort the zipped attributes by their coefficients



print("According to the Random Forest (most accurate), the most important factors for pricing are: ") #Start printing the most important labels

i=0 #Counter variable so only the top five are printed



#For each attribute in the sorted attributes

for label, coef in sortAtt:

    if i<5: #If there has not been five printed yet

        print(label) #Print the label as an important factor

    i += 1 #Increase i by 1