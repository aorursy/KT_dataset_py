# Hi all welcome to this introduction to machine learning and data visulisation workshop.

# I have tried to give enough explanation and references so it is simple for all to understand and enjoy.

# You can complete this exercise at your own paste and ask me any question you have.



# In this workshop we are using data related to the houses in Melbourne and we want to use machine learning 

# and build a model that predicts the house prices. But first I will take you guys through a few data visulisation

# exercises, from which you will get a feel for how to clean your data and choose the relevant data for your

# model.
# In this part we will setup and import the main libraries required for this exercise

import numpy as np # used for linear algebra and mathematical functions

import pandas as pd # used for data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Used for both plotting data and editing figures

import seaborn as sns # Another library for data visulisation

%matplotlib inline
# Import your data file



#1) Define the path to your data file

file_path = "../input/melb_data.csv"



#2) Get the data from your data file

melbourne_data = pd.read_csv(file_path)
# Look at the first five rows of your data

melbourne_data.head(5)
# Look at the columns and the data you have. 

# Have a think about what categories of data are there in the data set. Are they all numerical?

# 1) Do you see anything unusual there?

# 2) Have a play around with number of rows you see from the top

# 3) Have a play around with looking at the rows of data from the bottom of the dataset using "tail"
# Now that you had a chance to look at all the data from top and bottom of the dataset

# lets look at some statistical information about your numerical data

melbourne_data.describe()
# Have a look at these results

# 1) What differences do you see between these and the ones from using "head" and "tail"?

# 2) Have a look at the statistical values, do they all make sense? are they realistic?
# Get rid of rows with "NaN" values in them (Very simple data cleaning)

melbourne_data = melbourne_data.dropna(axis=0)



# Check to see how many rows of data have you lost

melbourne_data.describe()
# Set your target (What you want to predict)

target_price = melbourne_data.Price
# Now let's do some data visulisation to help us choose the features/data that 

# we want to include in our model later on.
#Let's start with plotting the House pirce against the year it was built



plt.figure(figsize=(8,8)) # Define size of your figure

plt.title('House price in Melbourne based on the year they were built') # Define the title for your figure

sns.scatterplot(x="YearBuilt", y= "Price", data = melbourne_data) #Plot
# What do you see in the figure? 

# Are there any odd/outlier data there? 
# Let's remove the outlier for "YearBuilt" and for "Price". 

# I will show you how to do it for "Price" and you get to do it for "YearBuilt"



# For "Price" most of the data are less than 5000000 so let's assume that for any house price above

# that are equal to 5000000

melbourne_data['Price'] = (melbourne_data['Price']).apply(lambda x: min(x,5000000))



# Now for "YearBuilt", can you only keep the houses that are built from 1900 onwards



 # -------- Your code here -----------#



#hint: Keep in mind the for the price you are removing data "above" a certain value whereas for

# year built you want to remove anything "below" 1900
# Now replot the same figure as before to see the changes



 # -------- Your code here -----------#
# Now let's use the same plotting approach which you have been using so far, i.e. scatterplot

# to look at house prices in relation to the Lattitude and Longtitude:



plt.figure(figsize=(8,8))

plt.title('House prices based on location in Melbourne')

sns.scatterplot(x='Longtitude', y= 'Lattitude',hue = 'Price',data = melbourne_data)



# In this plot we added "hue" which allows you to add a 3rd variable using colour.
# You can add another variable to the same scatter plot using "size" which changes 

# the size of points based on the variable you chose.



# Why don't you give it a try. Add "size" to your plot with another variable and see how it looks?



#--------------- Your code here --------------#
# Another way you can visulise your data is through "pairplot" where you can plot multiple variables against

# eachother to get an over view of your data and their relation to eachother.

# For this demo I have chosen these four variables but feel free to add or remove any of them just to get a 

# feel for it. 

sns.pairplot(melbourne_data, vars =["Price", "Rooms", "Bathroom", "Landsize"], kind="reg")



# Here you can choose the kind of plot you want by using the "kind" option. 

# Have a look at seaborn pairplot documentation for more info.
# If you want spend some time plotting different variables against each other to see which ones 

# affect the house price more, feel free to do so, if now please move to the next step. :)

# Also to try out different data visulisation tools please refer to 

# https://seaborn.pydata.org/index.html



# The code below is something extra I plotted as an example, if you want to have a look.

#===============================

# plt.figure(figsize=(16,6))

# plt.title('Distribution of house prices with number of bathrooms in the house')

# sns.swarmplot(x=melbourne_data['Bathroom'], y = melbourne_data['Price'])
# This will bring us to the end of the introduction to data visulisation using Python. 

# Now let's do some machine learning!!!
# As a first step let's convert your dataset into collection of columns so later you can

# carry out all the necessary analysis

melbourne_data.columns 
# Looking at the column names and the data visulisation excercise before

# let's choose the name of features (column names/variables) that  

# you think influences your prediction. 

# I have chosen the following ones but feel free to play around with different variables later on

feature_names =['Rooms', 'Bathroom','Landsize','Lattitude','Longtitude']



# Put the columns of data with your chosen name into a Dataframe (matrix)

features = melbourne_data[feature_names]
# Using "describe" have a look at your feature Dataframe to check that you got the right dataframe
# Given that in most real life cases we don't have separate testing data available, the most common practice

# is to divide the data we have into training and testing/validation data.

# To do that we need to import "train_test_split"; 

# this functionality will automatically divide your data into training and test data

from sklearn.model_selection import train_test_split 



# Now let's define your training and test data sets.

training_features,testing_features,training_target,testing_target = train_test_split(features,target_price, test_size=0.2,random_state=0)



# "Test_size" option is a number between 0-1 and if not defined it will be 0.25. 

# To find out more about this functionality have a look at 

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# Choose a model from "scikit-klearn" library, which in short is written as

# "sklearn". This library provides simple and efficient tools for data mining

# and data analysis

# For this example let's choose decision tree as our machine learning model

# (To learn more about decision tree have a look at this link 

# https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb)

from sklearn.tree import DecisionTreeRegressor



# Setup the model you chose to use and 

# choose a number for random_state so you ensure same results for each run.

# There are more parameters to set with this model but for this example we keep it simple.

# Look up the documentation for more information on that.

melbourne_house_price_model = DecisionTreeRegressor(random_state = 1)



# "random_stat" option: If int, random_state is the seed used by the random number generator;

# If RandomState instance, random_state is the random number generator;

# If None, the random number generator is the RandomState instance used by np.random.

# The next step is to train your model with your training data

# Here they call it "fit" but it is the same as training. This is the heart of modelling

melbourne_house_price_model.fit(training_features,training_target) 
# Now let's make some predictions of house prices in Melbourne!!!!

melbourne_house_price_prediction = melbourne_house_price_model.predict(testing_features)
# Let's see how well we have done by comparing our predictions with what they should be 

# 1) You can plot your predictions vs the actual value to see how well you did.

# For an ideal case you expect to see your result be close to the line x=y.



# Do you wnat to give it a try yourself before going to my solution? 


# I have added the orane line which represents the ideal prediction results for comparison

x = [0,100000, 1000000,5000000]

y = x



# Plotting our data

plt.figure(figsize =(10,6))

plt.title('Comparison between our prediction and the real house prices')



sns.regplot(x=melbourne_house_price_prediction, y= testing_target, label='Our result')

sns.lineplot(x= x, y=y, label='Ideal case')
# 2) Computing an error indicator such as Mean Absolut Error of MAE.

# To read more about MAE vivist: 

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html



# Import MAE from sklearn

from sklearn.metrics import mean_absolute_error



# Compute the MAE for your prediction

mean_absolute_error(testing_target,melbourne_house_price_prediction)
# Are you hapyy with your results? Do you wonder if there are ways of improving our model?
# Yes!! There are multiple ways of improving the accuracy of our model. 

# Here we only talk about two of them



# 1) First approach makes a use of "max_leaf_node" option of DecisionTreeRegressor model.

# This parameter allows you to avoid overfitting and under fitting. 

# To read more about this please have a look at 

# https://www.kaggle.com/dansbecker/underfitting-and-overfitting 
# In this part of the example I will define a function/method which contains all the steps from 

# the model, training to testing and computing the MAE that we just went through.

# This way we can easily see the effect of "max_leaf_node" on the accuracy of our model

def get_mae(max_leaf_nodes,training_features,testing_features,training_target,testing_target):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 0)

    model.fit(training_features,training_target)

    predicted_testing_target= model.predict(testing_features)

    mae = mean_absolute_error(testing_target,predicted_testing_target)

    return(mae)
# Feel free to use the above function to change the value of "max_leaf_nodes" and see it's

# effect. 



print(get_mae(200,training_features, testing_features,training_target, testing_target))
# One quick way of doing this is to choose a range of numbers for "max_leaf_node" 

# and run a for-loop to get a list of MAEs, as you can see below

for max_leaf_nodes in [5,50,100,500,5000,900]:

    mae = get_mae(max_leaf_nodes,training_features, testing_features,training_target, testing_target)

    print("Max leaf nod: %d \t\t Mean Absolut Error: %d" %(max_leaf_nodes,mae))
# 2) The second approach for imporving your predictions is to use random forest 

# instead of single tree



# Import the RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor



# Setup your RandomForestRegressor model with the parameters from

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# In the following example n_estimators defines how many trees to use in your forest

random_forest_model = RandomForestRegressor(n_estimators= 100,random_state=1)



# Train your model with your training data



 #------------ Your code -------------#



# Make predictions with your new trained model



 #------------ Your code -------------#



# Compute and show the MAE from your RandomForestRegressor model



 #------------ Your code -------------#

# Finally a challenge for you:

# Can you investigate and who the effect of "max_leaf_nodes" on the

# accuracy of your RandomForestRegressor model?
# I hope you learnt something new and enjoyed this workshop