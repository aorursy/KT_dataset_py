import pandas as pd

pd.set_option('display.max_columns', 45) # To print out more columns
iowa_file_path = '../input/home-data-for-ml-course/train.csv' # Path of the file to read

df = pd.read_csv(iowa_file_path)                              # Read the file into a variable home_data
df.describe() # Prints summary statistics
round(df["LotArea"].mean()) # Average lot size
import datetime

current_year = datetime.datetime.now().year

current_year
sold_at_latest_year = df[(df["YrSold"] == df["YrSold"].max())] # houses that are solded at latest year

sold_at_latest_year
sold_latest = sold_at_latest_year[(sold_at_latest_year["MoSold"] == sold_at_latest_year["MoSold"].max())] # houses that are solded latest

sold_latest
year_of_newest_home_age = int(sold_latest["YrSold"].iloc[0])

year_of_newest_home_age
age_of_newest_home = current_year - year_of_newest_home_age

age_of_newest_home
df.columns # Listing all the columns
df.head()
df = df[["LotFrontage", "LotArea", "OverallQual", "YearBuilt", "GarageArea", "Street", "Neighborhood", "SaleType", "SaleCondition", "SalePrice"]]

df
df.dropna(axis = 0, inplace = True) # Drops rows with missing values
""" 'Prediction Target' is what you want to predict. """

y = df.SalePrice # Getting prediction target as SalePrice

y
""" 'Features' are selected columns to make prediction. """

features = ["LotFrontage", "LotArea", "OverallQual", "YearBuilt", "GarageArea"] # Let's say we have selected these fields for making prediction.

features
X = df[features] # Our dataset to predict SalesPrice

X.head()
from sklearn.tree import DecisionTreeRegressor
home_model = DecisionTreeRegressor(random_state = 1) # Defines a model. Specifies a number for random_state to ensure same results each run.
home_model.fit(X, y) # Fitting model by providing what to predict and what are the features to make a prediction.
df.head(15) # Real result for first 15 records
home_model.predict(X.head(15)) # Making the prediction for first 15 records
home_model.predict(X) # Making the prediction for all data set