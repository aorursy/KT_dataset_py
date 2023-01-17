# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.tree import DecisionTreeRegressor, export_graphviz

from sklearn.model_selection import cross_validate

from sklearn.utils import shuffle

from sklearn.preprocessing import OrdinalEncoder



import seaborn as sns

import graphviz 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Creating method to see data correlation using spearman. 

# I prefer to create a method because I do this multiple times.





def correlation_plot(dataframe, method_='spearman'):

    correlations = dataframe.corr(method=method_)



    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',

                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})

    plt.show();
# Set datasets path



VALIDATION_SET_PATH = "../input/valid.csv"

EXAMPLE_SET_PATH = "../input/exemplo.csv"

TRAIN_SET_PATH = "../input/train.csv"

TEST_SET_PATH = "../input/test.csv"
# Load data



validation_set_data = pd.read_csv(VALIDATION_SET_PATH, parse_dates=["sale_date"])

example_set_data = pd.read_csv(EXAMPLE_SET_PATH)

train_set_data = pd.read_csv(TRAIN_SET_PATH, parse_dates=["sale_date"])

test_set_data = pd.read_csv(TEST_SET_PATH, parse_dates=["sale_date"])
# Merge data



all_data = pd.concat([train_set_data, validation_set_data, test_set_data], sort=False)
# Checking columns



all_data.columns
# Checking columns types



all_data.dtypes
# Checking data correlation



correlation_plot(all_data)
# Removing redundant/useless columns



all_data = all_data.drop(columns=["year_built", "total_units"])
# Encoding Date Types



def parse_date(dataframe):

    copy = dataframe

    

    copy["sale_month"] = copy.sale_date.dt.month

    copy["sale_year"] = copy.sale_date.dt.year

    

    copy = copy.drop(columns=["sale_date"])

    

    return copy



all_data = parse_date(all_data)
# Checking data correlation



correlation_plot(all_data)
# Removing redundant/useless columns



all_data = all_data.drop(columns=["sale_year", "sale_month"])
# Correcting bad inputs



empty = all_data.land_square_feet.head(1)[0]

mask = all_data.land_square_feet != (empty)

column_name = 'land_square_feet'

non_empty_columns = all_data.loc[mask, column_name]

non_empty_columns = non_empty_columns.astype(float) # We need to convert, so we dont get inf



mean = non_empty_columns.mean()



mask = all_data.land_square_feet == (empty)

all_data.loc[mask, column_name] = mean



all_data.land_square_feet = all_data.land_square_feet.astype(float)





empty = all_data.gross_square_feet.head(1)[0]

mask = all_data.gross_square_feet != (empty)

column_name = 'gross_square_feet'

non_empty_columns = all_data.loc[mask, column_name]



non_empty_columns = non_empty_columns.astype(float) # We need to convert, so we dont get inf

mean = non_empty_columns.mean()



mask = all_data.gross_square_feet == (empty)

all_data.loc[mask, column_name] = mean



all_data.gross_square_feet = all_data.gross_square_feet.astype(float)



all_data = all_data.drop(["ease-ment", "apartment_number", "address", "building_class_at_present", "tax_class_at_present"], axis=1)
# Checking data correlation



correlation_plot(all_data)
# One hot encoding columns



all_data = all_data.drop(columns=[ "building_class_at_time_of_sale"], axis=1)

all_data = pd.get_dummies(all_data, columns=["neighborhood", "building_class_category"], drop_first=True)
# Split Dataframes



train_set_data = shuffle(all_data.iloc[:38170])

validation_set_data = all_data.iloc[38170:47842]

test_set_data = all_data.iloc[47842:]
# Dividing X and Y



x_train, y_train = train_set_data.drop(columns=["sale_id","sale_price"]), train_set_data.sale_price



# Train model



model = DecisionTreeRegressor(max_leaf_nodes=100)

model.fit(x_train, y_train)
# Check model score



cross_validate(model, x_train, y_train, scoring='r2', cv=5, return_train_score=True)
# Creating new dataset with both validation and test data.

# Creating X for prediction (note: the validation set DOES NOT have the column sale_price)



test_and_validation_set = pd.concat([test_set_data, validation_set_data], sort=False)

x_test = test_and_validation_set.drop(columns=["sale_id", "sale_price"])
# Use model to predict the test/validation set



predicted = model.predict(x_test)
# Create new dataframe with the predicted values and concatenate with the example values



data = {'sale_id': test_and_validation_set.sale_id, 'sale_price': predicted}



output_dataframe = pd.DataFrame(data)



output_dataframe.to_csv("output.csv", encoding="utf8", index=False)
