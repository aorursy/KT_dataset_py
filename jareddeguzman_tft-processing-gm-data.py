import numpy as np 

import pandas as pd

import json

import ast



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tft_data_filepath = '../input/tft-match-data/TFT_GrandMaster_MatchData.csv'

tft_df = pd.read_csv(tft_data_filepath)

tft_df.head()
#Choosing Columns

important_columns = ['level', 'lastRound', 'Ranked', 'combination']

clean_tft_data = tft_df.copy()



#Choosing the first 500 rows to quicken computations

clean_tft_data = clean_tft_data[important_columns][0:500]

clean_tft_data.head()



#DataFrame for the "Data-Science Aspect"

statistical_df = tft_df.copy()

composition_column = statistical_df['combination']

shortened_composition_column = composition_column[0:500]



#Querying Columns from the DataFrame

X_columns = ['combination', 'lastRound', 'level']

y_column = ['Ranked']

X = clean_tft_data[X_columns]

y = clean_tft_data[y_column]
shortened_statistical_df = statistical_df.copy()[0:1000]

shortened_statistical_df['combination'] = shortened_statistical_df['combination'].apply(lambda x: ast.literal_eval(x))

team_comp_column = shortened_statistical_df['combination']



team_comp_dict = team_comp_column.copy().to_dict()

team_comp_list = [values for values in team_comp_dict.values()]

distinct_classes = []

for el in team_comp_list:

    for item in el:

        if item not in distinct_classes:

            distinct_classes.append(item)

            

for item in distinct_classes:

    shortened_statistical_df[item] = 0

    

trait_only_columns = shortened_statistical_df.copy()[distinct_classes]
#trait_only_columns['Blaster'] 

#trait_only_columns['Blaster'] = shortened_statistical_df['combination']['Blaster']





#trait_only_columns['Blaster'].map(team_comp_dict['Blaster']) #I think .map() is the answer I'm looking for.

#team_comp_dict
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
s = (X_train.dtypes == 'object')

categorical_cols = list(s[s].index)

print('Categorical Columns:')

print(categorical_cols)

numerical_cols = list(set(X_train.columns) - set(categorical_cols))

print(numerical_cols)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



numerical_transformer = SimpleImputer(strategy='constant')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0, n_estimators=100)

first_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('forest_model', model)

])

first_pipeline.fit(X_train, y_train)

first_pipeline_predictions = first_pipeline.predict(X_valid)
from sklearn.tree import DecisionTreeRegressor

second_model = DecisionTreeRegressor(random_state=1)

second_pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('DecisionTreeRegressor', second_model)

])



second_pipeline.fit(X_train, y_train)

second_pipeline_predictions = second_pipeline.predict(X_valid)
from sklearn.metrics import mean_absolute_error

print("Random Forest MAE")

print(mean_absolute_error(first_pipeline_predictions, y_valid))

print('')

print('Decision Tree Regressor MAE')

print(mean_absolute_error(second_pipeline_predictions, y_valid))