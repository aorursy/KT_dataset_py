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
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def get_numeric_columns(df):
    numerical_columns = ["Elevation", "Aspect", "Slope", 
                     "Horizontal_Distance_To_Hydrology",
                     "Vertical_Distance_To_Hydrology",
                     "Horizontal_Distance_To_Roadways",
                     "Horizontal_Distance_To_Fire_Points",]
    return df[numerical_columns]

def get_index_columns(df):
    index_columns = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",]
    return df[index_columns]

def get_bools_columns(df):
    bool_columns = [ col for col in df.columns 
                        if col.startswith("Soil_Type") 
                            or col.startswith("Wilderness_Area") ]
    return df[bool_columns]

transformations = [
    make_pipeline(FunctionTransformer(get_numeric_columns, validate=False), StandardScaler()),
    make_pipeline(FunctionTransformer(get_index_columns, validate=False), MinMaxScaler()),
    make_pipeline(FunctionTransformer(get_bools_columns, validate=False), Binarizer()),
]

transformer = make_union(*transformations)
model = RandomForestClassifier()

pipeline = Pipeline([('transformations', transformer), ('rf', model)])

ys = train_df.Cover_Type.values - 1 # 0 base coding 
pipeline.fit(train_df, ys)
y_pred = pipeline.predict(test_df)
def save_submission(test_df, y_pred, filename):
    series = pd.Series(y_pred + 1, index=test_df.Id, name="Cover_Type")
    series.to_csv(filename, header=True)

save_submission(test_df, y_pred, "submission.csv")