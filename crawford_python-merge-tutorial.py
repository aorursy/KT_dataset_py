import pandas as pd

import numpy as np



# Create dataframes for examples

left_dataframe = pd.DataFrame({"ID": [1,2,3,4], "left_side": "LEFT"})

right_dataframe = pd.DataFrame({"ID": [3,4,5,6], "right_side": "RIGHT"})

# Left merge on "ID" column 

pd.merge(left=left_dataframe, right=right_dataframe, on="ID", how="left")
# Code for a right merge

pd.merge(left=left_dataframe, right=right_dataframe, on="ID", how="right")
# Inner merge on ID column

pd.merge(left=left_dataframe, right=right_dataframe, on="ID", how="inner")
# Outer merge on ID column

pd.merge(left=left_dataframe, right=right_dataframe, on="ID", how="outer")
# Load restaurant ratings and parking data into dataframes

ratings = pd.read_csv("../input/rating_final.csv")

parking = pd.read_csv("../input/chefmozparking.csv")



# Left merge: Keep everything from the left, drop things from the right (if they don't match)

left_merge = pd.merge(left=ratings, right=parking, on="placeID", how="left")

left_merge.head()