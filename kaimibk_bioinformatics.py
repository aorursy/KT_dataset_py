import numpy as np
import pandas as pd
from os import listdir
from IPython.display import display
!ls ../input/alsa/pro_act_dataset\ 2/Pro_Act_Dataset/
data_dir = "../input/alsa/pro_act_dataset 2/Pro_Act_Dataset/"  # Specifying where I have the data stored
dfs = {}

# Loop over files in directory
for file in listdir(data_dir):
    
    if ".csv" not in file:
        continue
    
    # Extract file name without .csv extension
    file_ref =  file[0:-4].lower()  # casting string to lowercase
    print(f"Generating {file_ref} DataFrame...")
    
    _df = pd.read_csv(data_dir+file)
    
    # Summarize the contents of the dataframe
    display(_df.describe(include='all'))  # Using `display` to prettify output
    
    print(f"\t {file_ref} shape: {_df.shape}")
    
    # Storing each dataframe in a dictionary, for convinent storage
    dfs[file_ref] = _df
    
    # Delete the individual DataFrame to free up some memory
    del _df  
print(dfs.keys())

dfs["treatment"].head()
dfs["labs"][dfs["labs"]["subject_id"] == 329].head()
dfs["svc"].head()
alsfrs_copy = dfs["alsfrs_training"]
alsfrs_copy.head()
# Create a new column called "Q5_Merge", where we will store the summed values
alsfrs_copy["Q5_Merge"] = alsfrs_copy[["Q5a_Cutting_without_Gastrostomy", "Q5b_Cutting_with_Gastrostomy"]].fillna(0).sum(axis=1)

alsfrs_copy["Q5_Merge"].head()
# For each row in the "ALSFRS_Delta", divide by 365.24 and multiply by 12
alsfrs_copy["ALSFRS_Delta"] = alsfrs_copy["ALSFRS_Delta"].apply(lambda x:  (x/365.24) * 12)
_temp = alsfrs_copy[["subject_id", "ALSFRS_Total", "ALSFRS_R_Total", "ALSFRS_Delta"]].groupby("subject_id")

_temp = _temp.last() - _temp.first()

_temp.head()