import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import the data set
source_file_path = "../input/dsm-beuth-edl-demodata-dirty.csv"
source_data = pd.read_csv(source_file_path)

# inspect it
source_data.info()

# since the data set is reasonably small we can view it in its entirety
source_data
# remove rows with all missing values
source_data = source_data.dropna(how="all")
source_data
# check if any missing values are left

if source_data.isna().any().any():
    print("There are still missing values:")

# get all rows which have one or more missing values
source_data[source_data.isna().any(1)]
source_data.loc[5,"gender"] = "Male"
source_data[source_data.isna().any(1)]
# Check for any values of the age attribute which aren't numeric values:
source_data[pd.to_numeric(source_data["age"],downcast="integer",errors="coerce").isna()]
# Check for any values of the age attribute which aren't positive numeric values:
source_data[pd.to_numeric(source_data["age"],errors="coerce")<0]
source_data.loc[18,"age"] = "78"
source_data
all(source_data["full_name"] == source_data["first_name"]+" "+source_data["last_name"])
source_data = source_data.drop(columns="full_name")
source_data
all(source_data["gender"].isin(["Male","Female"]))
source_data = source_data.drop_duplicates(subset=("first_name","last_name","email"))
source_data