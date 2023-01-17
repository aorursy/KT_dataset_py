# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()
# Lines below will give you a hint or solution code
# step_1.hint()
# step_1.solution()
# Print summary statistics in next line
import pandas as pd
fetch_lowaData= '../input/home-data-for-ml-course/train.csv'
load_lowaData = pd.read_csv(fetch_lowaData)
load_lowaData.describe()
# What is the average lot size (rounded to nearest integer)?
import pandas as pd
fetch_lowaData= '../input/home-data-for-ml-course/train.csv'
load_lowaData = pd.read_csv(fetch_lowaData)
load_lowaData.describe()
avg_lot_size = load_lowaData["LotArea"].mean()
avg_lot_size = round(avg_lot_size,0)
print(avg_lot_size)

# As of today, how old is the newest home (current year - the date in which it was built)
#newest_home_age = pd.datetime.now().date()- pd.to_datetime(load_lowaData["YearBuilt"], coerce=True)
#load_lowaData["YearBuilt"] = pd.to_datetime(load_lowaData["YearBuilt"])
least_recent_date = load_lowaData["YearBuilt"].min()
print(least_recent_date)
recent_date = load_lowaData["YearBuilt"].max()
print(recent_date)
newest_home_age = recent_date

# Checks your answers
step_2.check()
#step_2.hint()
#step_2.solution()