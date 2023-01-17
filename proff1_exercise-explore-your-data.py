# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()
# Lines below will give you a hint or solution code
#step_1.hint()
#step_1.solution()
# Print summary statistics in next line
home_data.head()
home_data.describe()
home_data.LotArea.describe()['mean']
(home_data.LotArea.describe()['mean']).round()
home_data.info()
from datetime import date
current_date = date.today()
home_data['current_year'] =current_date.year
home_data['diff_years'] = home_data['current_year'].sub(home_data['YearBuilt'],axis=0)
home_data['diff_years'].head()
home_data['diff_years'].describe()['min']

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = (home_data.LotArea.describe()['mean']).round()

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = home_data['diff_years'].describe()['min']

# Checks your answers
step_2.check()
#step_2.hint()
#step_2.solution()