import pandas as pd

from pandas_profiling import ProfileReport



# Read the training data into a pandas DataFrame

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



# Now produce a profile report:

profile = ProfileReport(train_data, minimal=True, title="Pandas Profiling Report")

# For a more complete report, with correlations and dynamic binning etc. 

# remove the minimal=True flag.
import warnings

warnings.filterwarnings("ignore") # silence an iframe warning
# now display the profile report

profile.to_notebook_iframe()