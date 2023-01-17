import pandas as pd

import numpy as np



cereal_df = pd.read_csv('../input/cereal.csv')



cereal_df.columns = cereal_df.columns.str.upper()  ## Changing the columns to Upper case 

cereal_df.head()



cereal_df.describe()









cereal_df = cereal_df.describe()

cereal_df.transpose() ## Displaying the summary as columns for better view.
