# import pandas

import pandas as pd 



# read .csv file and store in cereal_data variable

cereal_data = pd.read_csv("../input/cereal.csv")



#use describe() function to summarize dataset

cereal_data.describe()