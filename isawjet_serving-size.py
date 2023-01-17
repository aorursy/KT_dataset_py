%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# loadind csv to dataframe
food_facts_df = pd.read_csv('../input/FoodFacts.csv')
food_facts_df.carbon_footprint_100g.plot(kind ='hist')
