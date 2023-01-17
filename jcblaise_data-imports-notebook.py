import os
import pandas as pd
os.listdir('../input')
reviews = pd.read_csv('../input/reviews.csv')
reviews.head()
