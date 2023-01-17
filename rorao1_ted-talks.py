import numpy as np
import pandas as pd
ted = pd.read_csv('../input/ted_main.csv')
#What are the top 5 most viewed Ted Talks?
most_viewed = ted.sort_values('views', ascending = False)
most_viewed.head()
