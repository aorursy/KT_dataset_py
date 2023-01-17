import pandas as pd
df = pd.read_csv('../input/park-biodiversity/parks.csv', index_col=['Park Code'])
df.columns = [col.replace(' ', '_').lower() for col in df.columns]
import sys
sys.path.append('../input/pandas-indexing-challenges-validation')
from pandas_indexing_challenges_validation import *










