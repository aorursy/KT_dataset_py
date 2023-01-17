import pandas as pd
import numpy as np
#downloading the session file
session = pd.read_csv('../input/sessions.csv')
#Unique columns in sessions files
print(session.columns)
np.unique(session.action.dropna())