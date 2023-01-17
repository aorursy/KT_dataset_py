# Setup

import numpy as np

import pandas as pd
blood_types = pd.DataFrame.from_dict({'blood_types': ['A', 'B', 'AB', 'O']})

blood_types
from sklearn.preprocessing import OneHotEncoder





encoder = OneHotEncoder()

encoded_blood_types = encoder.fit_transform(blood_types)

encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_blood_types)
encoded_df