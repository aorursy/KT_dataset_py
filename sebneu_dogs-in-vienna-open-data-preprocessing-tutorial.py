# Load the library for CSV file processing

import pandas as pd

# Load the csv file from the open data portal

data_path = '/kaggle/input/dogs-of-vienna/Hunde_Wien.csv'

# Look up the row file and specify the dataset format, e.g. delimiters

data = pd.read_csv(data_path, delimiter=';', encoding='latin-1')

data.head()