import pandas as pd
import numpy as np

iowa_file_path = '../input/titanic_fin.csv'
X = pd.read_csv(iowa_file_path)

X.to_csv('csv_2.csv')


