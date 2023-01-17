import pandas as pd

from pandas import read_csv

crimes = read_csv('../input/Chicago_Crimes_2012_to_2017.csv', index_col='Date')