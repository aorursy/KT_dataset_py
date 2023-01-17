import pandas as pd

file = '/kaggle/input/demographics-of-academy-awards-oscars-winners/Oscars-demographics-DFE.csv'        

oscar_demographics = pd.read_csv(file)
import chardet

with open(file, 'rb') as rawdata:

    result = chardet.detect(rawdata.read(100000))

result
oscar_demographics = pd.read_csv(file,encoding='ISO-8859-1')

oscar_demographics.head()