import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

%matplotlib inline



key = pd.read_csv('../input/key_1.csv')

pd.set_option("display.width", 150)

pd.set_option("display.max_colwidth", 150)
key.iloc[70920:70925]
key['Pag'] = key['Page'].str.slice(0,-11,None)

key[key['Pag'] == '2016_NFL_draft_en.wikipedia.org_all-access_all-agents']['Page']