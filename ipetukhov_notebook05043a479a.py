import numpy as np 

import pandas as pd
train = pd.read_json('../input/test.json', 'r')

test = pd.read_json('../input/train.json', 'r')

test.head(5)