import pandas as pd

import numpy as np

import requests
url ='https://raw.githubusercontent.com/fivethirtyeight/data/master/bob-ross/elements-by-episode.csv'

response = requests.get(url)
from io import BytesIO
content = BytesIO(response.content)

df = pd.read_csv(content)
df.head(10)