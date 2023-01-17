# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from google.cloud import bigquery
import matplotlib.pyplot as plt
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format

client = bigquery.Client()

query = """
SELECT block_number, transaction_index, from_address, to_address, value, input, receipt_status FROM `bigquery-public-data.crypto_ethereum.transactions` 
WHERE block_number >= 7710758 and
(LOWER(to_address) = "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b" or
LOWER(from_address) = "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b" or
LOWER(to_address) = "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5" or
LOWER(from_address) = "0x4ddc2d193948926d02f9b1fe9e1daa0718270ed5" or
LOWER(to_address) = "0x39aa39c021dfbae8fac545936693ac917d5e7563" or
LOWER(from_address) = "0x39aa39c021dfbae8fac545936693ac917d5e7563" or
LOWER(to_address) = "0xb3319f5d18bc0d84dd1b4825dcde5d5f7266d407" or
LOWER(from_address) = "0xb3319f5d18bc0d84dd1b4825dcde5d5f7266d407" or
LOWER(to_address) = "0x158079ee67fce2f58472a96584a73c7ab9ac95c1" or
LOWER(from_address) = "0x158079ee67fce2f58472a96584a73c7ab9ac95c1" or
LOWER(to_address) = "0xf5dce57282a584d2746faf1593d3121fcac444dc" or
LOWER(from_address) = "0xf5dce57282a584d2746faf1593d3121fcac444dc" or
LOWER(to_address) = "0x6c8c6b02e7b2be14d4fa6022dfd6d75921d90e4e" or
LOWER(from_address) = "0x6c8c6b02e7b2be14d4fa6022dfd6d75921d90e4e" or
LOWER(to_address) = "0xa7ff0d561cd15ed525e31bbe0af3fe34ac2059f6" or
LOWER(from_address) = "0xa7ff0d561cd15ed525e31bbe0af3fe34ac2059f6")
ORDER BY block_number, transaction_index
LIMIT 200000
"""

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
df.to_csv('out2.csv', index = False)
