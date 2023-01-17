import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/european-social-survey-ess-8-ed21-201617/ESS8e02.1_F1.csv')

df.groupby('cntry')['atcherp'].mean().sort_values(ascending=False)