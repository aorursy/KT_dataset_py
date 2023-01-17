import pandas_profiling 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



country_pr = pd.read_csv('/kaggle/input/socioeconomic-country-profiles/soci_econ_country_profiles.csv')

country_pr.info()
country_pr.profile_report()