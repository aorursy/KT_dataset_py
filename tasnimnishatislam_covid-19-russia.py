import os

import pandas as pd
confirmed_df = pd.DataFrame()

confirmed_df = pd.read_csv("../input/Russia_Subdivisions_summary_covid19_confirmed.csv")

confirmed_df.head()
death_df = pd.DataFrame()

death_df = pd.read_csv("../input/Russia_Subdivisions_summary_covid19_death.csv")

death_df.head()
recovered_df = pd.DataFrame()

recovered_df = pd.read_csv("../input/Russia_Subdivisions_summary_covid19_recovered.csv")

recovered_df.head()