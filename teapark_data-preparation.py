import numpy as np

import pandas as pd
#Clean the data

columns_info=  ["type",

             "forecastdate",

             "state",

             "startdate",

             "enddate",

             "pollster",

             "grade",

             "samplesize",

             "population",

             "poll_wt",

             "rawpoll_clinton",

             "rawpoll_trump",

             "rawpoll_johnson",

             "rawpoll_mcmullin",

             "adjpoll_clinton",

             "adjpoll_trump",

             "adjpoll_johnson",

             "adjpoll_mcmullin",

             "poll_id"]

date_dtype=["forecastdate","startdate","enddate"]

df=pd.read_csv("../input/presidential_polls.csv",

               usecols=use_columns,

               infer_datetime_format=True,

              parse_dates=date_dtype)

categorical_data=["type","state","pollster","grade","population"]

for category in categorical_data:

    df[category]=df[category].astype("category")
rawdata_columns=["type",

                 "forecastdate",

                 "state",

                 "startdate",

                 "enddate",

                 "pollster",

                 "samplesize",

                 "population",

                 "rawpoll_clinton",

                 "rawpoll_trump",

                 "rawpoll_johnson",

                 "rawpoll_mcmullin",

                 "poll_id"]

raw_data=(df[rawdata_columns]

          .drop_duplicates(subset=["poll_id"])

          .drop("poll_id",axis=1)

         )