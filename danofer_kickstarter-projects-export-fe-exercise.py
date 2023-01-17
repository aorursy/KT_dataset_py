# # Set up code checking

# from learntools.core import binder

# binder.bind(globals())

# from learntools.feature_engineering.ex1 import *



# import pandas as pd



# # click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv',

# #                          parse_dates=['click_time'])

# # click_data.head(10)
import pandas as pd

ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',

                 parse_dates=['deadline', 'launched']).drop("ID",axis=1)

print(ks.shape)

ks.head(10)
# Drop live projects

ks = ks.query('state != "live"')

# ks = ks.query('state != "undefined"')
# Add outcome column, "successful" == 1, others are 0

ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))
ks.to_csv("featEng_kickstarterProj_v1.csv.gz",index=False,compression="gzip")