import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Read datas

median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level.info()
percentage_people_below_poverty_level["Geographic Area"].unique()
percentage_people_below_poverty_level["City"].unique()
# to outcome Resulting of poverty rate total amount 

percentage_people_below_poverty_level.poverty_rate.count()
# Outcome entire of rate_count's value counts and number of it.

percentage_people_below_poverty_level.poverty_rate.value_counts()