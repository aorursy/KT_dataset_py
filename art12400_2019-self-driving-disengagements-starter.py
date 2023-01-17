import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/2019-autonomous-vehicle-disengagement-reports/2019AutonomousVehicleDisengagementReports.csv')

df1 = pd.read_csv('../input/2019-autonomous-vehicle-disengagement-reports/2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv')

df = pd.concat([df,df1],sort=False)
print("Number of reports: " + str(len(df.index)))

df.head()