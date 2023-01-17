%matplotlib inline

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sqlite3
import random

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
salaries = pd.read_csv('../input/Salaries.csv')
salaries.info()
salaries = salaries.convert_objects(convert_numeric=True)
# Some Data Munging and Type Conversion
df = df[(df.BasePay != 'Not Provided') & (df.BasePay != '0.00')].copy()

for column in ["BasePay", "OvertimePay", "Benefits", "TotalPay", "TotalPayBenefits"]:
    df[column] = df[column].map(float)
salaries = salaries.drop('Notes', axis=1)
salaries.describe()
# i am using seaborn to change aesthetics of the plots
sns.set_style("whitegrid")

# matplotlib.pyplot is the main module that provides the plotting API
x = [np.random.uniform(100) for _ in range(200)]
y = [np.random.uniform(100) for _ in range(200)]
plt.scatter(x,y)
sns.jointplot(x='X', y='Y', data=pd.DataFrame({'X': x, 'Y': y}))
