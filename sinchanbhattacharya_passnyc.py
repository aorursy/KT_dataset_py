import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os)
print("Hello")
os.getcwd()
print(os.listdir("../input"))
os.chdir("../input")
os.getcwd()
df_school = pd.read_csv('2016 School Explorer.csv')
df_D5 = pd.read_csv('D5 SHSAT Registrations and Testers.csv')

df_school.head()

df_D5.head()
