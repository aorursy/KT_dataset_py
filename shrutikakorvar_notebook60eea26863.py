import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/data-analysis/country_wise_latest.csv")
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='object')
df['WHO Region'].value_counts()
df['WHO Region'].value_counts(normalize="True")