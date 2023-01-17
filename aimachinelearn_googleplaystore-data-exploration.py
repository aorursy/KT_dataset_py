import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

from collections import Counter
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


googleplaystore = pd.read_csv("../input/googleplaystore.csv")

googleplaystore.head()
gps = googleplaystore

gps["Genres"].value_counts()
gps.info()
gps.corr()
gps.head(20)
gps.columns
gps.tail()
gps.shape
print(gps['Category'].value_counts(dropna =False))
gps.describe()
gps.dtypes
gps['App'].unique()
gps['Category'].unique()
gpsur = pd.read_csv("../input/googleplaystore_user_reviews.csv")

gpsur.head()

gpsur.sample(1)
gpsur.sample(5)
gpsur.info()
print(gpsur.shape)
gpsur.size
