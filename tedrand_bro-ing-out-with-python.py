# Maybe one too many adult sodas before this, but let's see
# Bro Imports
%matplotlib inline
import numpy as np
import matplotlib.pyplot as bro_plt
import pandas as pd
import colorsys
bro_plt.style.use('seaborn-talk')

# Create the bro df
bro_df = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv", sep=',')
HSV_tuples = [(x*1.0/10, 0.5, 0.5) for x in range(10)]
bro_df.head()
# quite a few columns here LOL
bro_df.columns.values
eduLevels = bro_df['SchoolDegree'].astype('category')
c = len(eduLevels.value_counts())
bro_df.hist(by='SchoolDegree', column = 'HoursLearning', figsize=(15,15))
bro_df.hist(by='SchoolDegree', column = 'Income', figsize=(15,15))
labels = bro_df.Gender.value_counts().index
y_pos = np.arange(len(bro_df.Gender.value_counts()))
bars = bro_plt.barh(y_pos,bro_df.Gender.value_counts())
bro_plt.legend(bars, labels)
bro_plt.title("Gender")
bro_plt.show()
bro_df.Gender.value_counts()
bro_df['AttendedBootcamp'].hist()
campers = bro_df[bro_df['AttendedBootcamp']==1.0]
nerds = bro_df[bro_df['AttendedBootcamp']==0.0]
campers.hist(column = 'Income', figsize=(15,15))
nerds.hist(column = 'Income', bins=4, figsize=(15,15))
