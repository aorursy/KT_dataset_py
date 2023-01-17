import pandas as  pd
import numpy as np

path = "../input/Davis.csv"
df = pd.read_csv(path)
df.head(10)
missing_data = df.isnull()

missing_data.columns.values.tolist()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
df.dropna(subset = ['repwt'], axis = 0, inplace = True)
df.dropna(subset = ['repht'], axis = 0, inplace = True)
#df.reset_index(drop = True, inplace = True)
df.head(10)
df.shape
df.describe()
import matplotlib.pyplot as plt
df["sex"].value_counts().plot.pie()
plt.gca().set_aspect("equal")
df.hist(column = 'height', rwidth = 0.85, bins = 12)
mdf = df[df.sex == 'M']
fdf = df[df.sex == 'F']
from matplotlib import pyplot

bins = np.linspace(145, 200, 40)

pyplot.hist(mdf['height'], bins, alpha=0.5, label='males', edgecolor = 'black')
pyplot.hist(fdf['height'], bins, alpha=0.5, label='females', edgecolor = 'black')
pyplot.legend(loc='upper right')
pyplot.show()
bins = np.linspace(25, 175, 70)

pyplot.hist(mdf['weight'], bins, alpha=0.5, label='males', edgecolor = 'black')
pyplot.hist(fdf['weight'], bins, alpha=0.5, label='females', edgecolor = 'black')
pyplot.legend(loc='upper right')
pyplot.show()
df['bmi']=df['weight']/((0.01*df['height'])*(0.01*df['height']))
df.head(1)
bins = np.linspace(15, 38, 10)

pyplot.hist(mdf['bmi'], bins, alpha=0.5,  label='males', edgecolor = 'black')
pyplot.hist(fdf['bmi'], bins, alpha=0.5,  label='females', edgecolor = 'black')
pyplot.legend(loc='upper right')
pyplot.show()
df['newbmi'] = df['repwt']/((0.01*df['repht'])*(0.01*df['repht'])) 
df['bmi_def'] = df['bmi'] - df['newbmi']
df.head(2)
bins = np.linspace(-7, 5, 50)

pyplot.hist(mdf['bmi_def'],bins,  alpha=0.5,  label='males', edgecolor = 'black')
pyplot.hist(fdf['bmi_def'],bins,  alpha=0.5,  label='females', edgecolor = 'black')
pyplot.legend(loc='upper right')
pyplot.show()