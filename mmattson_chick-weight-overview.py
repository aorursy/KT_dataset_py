# Colab seems to randomly crash.
# I've had it crash on loading the CSV file, even though that works normally most of the time.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pylab as plt
from scipy import stats
import os # accessing directory structure

df1 = pd.read_csv('/kaggle/input/weight-vs-age-of-chicks-on-different-diets/ChickWeight.csv', delimiter=',')
df1.dataframeName = 'ChickWeight.csv'
df1.columns.values[0] = "row_id"
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
# df1.head()  for some reason, this line is causing the notebook to crash without errors reported?
df1.groupby('Time').agg(
    min_weight=('weight', min),
    max_weight=('weight', max),
    avg_weight=('weight', 'mean'),
    num_chicks=('Chick', 'count')    
)

# Uncomment to get number of measurements per chick
# df1.groupby('Chick').agg(num_chicks=('Time', 'count'))
# exclude those 5 chicks from the sample
bad_chicks = [8, 15, 16, 18, 44]
good_chicks = [num for num in range(1,51) if num not in bad_chicks]
df2 = df1[df1.Chick.isin(good_chicks)]
#df2 = df1[~df1.Chick.isin(bad_chicks)]

df3 = df2[df2['Time'] == 21]
df3.groupby('Diet').agg(
    min_weight=('weight', min),
    max_weight=('weight', max),
    avg_weight=('weight', 'mean'),
    num_chicks=('Chick', 'count')    
)
fig, axs = plt.subplots(1, 4)
fig.set_size_inches(14.0, 4.0)
axs[0].set_title('Diet 1')
axs[1].set_title('Diet 2')
axs[2].set_title('Diet 3')
axs[3].set_title('Diet 4')

cdiet1 = df2[df2.Diet == 1]
cdiet2 = df2[df2.Diet == 2]
cdiet3 = df2[df2.Diet == 3]
cdiet4 = df2[df2.Diet == 4]
for id in good_chicks:
    axs[0].plot(cdiet1[cdiet1['Chick']==id].Time,cdiet1[cdiet1['Chick']==id].weight,label=id)
    axs[1].plot(cdiet2[cdiet2['Chick']==id].Time,cdiet2[cdiet2['Chick']==id].weight,label=id)
    axs[2].plot(cdiet3[cdiet3['Chick']==id].Time,cdiet3[cdiet3['Chick']==id].weight,label=id)
    axs[3].plot(cdiet4[cdiet4['Chick']==id].Time,cdiet4[cdiet4['Chick']==id].weight,label=id)

for i in range(4):
    axs[i].set_xlabel("day")
    axs[i].set_ylabel("weight")
#    axs[i].legend(loc='best')
for i in range(1,5):
    print('Diet',i,stats.describe(df3[df3['Diet'] == i].weight))
    print('  S-W normality test ',stats.shapiro(df3[df3['Diet'] == i].weight))

fig, axs = plt.subplots(1, 4)
fig.set_size_inches(12.0, 4.0)
axs[0].hist(df3[df3['Diet'] == 1].weight)
axs[0].set_title('Diet 1')
axs[1].hist(df3[df3['Diet'] == 2].weight)
axs[1].set_title('Diet 2')
axs[2].hist(df3[df3['Diet'] == 3].weight)
axs[2].set_title('Diet 3')
axs[3].hist(df3[df3['Diet'] == 4].weight)
axs[3].set_title('Diet 4')

for i in range(4):
    axs[i].set_xlim(75.0, 405.0)
df3.boxplot(by ='Diet', column =['weight'], grid = False) 

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog = df3['weight'],      # Data
                          groups = df3['Diet'],   # Groups
                          alpha=0.05)         # Significance level
tukey.summary()

# If the distributions were judged to not be normal, then we'd use stats.kruskal to test the null hypothesis
df4 = df2[df2['Time'] == 10]

df4.boxplot(by ='Diet', column =['weight'], grid = False) 

tukey = pairwise_tukeyhsd(endog = df4['weight'],      # Data
                          groups = df4['Diet'],   # Groups
                          alpha=0.05)         # Significance level
tukey.summary()
pivot = df2.pivot_table(index=['Diet','Chick'], columns=['Time'], values=['weight']).reset_index()
pdataf = pd.DataFrame(pivot.to_records())
pdataf.columns = [hdr.replace("('weight', ", "ti").replace(")", "") \
                     for hdr in pdataf.columns]
pdataf['late_diff'] = pdataf['ti21'] - pdataf['ti18']
pdataf.columns.values[1] = "diet"
pdataf.columns.values[2] = "chick"
pdataf.late_diff.hist(bins=30)
df5 = pdataf[pdataf['late_diff'] > 6.0]
df5.groupby('diet').agg(
    min_weight=('ti21', min),
    max_weight=('ti21', max),
    avg_weight=('ti21', 'mean'),
    num_chicks=('chick', 'count')    
)
df5.boxplot(by ='diet', column =['ti21'], grid = False) 

tukey = pairwise_tukeyhsd(endog = df5['ti21'],      # Data
                          groups = df5['diet'],   # Groups
                          alpha=0.05)         # Significance level
tukey.summary()