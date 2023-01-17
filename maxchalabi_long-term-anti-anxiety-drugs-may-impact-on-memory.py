# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style



style.use('ggplot')





df = pd.read_csv('/kaggle/input/memory-test-on-drugged-islanders-data/Islander_data.csv')

 

df['Drug']= df['Drug'].str.replace('A', 'Alprazolam', case = False)

df['Drug']= df['Drug'].str.replace('T', 'Triazolam', case = False)

df['Drug']= df['Drug'].str.replace('S', 'Sugar Tablet', case = False) 

df['Happy_Sad_group']= df['Happy_Sad_group'].str.replace('H', 'Happy', case = False)

df['Happy_Sad_group']= df['Happy_Sad_group'].str.replace('S', 'Sad', case = False)



df.head()
print('Sub-group sizes:')

print('')

print('Happy-Primed:', len(df[(df['Happy_Sad_group']=='Happy')]))

print('Sad-Primed:', len(df[(df['Happy_Sad_group']=='Sad')]))

print('')

print('Alprazolam:', len(df[(df['Drug']=='Alprazolam')]))

print('Triazolam:', len(df[(df['Drug']=='Triazolam')]))

print('Sugar Tablet:', len(df[(df['Drug']=='Sugar Tablet')]))

print('')

print('Alprazolam, 1xDosage:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==1)]))

print('Alprazolam, 2xDosage:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==2)]))

print('Alprazolam, 3xDosage:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==3)]))

print('Triazolam, 1xDosage:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==1)]))

print('Triazolam, 2xDosage:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==2)]))

print('Triazolam, 3xDosage:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==3)]))

print('Sugar Tablet, 1xDosage:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==1)]))

print('Sugar Tablet, 2xDosage:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==2)]))

print('Sugar Tablet, 3xDosage:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==3)]))

print('')

print('Alprazolam, 1xDosage, Happy-primed:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Happy')]))

print('Alprazolam, 2xDosage, Happy-primed:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Happy')]))

print('Alprazolam, 3xDosage, Happy-primed:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Happy')]))

print('Triazolam, 1xDosage, Happy-primed:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Happy')]))

print('Triazolam, 2xDosage, Happy-primed:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Happy')]))

print('Triazolam, 3xDosage, Happy-primed:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Happy')]))

print('Sugar Tablet, 1xDosage, Happy-primed:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Happy')]))

print('Sugar Tablet, 2xDosage, Happy-primed:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Happy')]))

print('Sugar Tablet, 3xDosage, Happy-primed:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Happy')]))

print('Alprazolam, 1xDosage, Sad-primed:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Sad')]))

print('Alprazolam, 2xDosage, Sad-primed:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Sad')]))

print('Alprazolam, 3xDosage, Sad-primed:', len(df[(df['Drug']=='Alprazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Sad')]))

print('Triazolam, 1xDosage, Sad-primed:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Sad')]))

print('Triazolam, 2xDosage, Sad-primed:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Sad')]))

print('Triazolam, 3xDosage, Sad-primed:', len(df[(df['Drug']=='Triazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Sad')]))

print('Sugar Tablet, 1xDosage, Sad-primed:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Sad')]))

print('Sugar Tablet, 2xDosage, Sad-primed:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Sad')]))

print('Sugar Tablet, 3xDosage, Sad-primed:', len(df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Sad')]))
fig = plt.figure(figsize=(10,5))

sns.distplot(df['Diff'], hist=True, kde=True, 

             bins=int(180/5), color = 'darkred', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

plt.title('Density Plot of score difference between memory tests')

plt.xlabel('Before/After Score Difference')



fig = plt.figure(figsize=(10,5))

before = df['Mem_Score_Before']

after = df['Mem_Score_After']

plt.hist([before, after], label=['Before', 'After'])

plt.legend(loc='best')

plt.title('Side-by-side histogram of memory test scores before & after experimental method')

plt.xlabel('Memory Test Score')



fig = plt.figure(figsize=(10,5))

sns.distplot(df['age'], hist=True, kde=True, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

plt.title('Density Plot of age in participants')

plt.xlabel('Age')



fig = plt.figure(figsize=(10,5))

plt.scatter(df['age'], df['Mem_Score_Before'], label='Before')

m1, b1 = np.polyfit(df['age'], df['Mem_Score_Before'], 1)

plt.plot(df['age'], m1*df['age']+b1, label='Line of Best Fit (Before)')

plt.scatter(df['age'], df['Mem_Score_After'], label='After')

m2, b2 = np.polyfit(df['age'], df['Mem_Score_After'], 1)

plt.plot(df['age'], m2*df['age']+b2, label='Line of Best Fit (After)')

plt.legend(loc='best')

plt.title('Distribution of Memory Test Scores over Age')

plt.xlabel('Age')

plt.ylabel('Memory Test Score')



fig = plt.figure(figsize=(10,5))

plt.scatter(df['age'], df['Diff'], color = 'darkred')

m3, b3 = np.polyfit(df['age'], df['Diff'], 1)

plt.plot(df['age'], m3*df['age']+b3, color = 'darkred', label='Line of Best Fit')

plt.legend(loc='best')

plt.title('Distribution of Score difference between Memory Tests over Age')

plt.xlabel('Age')

plt.ylabel('Score difference between Memory Tests')
fig = plt.figure(figsize=(10,7))

sns.catplot(x='Drug', y='Diff', hue='Dosage', col='Happy_Sad_group', data=df, kind="box")
A1H = df[(df['Drug']=='Alprazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

A2H = df[(df['Drug']=='Alprazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

A3H = df[(df['Drug']=='Alprazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

T1H = df[(df['Drug']=='Triazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

T2H = df[(df['Drug']=='Triazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

T3H = df[(df['Drug']=='Triazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

S1H = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

S2H = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

S3H = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Happy')]['Diff'].values

A1S = df[(df['Drug']=='Alprazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

A2S = df[(df['Drug']=='Alprazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

A3S = df[(df['Drug']=='Alprazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

T1S = df[(df['Drug']=='Triazolam')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

T2S = df[(df['Drug']=='Triazolam')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

T3S = df[(df['Drug']=='Triazolam')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

S1S = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==1)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

S2S = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==2)&(df['Happy_Sad_group']=='Sad')]['Diff'].values

S3S = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==3)&(df['Happy_Sad_group']=='Sad')]['Diff'].values



A1 = df[(df['Drug']=='Alprazolam')&(df['Dosage']==1)]['Diff'].values

A2 = df[(df['Drug']=='Alprazolam')&(df['Dosage']==2)]['Diff'].values

A3 = df[(df['Drug']=='Alprazolam')&(df['Dosage']==3)]['Diff'].values

T1 = df[(df['Drug']=='Triazolam')&(df['Dosage']==1)]['Diff'].values

T2 = df[(df['Drug']=='Triazolam')&(df['Dosage']==2)]['Diff'].values

T3 = df[(df['Drug']=='Triazolam')&(df['Dosage']==3)]['Diff'].values

S1 = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==1)]['Diff'].values

S2 = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==2)]['Diff'].values

S3 = df[(df['Drug']=='Sugar Tablet')&(df['Dosage']==3)]['Diff'].values





A = df[(df['Drug']=='Alprazolam')]

T = df[(df['Drug']=='Triazolam')]

S = df[(df['Drug']=='Sugar Tablet')]



D1 = df[(df['Dosage']==1)]

D2 = df[(df['Dosage']==2)]

D3 = df[(df['Dosage']==3)]



Happy = df[(df['Happy_Sad_group']=='Happy')] 

Sad = df[(df['Happy_Sad_group']=='Sad')]
from scipy.stats import levene



W1, p_val1 = levene(A1H, A2H, A3H, T1H, T2H, T3H, S1H, S2H, S3H, A1S, A2S, A3S, T1S, T2S, T3S, S1S, S2S, S3S)

W2, p_val2 = levene(A['Diff'].values, T['Diff'].values, S['Diff'].values)

p_val2 = f'{p_val2:.9f}'

W3, p_val3 = levene(Happy['Diff'].values, Sad['Diff'].values)

W4, p_val4 = levene(D1['Diff'].values, D2['Diff'].values, D3['Diff'].values)

p_val4 = f'{p_val4:.9f}'

W5, p_val5 = levene(A1, A2, A3, T1, T2, T3, S1, S2, S3)



print("Levene's Test results (Drug condition):")

print("W-score:", W2)

print("p-value:", p_val2)

print('')

print("Levene's Test results (Priming condition):")

print("W-score:", W3)

print("p-value:", p_val3)

print('')

print("Levene's Test results (Dosage condition):")

print("W-score:", W4)

print("p-value:", p_val4)

print('')

print("Levene's Test results (Drugs x Dosage x Happy/Sad):")

print("W-score:", W1)

print("p-value:", p_val1)

print('')

print("Levene's Test results (Drugs x Dosage):")

print("W-score:", W5)

print("p-value:", p_val5)
from scipy.stats import kruskal, f_oneway, ttest_ind



F_all, p_val_all = f_oneway(A1H, A2H, A3H, T1H, T2H, T3H, S1H, S2H, S3H, A1S, A2S, A3S, T1S, T2S, T3S, S1S, S2S, S3S)

p_val_all = f'{p_val_all:.9f}'

H_Drug, p_val_Drug = kruskal(A['Diff'].values, T['Diff'].values, S['Diff'].values)

p_val_Drug = f'{p_val_Drug:.9f}'

F_priming, p_val_priming = f_oneway(Happy['Diff'].values, Sad['Diff'].values)

H_Dosage, p_val_Dosage = kruskal(D1['Diff'].values, D2['Diff'].values, D3['Diff'].values)

p_val_Dosage = f'{p_val_Dosage:.20f}'

F_DosagexDrug, p_val_DosagexDrug = f_oneway(A1, A2, A3, T1, T2, T3, S1, S2, S3)

p_val_DosagexDrug = f'{p_val_DosagexDrug:.20f}'





print("Kruskal-Wallis Test results (Drug condition):")

print("H-score:", H_Drug)

print("p-value:", p_val_Drug)

print('')

print("1-Way ANOVA Test results (Priming condition):")

print("F-score:", F_priming)

print("p-value:", p_val_priming)

print('')

print("Kruskal-Wallis Test results (Dosage condition):")

print("H-score:", H_Dosage)

print("p-value:", p_val_Dosage)

print('')

print("1-Way ANOVA Test results (all subgroups):")

print("F-score:", F_all)

print("p-value:", p_val_all)

print('')

print("1-Way ANOVA Test results (Drug x Dosage):")

print("F-score:", F_DosagexDrug)

print("p-value:", p_val_DosagexDrug)
all_groups = [A1H, A2H, A3H, S1H, S2H, S3H, T1H, T2H, T3H, A1S, A2S, A3S, S1S, S2S, S3S, T1S, T2S, T3S]

dosages = [D1['Diff'].values, D2['Diff'].values, D3['Diff'].values]

drugs = [A['Diff'].values, S['Diff'].values, T['Diff'].values]

dosage_x_drug = [A1, A2, A3, S1, S2, S3, T1, T2, T3]





from matplotlib.markers import TICKDOWN



def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 12,color = 'k'):

    # draw a line with downticks at the ends

    plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)

    # draw the text with a bounding box covering up the line

    plt.text(0.5*(start+end),

             height+0.25,

             displaystring,

             ha = 'center',

             va='center',

             bbox=dict(facecolor='none', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),

             size = fontsize)



bonf_corrected_all_groups_threshold = 0.05 / (len(all_groups)*(len(all_groups)-1))/2

bonf_corrected_dosages_threshold = 0.05 / (len(dosages)*(len(dosages)-1))/2

bonf_corrected_drugs_threshold = 0.05 / (len(drugs)*(len(drugs)-1))/2

bonf_corrected_dosage_x_drug_threshold = 0.05 / (len(dosage_x_drug)*(len(dosage_x_drug)-1))/2



fig = plt.figure(figsize=(10,7))

plt.title('Main effect of Dosage')

plt.ylabel('Score difference between memory tests')

plt.xlabel('Dosage')

sig_index1 = []

sig_index2 = []

p_vals = []

heights = []

ind = np.arange(len(dosages))

for group in range(len(dosages)):

  a = dosages[group]

  plt.bar(ind[group], np.mean(a))

  for other_group in range(len(dosages[group+1:])):

    b = dosages[group+1:][other_group]

    t_stat, p_val = ttest_ind(a, b, equal_var=False)

    heights.append(np.mean(a))

    if p_val < bonf_corrected_dosages_threshold:

      p_val = f'{p_val:.9f}'

      sig_index1.append(ind[group])

      sig_index2.append(ind[group+1:][other_group])

      p_vals.append('')

for i in range(len(sig_index1)):

  significance_bar(sig_index1[i], sig_index2[i], max(heights), p_vals[i])

plt.xticks(ind, [1, 2, 3])

plt.show()



fig = plt.figure(figsize=(10,7))

plt.title('Main effect of Drug')

plt.ylabel('Score difference between memory tests')

plt.xlabel('Drug')

sig_index1 = []

sig_index2 = []

p_vals = []

heights = []

ind = np.arange(len(drugs))

for group in range(len(drugs)):

  a = drugs[group]

  plt.bar(ind[group], np.mean(a))

  for other_group in range(len(drugs[group+1:])):

    b = drugs[group+1:][other_group]

    t_stat, p_val = ttest_ind(a, b, equal_var=False)

    heights.append(np.mean(a))

    if p_val < bonf_corrected_drugs_threshold:

      p_val = f'{p_val:.9f}'

      sig_index1.append(ind[group])

      sig_index2.append(ind[group+1:][other_group])

      p_vals.append('')

for i in range(len(sig_index1)):

  significance_bar(sig_index1[i], sig_index2[i], max(heights)+i, p_vals[i])

plt.xticks(ind, ['Alprazolam', 'Sugar Tablet', 'Triazolam'])

plt.show()





fig = plt.figure(figsize=(15,12))

plt.title('Statistical comparisons of all sub groups')

plt.ylabel('Score difference between memory tests')

colours = 10*['red','blue','green']

# sns.catplot(x='Drug', y='Diff', hue='Dosage', col='Happy_Sad_group', data=df, kind="box")

sig_index1 = []

sig_index2 = []

p_vals = []

heights = []

ind = np.arange(len(all_groups))

for group in range(len(all_groups)):

  a = all_groups[group]

  plt.bar(ind[group], np.mean(a), color=colours[group])

  for other_group in range(len(all_groups[group+1:])):

    b = all_groups[group+1:][other_group]

    t_stat, p_val = ttest_ind(a, b, equal_var=False)

    heights.append(np.mean(a))

    if p_val < bonf_corrected_all_groups_threshold:

      p_val = f'{p_val:.9f}'

      sig_index1.append(ind[group])

      sig_index2.append(ind[group+1:][other_group])

      # p_vals.append(f'p-value: {p_val}')

      p_vals.append('')

for i in range(len(sig_index1)):

  significance_bar(sig_index1[i], sig_index2[i], max(heights)+i, p_vals[i])

plt.xticks(ind, ['Alprazolam, Dosage 1, Happy', 'Alprazolam, Dosage 2, Happy', 'Alprazolam, Dosage 3, Happy',

                 'Sugar Tablet, Dosage 1, Happy', 'Sugar Tablet, Dosage 2, Happy', 'Sugar Tablet, Dosage 3, Happy',

                 'Triazolam, Dosage 1, Happy', 'Triazolam, Dosage 2, Happy', 'Triazolam, Dosage 3, Happy',

                 'Alprazolam, Dosage 1, Sad', 'Alprazolam, Dosage 2, Sad', 'Alprazolam, Dosage 3, Sad',

                 'Sugar Tablet, Dosage 1, Sad', 'Sugar Tablet, Dosage 2, Sad', 'Sugar Tablet, Dosage 3, Sad',

                 'Triazolam, Dosage 1, Sad', 'Triazolam, Dosage 2, Sad', 'Triazolam, Dosage 3, Sad'])

plt.xticks(rotation=90)

plt.show()





fig = plt.figure(figsize=(10,7))

plt.title('Statistical comparisons of all Drug x Dosage groups')

plt.ylabel('Score difference between memory tests')

colours = 10*['red','blue','green']

sig_index1 = []

sig_index2 = []

p_vals = []

heights = []

ind = np.arange(len(dosage_x_drug))

for group in range(len(dosage_x_drug)):

  a = dosage_x_drug[group]

  plt.bar(ind[group], np.mean(a), color=colours[group])

  for other_group in range(len(dosage_x_drug[group+1:])):

    b = dosage_x_drug[group+1:][other_group]

    t_stat, p_val = ttest_ind(a, b, equal_var=False)

    heights.append(np.mean(a))

    if p_val < bonf_corrected_dosage_x_drug_threshold:

      p_val = f'{p_val:.9f}'

      sig_index1.append(ind[group])

      sig_index2.append(ind[group+1:][other_group])

      # p_vals.append(f'p-value: {p_val}')

      p_vals.append('')

for i in range(len(sig_index1)):

  significance_bar(sig_index1[i], sig_index2[i], max(heights)+i, p_vals[i])

plt.xticks(ind, ['Alprazolam, Dosage 1', 'Alprazolam, Dosage 2', 'Alprazolam, Dosage 3',

                 'Sugar Tablet, Dosage 1', 'Sugar Tablet, Dosage 2', 'Sugar Tablet, Dosage 3',

                 'Triazolam, Dosage 1', 'Triazolam, Dosage 2', 'Triazolam, Dosage 3'])

plt.xticks(rotation=90)

plt.show()
fig = plt.figure(figsize=(10,5))



plt.scatter(A['Dosage'], A['Diff'])

m1, b1 = np.polyfit(A['Dosage'], A['Diff'], 1)

plt.plot(A['Dosage'], m1*A['Dosage']+b1, label='Alprazolam')

plt.scatter(S['Dosage'], S['Diff'])

m2, b2 = np.polyfit(S['Dosage'], S['Diff'], 1)

plt.plot(S['Dosage'], m2*S['Dosage']+b2, label='Sugar Tablet')

plt.scatter(T['Dosage'], T['Diff'])

m3, b3 = np.polyfit(T['Dosage'], T['Diff'], 1)

plt.plot(T['Dosage'], m3*T['Dosage']+b3, label='Triazolam')





plt.legend(loc='best')

plt.title('Effect of Dosage on Before/After Memory Test score difference depending on Drug')

plt.xlabel('Dosage')

plt.ylabel('Score difference between Memory Tests')





fig = plt.figure(figsize=(10,5))



AH = A[(A['Happy_Sad_group']=='Happy')]

SH = S[(S['Happy_Sad_group']=='Happy')]

TH = T[(T['Happy_Sad_group']=='Happy')]

AS = A[(A['Happy_Sad_group']=='Sad')]

SS = S[(S['Happy_Sad_group']=='Sad')]

TS = T[(T['Happy_Sad_group']=='Sad')]



plt.scatter(AH['Dosage'], AH['Diff'])

m1, b1 = np.polyfit(AH['Dosage'], AH['Diff'], 1)

plt.plot(AH['Dosage'], m1*AH['Dosage']+b1, label='Alprazolam - Happy')

plt.scatter(SH['Dosage'], SH['Diff'])

m2, b2 = np.polyfit(SH['Dosage'], SH['Diff'], 1)

plt.plot(SH['Dosage'], m2*SH['Dosage']+b2, label='Sugar Tablet - Happy')

plt.scatter(TH['Dosage'], TH['Diff'])

m3, b3 = np.polyfit(TH['Dosage'], TH['Diff'], 1)

plt.plot(TH['Dosage'], m3*TH['Dosage']+b3, label='Triazolam - Happy')



plt.scatter(AS['Dosage'], AS['Diff'])

m1, b1 = np.polyfit(AS['Dosage'], AS['Diff'], 1)

plt.plot(AS['Dosage'], m1*AS['Dosage']+b1, label='Alprazolam - Sad')

plt.scatter(SS['Dosage'], SS['Diff'])

m2, b2 = np.polyfit(SS['Dosage'], SS['Diff'], 1)

plt.plot(SS['Dosage'], m2*SS['Dosage']+b2, label='Sugar Tablet - Sad')

plt.scatter(TS['Dosage'], TS['Diff'])

m3, b3 = np.polyfit(TS['Dosage'], TS['Diff'], 1)

plt.plot(TS['Dosage'], m3*TS['Dosage']+b3, label='Triazolam - Sad')





plt.legend(loc='best')

plt.title('Effect of Dosage on Before/After Memory Test score difference depending on Drug & Priming condition')

plt.xlabel('Dosage')

plt.ylabel('Score difference between Memory Tests')