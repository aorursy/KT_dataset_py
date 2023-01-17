import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='white',palette = 'Set3',context = 'talk')
dc = pd.read_csv('../input/dc-wikia-data.csv')

mar = pd.read_csv('../input/marvel-wikia-data.csv')

print(dc.shape,mar.shape)
def clean(x):

    x.name = x.name.apply(lambda x: x.split('(')[0])

    cols = ('ID','ALIGN','EYE','HAIR','SEX','ALIVE')

    for c in cols:

       x[c]=  x[c].fillna('Unknown')

       x[c]=  x[c].apply(lambda x: x.split(' ')[0])



clean(dc)
mar.columns

mar['YEAR'] = mar['Year']

clean(mar)
dc.head(2)
dc.groupby('YEAR')['SEX'].value_counts().unstack().plot(figsize = (16,6))

plt.title('DC - Character Gender Evolvement')



mar.groupby('YEAR')['SEX'].value_counts().unstack().plot(figsize = (16,6))

plt.title('Marvels - Character Gender Evolvement')
plt.subplots(1,2,figsize = (18,6))

plt.subplot(121)

sns.countplot(x= 'ALIGN',hue = 'SEX',data = dc)

plt.legend(loc='upper right')

plt.subplot(122)

sns.countplot(x= 'ALIVE',hue = 'SEX',data = dc)

plt.legend(loc='upper right')
plt.subplots(1,2,figsize = (18,6))

plt.subplot(121)

sns.countplot(x= 'ALIGN',hue = 'SEX',data = mar)

plt.legend(loc='upper right')

plt.subplot(122)

sns.countplot(x= 'ALIVE',hue = 'SEX',data = mar)

plt.legend(loc='upper right')
dead_mar = mar[mar.ALIVE == 'Deceased']

dead_dc = dc[dc.ALIVE == 'Deceased']



plt.subplots(1,2,figsize = (18,6))

plt.subplot(121)

sns.countplot(x='ALIGN',data = dead_mar)

plt.title ('Marvels - dead hero distribution')

plt.subplot(122)

sns.countplot(x='ALIGN',data = dead_dc)

plt.title ('DC - dead hero distribution')
display(dc[dc.SEX =='Transgender'][['name','YEAR']])

display(mar[mar.SEX =='Genderfluid'][['name','YEAR']])
tmp =mar.sort_values(by = 'APPEARANCES',ascending = False)[:10][['name','SEX','APPEARANCES']]

tmp
tmp =dc.sort_values(by = 'APPEARANCES',ascending = False)[:10][['name','SEX','APPEARANCES']]

tmp
good_mar = mar[mar.ALIGN == 'Good'].sort_values(by = 'APPEARANCES',ascending = False)[:10]

good_dc = dc[dc.ALIGN == 'Good'].sort_values(by = 'APPEARANCES',ascending = False)[:10]

bad_mar = mar[mar.ALIGN == 'Bad'].sort_values(by = 'APPEARANCES',ascending = False)[:10]

bad_dc = dc[dc.ALIGN == 'Bad'].sort_values(by = 'APPEARANCES',ascending = False)[:10]

plt.subplots(1,2,figsize=(18,6))

plt.subplots_adjust(wspace =0.3)

plt.subplot(121)

sns.boxenplot(x='APPEARANCES', y='HAIR',data=good_dc,hue='EYE').set_title('DC-Top Appearance good hero looking')

plt.subplot(122)

sns.boxenplot(x='APPEARANCES', y='HAIR',data=bad_dc,hue='EYE').set_title('DC-Top Appearance bad hero looking')
#bad_dc[bad_dc.HAIR == 'Green'] # JOKER 1940

good_dc[(good_dc.HAIR == 'Black') &(good_dc.EYE == 'Blue')]
plt.subplots(1,2,figsize=(18,6))

plt.subplots_adjust(wspace =0.3)

plt.subplot(121)

sns.boxplot(x='APPEARANCES', y='HAIR',data=good_mar,hue='EYE').set_title('DC-Top Appearance good hero looking')

plt.subplot(122)

sns.boxplot(x='APPEARANCES', y='HAIR',data=bad_mar,hue='EYE').set_title('DC-Top Appearance bad hero looking')
bad_mar[bad_mar.HAIR == 'Auburn'] #Norman Osborn 1964

bad_mar[bad_mar.HAIR == 'Bald'] #Wilson Fisk 1967