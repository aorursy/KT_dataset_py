import pandas as pd

df = pd.read_csv('../input/USvideos.csv',error_bad_lines=False)

df
df = pd.read_csv('../input/USvideos.csv',error_bad_lines=False).filter(items=['title','views'])

max(df['views'].tolist())
df.iloc[0]['views'] = '0'
df['title'][0]


views = [ int(x) for x in df['views'].tolist()]

df['views'] = views
df1 = df.loc[df['views'] >= 10000000]#10 Million
df1.head()
df1['title'].tolist()[0]
from textstat.textstat import textstat



test_data = df1['title'].tolist()[0]



print(textstat.flesch_reading_ease(test_data))

print (textstat.smog_index(test_data))

print (textstat.flesch_kincaid_grade(test_data))

print (textstat.coleman_liau_index(test_data))

print (textstat.automated_readability_index(test_data))

print (textstat.dale_chall_readability_score(test_data))

print (textstat.difficult_words(test_data))

print (textstat.linsear_write_formula(test_data))

print (textstat.gunning_fog(test_data))

print (textstat.text_standard(test_data))

print(textstat.sentence_count(test_data))

print(textstat.lexicon_count(test_data))
df2 = df.loc[df['views'] < 1000000]
df2.head()
from textstat.textstat import textstat

for i in range(10):

    test_data = df1['title'].tolist()[i]

    test_data2 = df2['title'].tolist()[i]



    print(textstat.flesch_reading_ease(test_data),textstat.flesch_reading_ease(test_data2))

#     print (textstat.smog_index(test_data),textstat.smog_index(test_data2))

    print (textstat.flesch_kincaid_grade(test_data),textstat.flesch_kincaid_grade(test_data2))

#     print (textstat.text_standard(test_data), textstat.text_standard(test_data2))

#     print (textstat.coleman_liau_index(test_data),textstat.coleman_liau_index(test_data2))

    print (textstat.automated_readability_index(test_data),textstat.automated_readability_index(test_data2))

#     print (textstat.dale_chall_readability_score(test_data), textstat.dale_chall_readability_score(test_data2))

#     print (textstat.difficult_words(test_data), textstat.difficult_words(test_data2))

#     print (textstat.linsear_write_formula(test_data),textstat.linsear_write_formula(test_data2))

#     print (textstat.gunning_fog(test_data), textstat.gunning_fog(test_data2))

    

#     print(textstat.sentence_count(test_data), textstat.sentence_count(test_data2))

#     print(textstat.lexicon_count(test_data), textstat.lexicon_count(test_data2))

    print('****************************************************************************************')
from textstat.textstat import textstat as ts
title = df['title'][1]

fre = ts.flesch_reading_ease(title)

fkg = ts.flesch_kincaid_grade(title)

ari = ts.automated_readability_index(title)

views = df['views'][1]
fre,fkg,ari

reading = 1 - ((fre-fkg)/100)
reading < 0.4
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



title = df['title'][100]

fre = ts.flesch_reading_ease(title)

fkg = ts.flesch_kincaid_grade(title)

ari = ts.automated_readability_index(title)

views = df['views'][100]

print(views) 

objects = ('flesch_reading_ease','flesch_kincaid_grade','automated_readability_index')

y_pos = np.arange(len(objects))

performance = [fre,fkg,ari]

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

# plt.figure(figsize=(1,1))

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Readabilty score Index')

plt.title(title)

 

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



title = df['title'][10]

fre = ts.flesch_reading_ease(title)

fkg = ts.flesch_kincaid_grade(title)

ari = ts.automated_readability_index(title)

views = df['views'][10]

print(views) 

objects = ('flesch_reading_ease','flesch_kincaid_grade','automated_readability_index')

y_pos = np.arange(len(objects))

performance = [fre,fkg,ari]

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

# plt.figure(figsize=(1,1))

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Readabilty score Index')

plt.title(title)

 

plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt



title = df['title'][506]

fre = ts.flesch_reading_ease(title)

fkg = ts.flesch_kincaid_grade(title)

ari = ts.automated_readability_index(title)

views = df['views'][506]

print(views) 

objects = ('flesch_reading_ease','flesch_kincaid_grade','automated_readability_index')

y_pos = np.arange(len(objects))

performance = [fre,fkg,ari]

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

# plt.figure(figsize=(1,1))

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects)

plt.ylabel('Readabilty score Index')

plt.title(title)

 

plt.show()