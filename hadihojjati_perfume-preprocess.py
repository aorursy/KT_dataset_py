import pandas as pd
data = pd.read_csv('../input/fragrances-and-perfumes/perfume.csv')

data.head()
data.info()
import re

split_data = data

split_data.fillna('-1',inplace=True)

for i in range(1,21):

    split_data['notes_{}'.format(str(i))] = data['notes_{}'.format(str(i))].apply(lambda s:re.split('-1|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20',s))
split_data['notes_1']
notes = split_data['notes_1']

for i in range(2,21):

    notes = notes + split_data['notes_{}'.format(str(i))]
notes = notes.apply(lambda s:list(filter(lambda x:x!='',s)))
notes
def top_extract(s):

    top = []

    for i in range (1,len(s)+1):

        if s[i-1].lower() == 'top':

            top.append(s[i])

    return top
def base_extract(s):

    base = []

    for i in range (1,len(s)+1):

        if s[i-1].lower() == 'base':

            base.append(s[i])

    return base
def middle_extract(s):

    middle = []

    for i in range (1,len(s)+1):

        if s[i-1].lower() == 'middle':

            middle.append(s[i])

    return middle
top_notes = notes.apply(top_extract)

middle_notes = notes.apply(middle_extract)

base_notes = notes.apply(base_extract)
top_notes = top_notes.apply(lambda s:list(filter(lambda x:x!='nan',s)))

middle_notes = middle_notes.apply(lambda s:list(filter(lambda x:x!='nan',s)))

base_notes = base_notes.apply(lambda s:list(filter(lambda x:x!='nan',s)))
for i in range(1,21):

    data = data.drop(columns='notes_{}'.format(str(i)))
data.info()
data['top_notes'] = top_notes

data['middle_notes'] = middle_notes

data['base_notes'] = base_notes
data.info()
data.head()
data.to_csv('perfume_notes.csv',index=False)