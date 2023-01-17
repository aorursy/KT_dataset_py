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
data = pd.read_excel('/kaggle/input/language-list-by-country-and-place/Language List by Country and Place.xlsx')
data.columns = ['Country','Language']
language_dic = {}

for index,row in data.iterrows():

    language_dic[row['Country'][0:-1]] = data['Language'].str.split(',')[index]

    

    
flat_list = []

for sublist in data['Language'].str.split(','):

    if type(sublist) == list:

        for item in sublist:

            flat_list.append(item)
flat_list = set(flat_list)
del language_dic['']
for key in language_dic:

    for i in range(len(language_dic[key])):

        language_dic[key][i] = language_dic[key][i].replace(" ", "")
flat_list = list(flat_list)
for i in range(len(flat_list)):

    flat_list[i] = flat_list[i].replace(" ", "")
dic_lang = {}

for lang in flat_list:

    list_of_countries = []

    for country in language_dic: 

        if lang in language_dic[country]:

            list_of_countries.append(country)

            

    dic_lang[lang] = list_of_countries

            

    

            

    

            

        
new_count_dic = {}

for key in dic_lang:

    new_count_dic[key] = len(dic_lang[key])

    
df = pd.DataFrame.from_dict(new_count_dic, orient='index').reset_index()
df.columns = ['Country','lang_count']
df.sort_values('lang_count',ascending = False).head(10)