# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#data sent by one of teh intern

df_2 = pd.read_csv('/kaggle/input/title-2/Public Script Sources.xlsx - public_scripts.csv')

#data sent by Katch

df = pd.read_csv('/kaggle/input/imdb-id/KatchU - Movie Selection Dashboard - IDs from Invisible - KatchU - Movie Selection Dashboard - IDs from Invisible.csv')
df
dict = df.set_index('Title')['IMDb #'].to_dict()

dict

df_2
title = df['Title'].tolist()

title_2 = df_2['title'].tolist()
# initializing punctuations string 

punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''



for i in range(len(title)):

    #removing release year which was mentioned in (....) after the movie title

    if ("(" in title[i]):

        num = title[i].split(" ")[-1]

        title[i] = title[i].replace(num,"")

        

    #removing punctuations

    for ele in title[i]:

        if ele in punc:

            title[i] = title[i].replace(ele,"")

    title[i] = title[i].lower()

    title[i] = title[i].replace(" ","")

    

#Doing the same for df_2



for i in range(len(title_2)):

    if ("(" in title_2[i]):

        num = title_2[i].split(" ")[-1]

        title_2[i] = title_2[i].replace(num,"")

        

    for ele in title_2[i]:

        if ele in punc:

            title_2[i] = title_2[i].replace(ele,"")

    title_2[i] = title_2[i].lower()

    title_2[i] = title_2[i].replace(" ","")



    

dict = df.set_index('Title')['IMDb #'].to_dict()

title_id = df['IMDb #'].tolist()

orig_title = df['Title'].tolist()
orig_title
dict = {"Original": orig_title,"Title": title, 'IMDB id': title_id}
dict
df_3 = pd.DataFrame(dict)
df_3.head(50)
title_2
#list our team found scripts for

df_our = pd.read_csv('/kaggle/input/final-list-scrapped/final_list.csv')
df_
df_our[df_our['0'] == 'thesecretoftheplanetoftheapesworkingtitleescapefromtheplanetoftheapesreleasetitlepart1']
#getting titles kin a list

title_our = df_our['0'].tolist()
#in many titles "the town" was extracted as "town,the"



for i in range(len(title_our)):

    

    #Fix the "The town" issue

    if(',' in title_our[i]):

        title_list = title_our[i].split(',')

        title_our[i] = title_list[1]+title_list[0]

  

        

    #removing punctuations

    for ele in title_our[i]:

        if ele in punc:

            title_our[i] = title_our[i].replace(ele,"")



print(title_our)
#checking for missing/unmatched titles

missing = []

present = []

missing_2 = []

missing_3 = []



#titles in Katch's data but not in our data

for i in range(len(title)):

    if (title[i] in title_our ):

        present.append(title[i])

    else:

        missing.append(title[i])

        

        

        

# # #titles in df_2 data but not in our data

# for i in range(len(title_2)):

#     if (title_2[i] in title_our ):

#         pass

#     else:

#         missing_3.append(title_2[i])

        

# #titles in our data but not in df_2 data

# for i in range(len(title_our)):

#     if (title_our[i] in title_2):

#         pass

#     else:

#         missing_3.append(title_our[i])

        





# #titles in our data but not in Katch's data

# for i in range(len(title_our)):

#     if (title_our[i] in title ):

#         pass

#     else:

#         missing_2.append(title_our[i])







        



print(len(present))
len(missing) + len(present)
dict_2 = {'Our title': title_our}
dict_2['thesecretoftheplanetoftheapesworkingtitleescapefromtheplanetoftheapesreleasetitlepart1']
common_title = {'Title': present}
df_5 = pd.DataFrame(common_title)
df_5
df_3[df_3['Title'] == 'hero']
df_4 = pd.DataFrame(dict_2)
df_4
mergedStuff = pd.merge(df_5, df_3, on=['Title'])

merged = df_5.merge(df_3,how='outer',left_on=['Title'],right_on=["Title"])



df_3
df_5
mergedStuff.groupby('Title')
mergedStuff
mergedStuff = mergedStuff.drop_duplicates()
mergedStuff.head(100)
merged = merged.drop_duplicates()

merged
mergedStuff.to_csv('mycsvfile_2.csv',index=False)