# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for EDA

import seaborn as sns # for eda

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

#data = pd.read_excel('../input/emap-data-analysis-/emap_analysis.xls')

#data = pd.read_excel('../input/emap-db/emap_1.xls')



data = pd.read_excel('../input/emap_mstp.xls')

data_cldf = pd.read_excel('../input/cldf_1.xls')

data_fields = pd.read_csv('../input/fields.csv')



data.info()
data = data[['EXPR_DSPY_TXT', 'TBL_ID', 'COL_NME', 'MAPPED_ATTR_NME',

            'TRANSLATION_TBL_LIST_TXT','EXPR_XML']]
data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].apply(lambda x: x.split('"')[0])



data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('return', '')

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('if', '')

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('else', '')

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('Config_Options', '')

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('true', '')

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('false', '')

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].apply(lambda x:  re.sub('".*?"', '', x))  # remove words between quotes





def remove_function(display_text):

    display_text = re.sub(r'(\w+)(?=\()', '', display_text)  # remove function

    display_text = re.sub(r'[^.^\w]', ' ', display_text)  # remove symbols

    display_text = re.sub(" \d+", ' ', display_text)  # remove digits

    display_text = re.sub(r'(\w+)(?=\.)', ' ', display_text) #remove input fields before dot

    return display_text





data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].apply(lambda x: remove_function(x))

data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].str.replace('.', '')
data['EXPR_DSPY_TXT'] = data['EXPR_DSPY_TXT'].apply(lambda x: x.strip())  # trim string

data = data[(data.MAPPED_ATTR_NME == 'VALUE') & (data.TBL_ID != '') & (data.COL_NME != '') & (data.EXPR_DSPY_TXT != '')]

data = data[['EXPR_DSPY_TXT', 'TBL_ID', 'COL_NME','EXPR_XML']]



# data['EXPR_DSPY_TXT'].dropna(axis=1)



# pd.concat([pd.Series(row['TBL_ID'], row['COL_NME'], row['EXPR_DSPY_TXT'].split()) for _, row in data.iterrows()]).reset_index()



#print data['EXPR_DSPY_TXT'].str.len()



# data.info()



# data.to_csv('gs_emap_out.csv')
data.sort_values(by=['EXPR_DSPY_TXT'])

data.head()
data_frame = pd.concat([pd.Series(row['TBL_ID'] + '::' + row['COL_NME'] + '::' +  row['EXPR_XML'],

                       row['EXPR_DSPY_TXT'].split()) for (_, row) in

                       data.iterrows()]).reset_index()



# pd.concat([pd.Series(row['TBL_ID'], row['COL_NME'], row['EXPR_DSPY_TXT'].split()) for _, row in data.iterrows()]).reset_index()



# pd.concat([pd.Series(row['TBL_ID'], row['COL_NME'], row['EXPR_DSPY_TXT'].split()) for _, row in data.iterrows()]).reset_index()



# In[54]:



data_frame.head(4)

data_1 = data_frame.iloc[:, 1]

data_2 = data_1.to_frame()



# data_3 = data_2.split(':')



data_3 = data_2[0].apply(lambda x: x.split('::'))



data_4 = data_3.to_frame()



data_frame.columns = ['Input', 'TblColname']





data_frame.head()
#print(data_frame['TblColname'])

#print(data_frame['TblColname'].str.split('::', expand=True))

data_frame[['TBL_ID','COL_NME','EXPR_XML']] = data_frame['TblColname'].str.split('::', expand=True)

data_frame = data_frame[['Input', 'TBL_ID', 'COL_NME', 'EXPR_XML']]



data_frame = data_frame.sort_values(by=['Input'])
data_frame.head(5000)

data_cldf = data_cldf[['COL_NME', 'TBL_ID', 'LOGL_NME', 'COL_DESC']]

data_fields = data_fields[['Field Mnemonic', 'Description', 'Old Mnemonic', 'Definition']]

data_frame_cldf = pd.merge(data_frame, data_cldf, on=['TBL_ID','COL_NME'], how='left', sort=False)

data_fields = data_fields.rename(columns={'Field Mnemonic': 'Input'})

data_frame_cldf_fields = pd.merge(data_frame_cldf, data_fields, on=['Input'], how='left', sort=False)
data_frame_cldf_fields = data_frame_cldf_fields.drop_duplicates(['Input','TBL_ID','COL_NME','EXPR_XML'])

data_frame_cldf_fields.head(4000)
#sns.set(style="darkgrid")

#plt.figure(figsize=(80,40))

#ax = sns.countplot(x="Input", data=data_frame)



data_frame_cldf_fields.info()
data_frame_col_desc = data_frame_cldf_fields[['Input','COL_DESC', 'Definition']]

data_frame_cldf_fields.columns

                                              

                                              
#data_frame_cldf_fields.TBL_ID.count()



import matplotlib.pyplot as plt # for eda 

import collections



#x = collections.Counter(data_frame_cldf_fields.TBL_ID)

x = collections.Counter(data_frame_cldf_fields.Input)



from sklearn import preprocessing

le = preprocessing.LabelEncoder()





fit = df.apply(lambda x: d[x.name].fit_transform(x))

data_frame_cldf_fields['Input','TBL_ID'] = le.fit_transform(data_frame_cldf_fields['Input','TBL_ID'].astype('str'))

#data_frame_cldf_fields = le.fit_transform(data_frame_cldf_fields)

#knn.fit(data_frame_cldf_fields['Input', 'EXPR_XML', 'LOGL_NME', 'COL_DESC',

#       'Description', 'Old Mnemonic', 'Definition'], data_frame_cldf_fields['TBL_ID', 'COL_NME'])
#data_frame_cldf_fields.head()

print(le.classes_)

data_frame_cldf_fields['Input'] = le.inverse_transform(data_frame_cldf_fields['Input'])

data_frame_cldf_fields.head()