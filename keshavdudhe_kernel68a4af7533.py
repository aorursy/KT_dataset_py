import pandas as pd

import numpy as np

import os

import json

from pprint import pprint

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from os import walk

def conver_json_to_dataframe(file_name):

    path      = "/kaggle/input/CORD-19-research-challenge/" + "/" + file_name + "/" + file_name 

    files = []

    for (dirpath, dirnames, filenames) in walk(path):

        files.extend(filenames)



    all_json_files_in_one_list = []



    for file in files:

    #     print(file)

        mypath = (path+"/"+file)

        with open(mypath) as json_file:

            json_data = json.load(json_file)

            all_json_files_in_one_list = all_json_files_in_one_list + [json_data]



    df=pd.DataFrame(all_json_files_in_one_list)

    return df
def get_bib_entries_from_datafrmae(df):

    

    bib_entries_1 = []

    

    for i in range(0,df.shape[0]):

        keys = list(df['bib_entries'][i].keys())

        for key in keys:

    #         print(key)

    #         print([df['bib_entries'][i][key]])

            bib_entries_1 = bib_entries_1 + [df['bib_entries'][i][key]]   

        

        

    #sort by year

    df_1=pd.DataFrame(bib_entries_1).sort_values('year').dropna().reset_index(drop=True)

    

    return df_1

# plt.plot(df_1['year'])
def subset_data(df_1):    

    df_2=df_1.loc[df_1['year']>=2019.0,:].reset_index(drop=True)

    print(df_2.shape)

    return df_2

    
def plot_key(keyword_look_in_data_list,look,file_name,df_2):

    

    

    index = []

    for i in range(0,df_2.shape[0]):

        for keyword_look_in_data in keyword_look_in_data_list:

            if keyword_look_in_data in df_2['title'][i].lower():

                if keyword_look_in_data == look:

                    #print("_____________________________________________________")

                    #print(i)

                    index = index + [i]

                    #print(df_2['title'][i])



    df_3=df_2.loc[index,:].reset_index(drop=True)

    #print(df_3.shape)

    #df_3.head(1)    

    

   





    ref_id = list(df_3['ref_id'])



    ref_id=[int(i[1:]) for i in ref_id]



    ref_id = list(np.sort(ref_id))



    ax = plt.subplot(111)

    ax.plot(ref_id ,label=look)



    ax.legend(loc='upper left');

    plt.xlabel('unique count of bibliography')

    plt.ylabel(' bibliography  number ') 



    plt.title(('compare bibliography '+file_name))

    

def convert_data(file_name):

    

    df = conver_json_to_dataframe(file_name)

    df_1 = get_bib_entries_from_datafrmae(df)

    df_2 = subset_data(df_1) 

    

    return df_2
keyword_look_in_data_list = ["pandemic" , "epidemic", "quarantine", "isolation", "respirator","ventilator"]
file_names = ["biorxiv_medrxiv","comm_use_subset","noncomm_use_subset","pmc_custom_license"]
plt.figure(figsize=(15,5));

file_name = "biorxiv_medrxiv"

df_2 = convert_data(file_name)

for look in keyword_look_in_data_list:

    plot_key(keyword_look_in_data_list,look,file_name,df_2);
plt.figure(figsize=(15,5));

file_name = "comm_use_subset"

df_2 = convert_data(file_name)

for look in keyword_look_in_data_list:

    plot_key(keyword_look_in_data_list,look,file_name,df_2);
plt.figure(figsize=(15,5));

file_name = "noncomm_use_subset"

df_2 = convert_data(file_name)

for look in keyword_look_in_data_list:

    plot_key(keyword_look_in_data_list,look,file_name,df_2);
# plt.figure(figsize=(15,5));

# file_name = "pmc_custom_license"

# df_2 = convert_data(file_name)

# for look in keyword_look_in_data_list:

#     plot_key(keyword_look_in_data_list,look,file_name,df_2);
def prepare_data_for_analysis(df):

    outer_key = 'abstract'

    inner_key = 'cite_spans'

    BIBREF = []



    for rowNumber in range(0,df.shape[0]):

        rowSampleCounts = len(df[outer_key][rowNumber])



        for rowSampleNumber in range(0,rowSampleCounts):        



             if len(df[outer_key][rowNumber][rowSampleNumber][inner_key]) > 0:

                rowSampleCountCounts = len(df[outer_key][rowNumber][rowSampleNumber][inner_key])

                #print(rowSampleCountCounts)



                for rowSampleNumberNumber in range(0,rowSampleCountCounts):

                    try:

                        BIBREF = BIBREF + [df[outer_key][rowNumber][rowSampleNumber][inner_key][rowSampleNumberNumber]['ref_id']]

                    except:

                        pass



    return BIBREF
file_name = "comm_use_subset"

df=conver_json_to_dataframe(file_name)
BIBREF = prepare_data_for_analysis(df)
df_count = pd.DataFrame({"BIBREF":BIBREF}).dropna()

print(df_count.shape)

df_count = df_count.sort_values("BIBREF")

df_count=pd.DataFrame(df_count['BIBREF'].value_counts()).sort_values("BIBREF")

df_count['name'] =df_count.index

df_count['number'] = [int(i.replace("BIBREF","")) for i in df_count['name']]

df_count = df_count.sort_values("number")

del df_count['name']

del df_count['number']
df_count_new = pd.DataFrame()



values       = []



for i in range(0, df_count.shape[0], 10):

    values= values + [sum(df_count['BIBREF'][i:i+10])] 

    

for i in range(0,len(values)):

    df_count_new.loc[i,'BIBREF'] = int(values[i])    

    

    

index_name_list = []

for r in (df_count_new.index+1):

    l = r - 1

    index_name = 'BIBREF ' +str(l*10)+" - "+str(r*10)

    index_name_list = index_name_list + [index_name]

    

df_count_new.index = index_name_list
plt.plot(df_count)
df_count_new.plot(kind='bar')