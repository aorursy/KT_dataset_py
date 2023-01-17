import json

import numpy as np 

import os

import pandas as pd 

counter = 0

counter_errors = 0

# for debugging, change counter_max to 10

counter_max = 9999999999



all_data = []



# walk the input files

for dirname, _, filenames in os.walk('/kaggle/input'):

    if counter >= counter_max or counter_errors >= counter_max:

        break

    for filename in filenames:

        if counter >= counter_max:

            break

# for v19 just look in pdf_json folder - no HTML tables in other folders

        if "pdf_json" in dirname:  # debug and "000b0174f992cb326a891f756d4ae5531f2845f7" in filename:

#             print(os.path.join(dirname, filename))

            try:

                with open(os.path.join(dirname, filename), 'rb') as f:

                    each_paper_dict = json.load(f)

#             print(each_paper_dict['ref_entries'])



#find each key: ref_entries, type: table, key: html

                for ref_id, ref_object in each_paper_dict['ref_entries'].items():

                    if "html" in ref_object:

                        counter = counter + 1

#                     print('html table: ' + str(ref_object['text']) + ' found in: ' + os.path.join(dirname, filename))



# read the html code into a pandas df

                        each_data = pd.read_html(ref_object['html'])

# insert some reference columns

                        each_data[0].insert(0, 'Table Title' , str(ref_object['text']))

                        each_data[0].insert(0, 'Table Name' , str(ref_id))

                        each_data[0].insert(0, 'File Path', dirname)

                        each_data[0].insert(0, 'File Name', filename)

# concatenate with prior table data

                        all_data.append(each_data[0])

            except Exception as e:

                print('Error thrown while reading file: ' + os.path.join(dirname, filename) + ' | ' + str(e) )

                counter_errors = counter_errors + 1



all_data = pd.concat(all_data, axis=0, ignore_index=True, sort=False)



# read CORD-19 metadata, explode sha (delimited by ;) onto rows

metadata_df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv", usecols = ['cord_uid', 'sha'], low_memory = False)

metadata_df = metadata_df.assign(sha=metadata_df['sha'].str.split(';')).explode('sha')

metadata_df['sha'] = metadata_df['sha'].str.strip()



# debug

# metadata_df = metadata_df[metadata_df['sha'].str.contains("000b0174f992cb326a891f756d4ae5531f2845f7", na = False)]

# print(metadata_df)



# derive sha from File Name (without extension)

all_data['sha'] = all_data['File Name'].str.replace('.json', '')



combined_df = all_data.merge(metadata_df, how="left", left_on=['sha'], right_on=['sha'])

              

#finish by writing out a concatenated CSV file                      

combined_df.to_csv("CORD-19 HTML table data.csv", index=False)

# print(combined_df)



print("Finished! Found: " + str(counter) + " HTML tables in CORD-19, wrote a CSV file with: " + 

      str(len(all_data.index)) + " rows of table data. Errors found in " + str(counter_errors) + " input files - see above for details" )            
