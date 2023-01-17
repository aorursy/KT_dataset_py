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
!git clone "https://github.com/pursh2002/analyzing_NYC_high_school_data.git"
ls
import os
os.chdir('/kaggle/working/analyzing_NYC_high_school_data/')
import pandas as pd 
data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]
data = {}
for f in data_files:
    d = pd.read_csv("schools/{0}".format(f))
    key_name = f.replace(".csv","")
    data[key_name] = d
data['sat_results'].head()
for k in data:
    print(data[k].head())
all_survey = pd.read_csv('schools/survey_all.txt',delimiter="\t",encoding="windows-1252")
d75_survey = pd.read_csv('schools/survey_d75.txt',delimiter="\t",encoding="windows-1252")

survey = pd.concat([all_survey,d75_survey],axis=0)

survey.head()
survey["DBN"] = survey["dbn"]
survey_fields = [
    "DBN", 
    "rr_s", 
    "rr_t", 
    "rr_p", 
    "N_s", 
    "N_t", 
    "N_p", 
    "saf_p_11", 
    "com_p_11", 
    "eng_p_11", 
    "aca_p_11", 
    "saf_t_11", 
    "com_t_11", 
    "eng_t_11", 
    "aca_t_11", 
    "saf_s_11", 
    "com_s_11", 
    "eng_s_11", 
    "aca_s_11", 
    "saf_tot_11", 
    "com_tot_11", 
    "eng_tot_11", 
    "aca_tot_11",
]
# Filter survey so it only contains the columns we listed above. You can do this using pandas.DataFrame.loc[].
survey = survey.loc[:,survey_fields]
# Assign the dataframe survey to the key survey in the dictionary data.
data["survey"] = survey

print(survey.head())
data['class_size'].head()
data['sat_results'].head()
# copy the dbn column in hs_directory into a new column called DBN.
data['hs_directory']['DBN']= data['hs_directory']['dbn']

# zfill() pads string on the left with zeros to fill width
def pad_csd(num):
    return str(num).zfill(2)

# Create a new column called padded_csd in the class_size data set
# Use the pandas.Series.apply() method along with a custom function to generate this column.
data['class_size']['padded_csd'] = data["class_size"]["CSD"].apply(pad_csd)
# Use the addition operator (+) along with the padded_csd and SCHOOL CODE columns of class_size, then assign the result to the DBN column of class_size
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]
print(data["class_size"].head())
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']

for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c],errors="coerce")
    
data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] +data['sat_results'][cols[2]] 
print(data['sat_results']['sat_score'].head())
cols = ['SAT Math Avg. Score','SAT Critical Reading Avg. Score','SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c],errors="coerce")
data["sat_results"]['sat_score']=data['sat_results'][cols[0]]+data['sat_results'][cols[1]]+data['sat_results'][cols[2]]
print(data['sat_results']['sat_score'].head())
import re
# function that:Takes in a string
def find_lat(loc):
    # Uses the regular expression above to extract the coordinates
    coords = re.findall("\(.+\)", loc)
    # Uses string manipulation functions to pull out the latitude
    lat = coords[0].split(",")[0].replace("(", "")
    # Returns the latitude
    return lat
# Use the Series.apply() method to apply the function across the Location 1 column of hs_directory. Assign the result to the lat column of hs_directory.
data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)

print(data["hs_directory"].head())
import re
def find_lon(loc):
    coords = re.findall("\(.+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon

data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")

print(data["hs_directory"].head())