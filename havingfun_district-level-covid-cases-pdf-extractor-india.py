# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install tabula-py
import tabula
dfs = {}
def process_df(df):
    df = df.copy()
    # Tabula doesn't give best of the results, so had to manually parse it. 
    # Can write a better function for regex parsing by using some other tool.
    prev_state = ""
    prev_affected_district = ""
    rows = []
    for val in df[0].values[2:-2]:
        # Fix for Districts data getting moved to state columns
        if np.all(pd.isnull(val[2:])):
            val[2:] = val[:2]
            val[:2] = np.nan
        elif pd.isnull(val[3]):
            val[2:] = val[1:3]
            val[1] = np.nan

        # Fix for adding info of previous state and count
        if pd.isnull(val[0]):
            val[0] = prev_state
        if pd.isnull(val[1]):
            val[1] = prev_affected_district

        # Fix for removing \r that was splitting on csv creation
        if isinstance(val[2], str):
            val[2] = val[2].replace('\r', '')

        prev_state = val[0]
        prev_affected_district = val[1]
        rows.append(val)

    final_df = pd.DataFrame(rows, columns=["state", "affected_districts", "district", "count"])
    final_df = final_df[~final_df["district"].isnull()]
    final_df = final_df[~final_df["count"].isnull()]
    return final_df
def process_district_pdf(file, key):
    df = tabula.read_pdf(file, output_format='dataframe', pages='all', pandas_options={'header': None, 'names': ["state", "affected_districts", "district", "count"]})
    processed_df = process_df(df)
    dfs[key] = processed_df
    print(processed_df.shape)
    processed_df.to_csv(f'{key}.csv', index=False)
files = {
    "district_324": "https://www.mohfw.gov.in/pdf/DistrictWiseList324.pdf",
    "district_354": "https://www.mohfw.gov.in/pdf/DistrictWiseList354.pdf",
    "district_408": "https://www.mohfw.gov.in/pdf/Districtreporting408.pdf"
}
for key, file in files.items():
    process_district_pdf(file, key)
dfs["district_408"].groupby('state').count().reset_index().sort_values(['state'], ascending=[1])
test_df = tabula.read_pdf("https://www.mohfw.gov.in/pdf/Districtreporting408.pdf", lattice=True, pages='all', output_format='dataframe')
np.NaN
vals = []
for rows in test_df[0].values:
    arr = []
    for val in rows:
        if not pd.isna(val):
            arr.append(val)
    arr = [np.NaN]*(4-len(arr)) + arr
    vals.append(arr)
dff = pd.DataFrame(vals[1:-2], columns=["state", "affected_districts", "district", "count"])

for value in dff.values:
    print(value)
def process_df2(df):
    df = df.copy()
    # Tabula doesn't give best of the results, so had to manually parse it. 
    # Can write a better function for regex parsing by using some other tool.
    prev_state = ""
    prev_affected_district = ""
    rows = []
    for val in df.values:
        if pd.isnull(val[0]):
            val[0] = prev_state
        if pd.isnull(val[1]):
            val[1] = prev_affected_district

        
        # Fix for removing \r that was splitting on csv creation
        if isinstance(val[0], str):
            val[0] = val[0].replace('\r', '')
        if isinstance(val[2], str):
            val[2] = val[2].replace('\r', '')
            

        prev_state = val[0]
        prev_affected_district = val[1]
        rows.append(val)

    final_df = pd.DataFrame(rows, columns=["state", "affected_districts", "district", "count"])
    final_df = final_df[~final_df["district"].isnull()]
    final_df = final_df[~final_df["count"].isnull()]
    return final_df
df_p = process_df2(dff)
df_p.drop_duplicates()
df_p[df_p["state"] == "ASSAM"]
df_p.groupby('state').count().reset_index().sort_values(['state'], ascending=[1])