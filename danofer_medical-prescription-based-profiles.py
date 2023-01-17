import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_json("../input/roam_prescription_based_prediction.jsonl", lines=True)
data.head(20)
data.shape
### not needed, data already parsed as dictionaries from json
# https://stackoverflow.com/questions/38231591/splitting-dictionary-list-inside-a-pandas-column-into-separate-columns
# data['provider_variables'].map(eval)
data.dtypes
provider_data = pd.concat([data.drop(['provider_variables'], axis=1), data['provider_variables'].apply(pd.Series)], axis=1)
# provider_data = pd.DataFrame([v for v in data["provider_variables"]]) ## ORIG - made a new DF
provider_data.shape
provider_data.head()
provider_data.groupby("specialty")["years_practicing"].mean().sort_values(ascending=False)
provider_data.brand_name_rx_count.describe()
provider_data.generic_rx_count.describe()
# provider_data.drop_duplicates(inplace=True)
# provider_data = provider_data.loc[(provider_data.generic_rx_count> 9) & (provider_data.brand_name_rx_count > 9) ]
provider_data = provider_data.loc[(provider_data.generic_rx_count + provider_data.brand_name_rx_count) > 24 ]
provider_data.shape
from collections import Counter

rx_counts = Counter()

for rx_dist in data.cms_prescription_counts:
    rx_counts.update(rx_dist)

rx_series = pd.Series(rx_counts)

rx_series.sort_values(ascending=False)
# def merge_counts(dicts):
#     merged = Counter()
    
#     for d in dicts:
#         merged.update(d)
#     return merged.most_common(20)

# merged_data = pd.concat([data, provider_data], axis=1)

# merged_data.groupby("specialty")["cms_prescription_counts"].apply(merge_counts)
# merged_data.head()
# merged_data.shape
# merged_data.groupby("specialty")["cms_prescription_counts"].apply(merge_counts)["General Practice"]
provider_data["specialty"].value_counts()
provider_data.head()
provider_data["specialty"] = provider_data["specialty"].str.replace("Radiation Oncology","Oncology").str.replace("Medical Oncology","Oncology").str.replace("General Practice","General_Practice")
provider_data["specialty_abbrev"] = provider_data["specialty"].str.split().str.get(0)
provider_data.isna().sum()
provider_data.columns
provider_data[[ 'npi', 'settlement_type', 'generic_rx_count',
       'specialty', 'years_practicing', 'gender', 'region',
       'brand_name_rx_count', 'specialty_abbrev']].nunique()
# get counts of specialities and filter rare ones
provider_data["specialty_abbrev_counts"] = provider_data.groupby("specialty_abbrev")["years_practicing"].transform("count")
provider_data["specialty_abbrev_counts"].hist()
provider_data["specialty_abbrev_counts"].describe()
# filter data for rare specialties!
provider_data = provider_data.loc[provider_data.groupby("specialty")["years_practicing"].transform("count") > 40]
# provider_data = provider_data.loc[provider_data["specialty_abbrev_counts"]>300]
provider_data = provider_data.loc[provider_data.groupby("specialty_abbrev_counts")["years_practicing"].transform("count") > 400]
provider_data.shape
provider_data.groupby("specialty")["years_practicing"].transform("count").describe()
provider_data["specialty"].value_counts()
# many all duplicate rows 
print(provider_data.shape)
provider_data.drop_duplicates(inplace=True,subset=['settlement_type', 'generic_rx_count',
       'specialty', 'years_practicing', 'gender', 'region',
       'brand_name_rx_count'])
print(provider_data.shape)
provider_data.nunique()
provider_data.to_csv("medical_provider_specialtyFilt.csv.gz",index=False,compression="gzip")
