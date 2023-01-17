import re
import pandas as pd
import numpy as np
from iso3166 import countries
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import strip_numeric, strip_non_alphanum
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# The data used in this can be found on https://www.kaggle.com/andresionek/data-jobs-listings-glassdoor

glass = pd.read_csv("/kaggle/input/data-jobs-listings-glassdoor/glassdoor.csv")
country_codes = pd.read_csv("/kaggle/input/data-jobs-listings-glassdoor/country_names_2_digit_codes.csv", index_col=1)
print(glass.shape)
glass.head()
print(glass.columns.values)
print(list(country_codes.index))
country_list = glass["map.country"].dropna()
country_list.unique()
clist = []
for item in country_list:
    temp = item.split(" - ")
    temp = temp[0].split(" [")
    t = re.sub(r"[^a-zA-Z]+",' ',temp[0])
    t = t.strip()
    clist.append(t)

pd.Series(clist).unique()
country_list_new = pd.Series([country_codes.loc[x,"Name"] if x in country_codes.index else x for x in clist])
country_list_new.dropna()
print(country_list_new.unique())
def rename(country):
    try:
        return countries.get(country).name
    except:
        return (np.nan)

country_list_new = pd.Series(clist)
rows_before = country_list_new.shape[0]
country_list_new = country_list_new.apply(rename)
country_list_new = country_list_new.dropna()
rows_after = country_list_new.shape[0]
print("Samples after: {}".format(rows_before))
print("Samples earlier: {}".format(rows_after))
print("Lost {} samples after converting which is {:.2f}% of the data\n".format((rows_before-rows_after),
                                                                    ((rows_before-rows_after)/rows_before)*100))

print(country_list_new.unique())
glass_country = pd.DataFrame(data = [country_list_new.value_counts().index, country_list_new.value_counts().values], index=["country","count"]).T
glass_country["count"] = pd.to_numeric(glass_country["count"])
glass_country.set_index("country", drop=True, inplace=True)

top_x = 20
glass_country_top = glass_country.head(top_x)
glass_country_top
#plt.figsize=(12,6)
plt.bar(glass_country_top.index,glass_country_top["count"])
plt.xticks(rotation=90)
plt.gcf().set_size_inches(10,6)
plt.show()