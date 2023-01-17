import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
supplier_directory_df = pd.read_csv('/kaggle/input/medicaregov-supplier-directory/4e8fb79e-1e30-42f2-b06f-0e4cca44506f', encoding="ISO-8859-1", compression='zip')
# drop immaterial columns (All `NaN` or `F`)

supplier_directory_df.dropna(axis='columns', thresh=5, inplace=True)

supplier_directory_df.drop(columns=['Competitive Bid?'], inplace=True)
# Compute frequency count

svc = supplier_directory_df.loc[:, 'Product Category Name'].value_counts()

suppliers_vc = pd.DataFrame({ 'Product Category Name': svc.index, 'Count': svc.values})
# function to highlight rows

def highlight_rows(val, match_strings_list):

    v = 'background-color: %s; font-weight: %s;' % ('#ff677d', 'bold')

    return [v, v] if val[0] in match_strings_list else ['', '']
vent_supplies = ['CPAP, RADs, & Related Supplies & Accessories', 'Ventilators, Accessories & Supplies']
suppliers_vc.head(100).style.apply(highlight_rows, match_strings_list=vent_supplies, axis=1)