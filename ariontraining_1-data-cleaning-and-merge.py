import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
bill_of_materials_df = pd.read_csv('../input/bill_of_materials.csv')

comp_adaptor_df = pd.read_csv('../input/comp_adaptor.csv')

comp_boss_df = pd.read_csv('../input/comp_boss.csv')

comp_elbow_df = pd.read_csv('../input/comp_elbow.csv')

comp_float_df = pd.read_csv('../input/comp_float.csv')

comp_hfl_df = pd.read_csv('../input/comp_hfl.csv')

comp_nut_df = pd.read_csv('../input/comp_nut.csv')

comp_other_df = pd.read_csv('../input/comp_other.csv')

comp_sleeve_df = pd.read_csv('../input/comp_sleeve.csv')

comp_straight_df = pd.read_csv('../input/comp_straight.csv')

comp_tee_df = pd.read_csv('../input/comp_tee.csv')

comp_threaded_df = pd.read_csv('../input/comp_threaded.csv')

components_df = pd.read_csv('../input/components.csv')

specs_df = pd.read_csv('../input/specs.csv')

test_set_df = pd.read_csv('../input/test_set.csv')

train_set_df = pd.read_csv('../input/train_set.csv')

tube_end_form_df = pd.read_csv('../input/tube_end_form.csv')

tube_df = pd.read_csv('../input/tube.csv')

type_component_df = pd.read_csv('../input/type_component.csv')

type_connection_df = pd.read_csv('../input/type_connection.csv')

type_end_form_df = pd.read_csv('../input/type_end_form.csv')
train_set_df.quote_date = pd.to_datetime(train_set_df.quote_date)
test_set_df.quote_date = pd.to_datetime(test_set_df.quote_date)
#bill_of_materials_df



#how to replace data that has null id but numeric quantity???
components_df.replace("9999", "other", inplace=True)
# replace 9999.0 entries in bend_radius column with np.nan entries

tube_df = tube_df.replace(9999.0, np.nan)

tube_df = tube_df.replace('9999', 'other')

print (tube_df.shape)
tube_end_form_df.head()
bill_of_materials_df.head()
comp_adaptor_df.head()
components_df.head()
type_component_df.head()
type_connection_df.head()
type_end_form_df.head()
#merge1: train + tube_df



merge1 = train_set_df.merge(tube_df)
train_set_df.head()
tube_df.head()
merge1.head()
#merge2: train + tube_df + bill_of_materials_df(bill_of_materials_summary_df)



#The 1,3,5,7...15 columns of bill_comp_types_df are informations about component_id.



#We calculate each tube_assembly uses how many different component for assembly, shown as component_series

bill_comp_types_df = bill_of_materials_df.iloc[:,[1,3,5,7,9,11,13,15]]

bill_comp_types_logical_df = ~bill_comp_types_df.isnull()

component_series = bill_comp_types_logical_df.sum(axis = 1)





#The 2,4,6,8...16 columns of bill_comp_types_df are informations about how many number of 

#components needed for assembly



#Then we calculate the total number of components needed for assembly, shown as quants_series.

bill_comp_quants_df = bill_of_materials_df.iloc[:,[2,4,6,8,10,12,14,16]]

quants_series = bill_comp_quants_df.sum(axis = 1)



bill_of_materials_summary_df = bill_of_materials_df.copy()

bill_of_materials_summary_df['type_totals'] = component_series

bill_of_materials_summary_df['component_totals'] = quants_series



merge2 = merge1.merge(bill_of_materials_summary_df[['tube_assembly_id', 'type_totals', 'component_totals']])
bill_of_materials_summary_df.head()
merge2.head()
#merge3: train + tube_df + bill_of_materials_df(bill_of_materials_summary_df) + specs_df(totals_spec)

specs_only_df = specs_df.iloc[:, 1:11]

specs_logical_df = ~specs_only_df.isnull()

specs_totals = specs_logical_df.sum(axis=1)



specs_with_totals_df = specs_df.copy()

specs_with_totals_df['spec_totals'] = specs_totals



merge3 = merge2.merge(specs_with_totals_df[['tube_assembly_id', 'spec_totals']])
specs_with_totals_df.head()
merge3.head()
merge3.columns
result = merge3.copy()
#result.to_csv("../output/combination.csv", index=False)

result