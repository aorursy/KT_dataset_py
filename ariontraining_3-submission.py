import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import time



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
test_set_df.head()
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()
train_set_df.quote_date = pd.to_datetime(train_set_df.quote_date)
test_set_df.quote_date = pd.to_datetime(test_set_df.quote_date)
#bill_of_materials_df



#how to replace data that has null id but numeric quantity???
components_df.component_id.replace("9999", "other", inplace=True)
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



test_merge1 = test_set_df.merge(tube_df)
#merge2: train + tube_df + bill_of_materials_df(bill_of_materials_summary_df)

bill_comp_types_df = bill_of_materials_df.iloc[:,[1,3,5,7,9,11,13,15]]

bill_comp_types_logical_df = ~bill_comp_types_df.isnull()

component_series = bill_comp_types_logical_df.sum(axis = 1)



bill_comp_quants_df = bill_of_materials_df.iloc[:,[2,4,6,8,10,12,14,16]]

quants_series = bill_comp_quants_df.sum(axis = 1)



bill_of_materials_summary_df = bill_of_materials_df.copy()

bill_of_materials_summary_df['type_totals'] = component_series

bill_of_materials_summary_df['component_totals'] = quants_series



merge2 = merge1.merge(bill_of_materials_summary_df[['tube_assembly_id', 'type_totals', 'component_totals']])



test_merge2 = test_merge1.merge(bill_of_materials_summary_df[['tube_assembly_id', 'type_totals', 'component_totals']])
#merge3: train + tube_df + bill_of_materials_df(bill_of_materials_summary_df) + specs_df(totals_spec)

specs_only_df = specs_df.iloc[:, 1:11]

specs_logical_df = ~specs_only_df.isnull()

specs_totals = specs_logical_df.sum(axis=1)



specs_with_totals_df = specs_df.copy()

specs_with_totals_df['spec_totals'] = specs_totals



merge3 = merge2.merge(specs_with_totals_df[['tube_assembly_id', 'spec_totals']])



test_merge3 = test_merge2.merge(specs_with_totals_df[['tube_assembly_id', 'spec_totals']])
merge3.head()
test_merge3.head()
train = merge3.copy()



test = test_merge3.copy()
train.drop("tube_assembly_id", axis=1, inplace=True)



test.drop("tube_assembly_id", axis=1, inplace=True)

test.drop("id", axis=1, inplace=True)
train.head().transpose()
test.head().transpose()
train.quote_date = pd.to_datetime(train.quote_date)



test.quote_date = pd.to_datetime(train.quote_date)
train["year"] = train.quote_date.dt.year

train["month"] = train.quote_date.dt.month

train["day"] = train.quote_date.dt.day

train["day_of_week"] = train.quote_date.dt.dayofweek



test["year"] = test.quote_date.dt.year

test["month"] = test.quote_date.dt.month

test["day"] = test.quote_date.dt.day

test["day_of_week"] = test.quote_date.dt.dayofweek
#only use numeric data

data = train.select_dtypes(include=['int', 'float'])



test = test.select_dtypes(include=["int", "float"])
#fill null by 0

data.replace(np.nan, 0, inplace=True)



test.replace(np.nan, 0, inplace=True)
train_data, valid_data = train_test_split(data, test_size = 0)
label = "cost"
data_labels = train_data.columns.tolist()

data_labels.remove(label)
train_df = train_data[data_labels]

valid_df = valid_data[data_labels]

train_label = train_data[label]

valid_label = valid_data[label]
train_df.shape
test.shape
# sklearn random forest regression

from sklearn.ensemble import RandomForestRegressor



def rf_learning(labels, train, test):

    label_log=np.log1p(labels)

    clf=RandomForestRegressor(n_estimators=50, n_jobs=-1)

    model=clf.fit(train, label_log)

    preds1=model.predict(test)

    preds=np.expm1(preds1)

    return  preds
rf_preds = rf_learning(train_label, train_df, test)

sample_submission.cost = rf_preds
#sample_submission.to_csv("../output/submission.csv", index=False)

sample_submission
#LB with 100% train data is 970/1323 0.354972/0.196556