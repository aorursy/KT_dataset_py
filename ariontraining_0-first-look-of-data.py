import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



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
train_set_df.head()
train_set_df.info()
train_set_df.quote_date = pd.to_datetime(train_set_df.quote_date)
print (train_set_df.quote_date.sort_values())
test_set_df.head()
test_set_df.info()
# test_set_df



#id: (integer) observation index

#tube_assembly_id: (string) ID

#supplier: (string) ID

#quote_date: (string) when price was quoted by supplier 

#annual_usage: (int) estimate of how many tube assemblies will be purchased in a given year

#min_order_quantity: (integer) number of assemblies required, at minimum, for non-bracket pricing

#bracket_pricing: (string) does this assembly have bracket-pricing or not?

#quantity: (integer) how many assemblies are sought for purchase?
test_set_df.quote_date = pd.to_datetime(test_set_df.quote_date)
print (test_set_df.quote_date.sort_values())
train_id_unique = set(train_set_df.tube_assembly_id.unique())

test_id_unique = set(test_set_df.tube_assembly_id.unique())



train_id_unique & test_id_unique

#There is no common tube_assembly_id between train and test data.
train_supplier_unique = set(train_set_df.supplier.unique())

test_supplier_unique = set(test_set_df.supplier.unique())



print ("Train", len(train_supplier_unique))

print ("Test", len(test_supplier_unique))

print ("Common", len(train_supplier_unique & test_supplier_unique))

#There is 45 common supplier between train and test data.
#bill_of_materials_df.head().transpose()

bill_of_materials_df.head()
# bill_of_materials variables



# tube_assembly_id: ID

# component_id_1: ID

# quantity_1: number of components needed for assembly (conceptually integers, practically floats)

# component_id_2: ID

# quantity_2: number of components needed for assembly (conceptually integers, practically floats)

# component_id_3: ID

# quantity_3: number of components needed for assembly (conceptually integers, practically floats)

# etc.
bill_of_materials_df.info()
#The 1,3,5,7...15 columns of bill_comp_types_df are informations about component_id.



#We calculate each tube_assembly uses how many different component for assembly 

#Shown as component_series, then use a graph to present the information.



bill_comp_types_df = bill_of_materials_df.iloc[:,[1,3,5,7,9,11,13,15]] 

bill_comp_types_logical_df = ~bill_comp_types_df.isnull()

component_series = bill_comp_types_logical_df.sum(axis = 1)



plt.figure(figsize=(16, 6))

sns.countplot(component_series)

# almost half of all tube assemblies have exactly 2 types of components
#The 2,4,6,8...16 columns of bill_comp_types_df are informations about how many number of 

#components needed for assembly



#Then we calculate the total number of components needed for assembly, present the information by a graph.



bill_comp_quants_df = bill_of_materials_df.iloc[:,[2,4,6,8,10,12,14,16]]

quants_series = bill_comp_quants_df.sum(axis = 1)

quants_series = quants_series.astype(int)



plt.figure(figsize=(16, 6))

sns.countplot(quants_series)
comp_adaptor_df.head()
# comp_[type] variables: 



# 1. we're not sure what each column physically refers to or the length units.

# 2. 9999 could be another NaN



# There appear to be 11 basic component types, with further broken down subtypes. Each type has type-specific

# variables that may not apply to other component types.
comp_adaptor_df.info()
components_df.head()
components_df.info()
specs_df.head()
# specs_df



# list of specifications for components of each tube assembly (may be 0-10)
specs_df.info()
#calculate the number of specs needed for assembly of each tube assembly 



#specs_only_df: information about specs

#specs_totals: the number of specs needed for assembly 



specs_only_df = specs_df.iloc[:, 1:11]

specs_logical_df = ~specs_only_df.isnull()

specs_totals = specs_logical_df.sum(axis=1)



plt.figure(figsize=(16, 6))

sns.countplot(specs_totals)
tube_end_form_df.head()
# tube_end_form_df

# 9999 might be 'other' category



# end_form_id: (string) ID

# forming: (string) forming or not
tube_df.head()
# tube_df



#tube_assembly_id: (string) ID

#material_id: (string) material specification for the tube

#diameter: (float) tube diameter

#wall: (float) wall thickness

#length: (float) tube length

#num_bends: (integer) number of bends in tube

#bend_radius: (float) (guess) radius of all bends in tube

#end_a_1x: (string) length of end a is less than 1x diameter

#end_a_2x: (string) length of end a is less than 2x diameter

#end_x_1x: (string) length of end x is less than 1x diameter

#end_x_2x: (string) length of end x is less than 2x diameter

#end_a: (string) end form of end a

#end_x: (string) end form of end x

#num_boss: (integer) number of bosses

#num_bracket: (integer) number of brackets

#other: (integer) other tube prep required steps
type_component_df.head()
# type_component_df



# component_type_id: (string) ID

# name: (string) defines how component is attached to tube
type_connection_df.head()
# type_connection_df

# refer only to adaptors



# connection_type_id: (string) ID

# name: (string) connection type description
type_end_form_df
# type_end_form_df



#end_form_id = (string) ID

#name = (string) end form description