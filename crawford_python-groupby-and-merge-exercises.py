import pandas as pd



#Squelch SettingWithCopyWarning

pd.options.mode.chained_assignment = None



# Load dataframe

bill_of_materials = pd.read_csv('../input/bill_of_materials.csv')

bill_of_materials.head()



# Function to get component data

def pull_out_component(raw, component_num):

    return pd.DataFrame({'tube_assembly_id': raw.tube_assembly_id,

                         'component_id': raw['component_id_' + str(component_num)],

                         'component_count': raw['quantity_' + str(component_num)]})



### Component counts ###

component_counts = pd.concat((pull_out_component(bill_of_materials, i) for i in range(1, 9)), axis=0)

component_counts.dropna(axis=0, inplace=True)



# List of component files

files = ['comp_adaptor.csv', 'comp_boss.csv', 'comp_elbow.csv', 'comp_float.csv',

         'comp_hfl.csv', 'comp_nut.csv', 'comp_other.csv', 'comp_sleeve.csv',

         'comp_straight.csv', 'comp_tee.csv', 'comp_threaded.csv']



# Combine all component data into one dataframe

all_component_data = pd.concat([pd.read_csv('../input/'+f) for f in files], axis=0)



### Component weights ###

component_weights = all_component_data[['component_id', 'weight']]

component_weights.fillna(0, inplace=True)
# Find the five heaviest tube assembly id's and their weights
# How many tube_assembly_id's have more than five components? (sum of component_count)
# How many component_id's are used in more than 50 tube assemblies?
# What is the average weight of the five heaviest component_id's?