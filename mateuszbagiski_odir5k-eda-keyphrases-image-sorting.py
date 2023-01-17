# Import everything we need



import os

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt, image as mpimg

from tqdm import tqdm

from time import time

from collections import Counter

import random



import tensorflow as tf

from tensorflow.keras import models, layers, optimizers, utils, callbacks

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

from skimage.transform import resize



import re
# Set up all the paths



train_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Training Images'

test_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K/Testing Images'

main_dir = '../input/ocular-disease-recognition-odir5k/ODIR-5K'

work_dir = '../working'

data = pd.read_excel(os.path.join(main_dir,'data.xlsx'), sheet_name=None)

data = pd.DataFrame(data[list(data.keys())[0]])



data
data_num = data.copy()[['Patient Age', 'Patient Sex', 'N', 'D','G','C','A','H','M','O']]

data_num['Patient Sex'] = data_num['Patient Sex'].apply(lambda x:0  if x=='Female' else 1) # we encode sex: Female => 0; Male => 1



data_num.hist(figsize=(15,12))



plt.show()
data_num_corr = data_num.corr()

data_num_corr
cutoff = .08

data_disease_corr = data_num_corr.iloc[3:,3:].copy() # we cut off everything, except the information about correlation between specific conditions

data_disease_corr = data_disease_corr.applymap(lambda x:np.NaN if abs(x)<cutoff or x==1 else x ).round(2) # we set a cutoff point at .08 and round data to the second decimal point

data_disease_corr
data_diseases = data.iloc[:,-7:].copy() # a slice of data containing only information about diseases occuring in each patient

disease_counter = Counter()

for i, row in data_diseases.iterrows():

    disease_counter[row.sum()]+=1

disease_counter
# Let's create a dictionary to store information about keyphrases used in the diagnosis for each disease and the frequency of their usage

conditions = list(data.columns[-8:])

conditions_keyphrases = {condition: Counter() for condition in conditions}



for i, row in data.iterrows():

    # Keyphrases used in the diagnosis for each eye:

    keyphrases = list(set(row['Left-Diagnostic Keywords'].replace('，',',').split(',') + row['Right-Diagnostic Keywords'].replace('，',',').split(',')))

        # Keyphrases are both split with an ordinary comma (',') and a 'weird' comma ('，'), so we replace all the cases in which the latter occurs with the first one

        #  and only then split the string of keywords into keyphrases. We turn it into a set to eliminate any doubles and then into a list again because we may want to

        #  use some list-specific methods, unavailable for sets. 

    

    # For every condition diagnosed

    for condition in conditions:

        if row[condition]==1:

            # Add 1 to the counter for the keyphrases present

            for keyphrase in keyphrases:

                conditions_keyphrases[condition][keyphrase] += 1



# Sort the dictionary for each condition in the descending order:

def dicsort(d):

    d_items_rev = [ (value, key) for (key, value) in d.items() ]

    d_items_sorted = [ (key, value) for (value, key) in sorted(d_items_rev, reverse=True) ]

    d_sorted = {key: value for (key, value) in d_items_sorted}

    return d_sorted



conditions_keyphrases = {condition: dicsort(conditions_keyphrases[condition]) for condition in conditions_keyphrases.keys()}



conditions_keyphrases
conditions_keyphrases_metadata_1 = {

    'n_keyphrases': {condition: len(conditions_keyphrases[condition]) for condition in conditions_keyphrases.keys()},

    'n_occurences': {condition: np.sum(list(conditions_keyphrases[condition].values())) for condition in conditions_keyphrases.keys()}

}



conditions_keyphrases_metadata_1
conditions_keyphrases = {condition: Counter() for condition in conditions}



for i, row in data.iterrows():

    # If there was only one condition diagnosed, ('N' included) which one is it?

    if row.iloc[-8:].sum()==1:

        condition = row.index[len(row.index)-8 + row.iloc[-8:].astype(np.int32).argmax()]

        # Keyphrases used in the diagnosis for each eye

        keyphrases_L = row['Left-Diagnostic Keywords'].replace('，',',').split(',')

        keyphrases_R = row['Right-Diagnostic Keywords'].replace('，',',').split(',')

        keyphrases = []    

        # If this patient was not diagnosed as healthy

        if condition!='N':

            # If this eye was not diagnosed as healthy, add its diagnostic keyphrases to the list of keyphrases

            if 'normal fundus' not in keyphrases_L:

                keyphrases += keyphrases_L

            if 'normal fundus' not in keyphrases_R:

                keyphrases += keyphrases_R

        # If this patient was diagnosed as healthy, then both of his eyes should be ascribed to category 'N'

        else:

            keyphrases += keyphrases_L + keyphrases_R



        # Add the keyphrases to the conditions_keyphrases dictionary:

        for keyphrase in keyphrases:

            conditions_keyphrases[condition][keyphrase] += 1





conditions_keyphrases = {condition: dicsort(conditions_keyphrases[condition]) for condition in conditions_keyphrases.keys()}



conditions_keyphrases_metadata_2 = {

    'n_keyphrases': {condition: len(conditions_keyphrases[condition]) for condition in conditions_keyphrases.keys()},

    'n_occurences': {condition: np.sum(list(conditions_keyphrases[condition].values())) for condition in conditions_keyphrases.keys()}

}
print('\tInclusive, cross-pollination-prone method:')

print('n_keyphrases:')

print(conditions_keyphrases_metadata_1['n_keyphrases'])

print('n_occurences:')

print(conditions_keyphrases_metadata_1['n_occurences'])



print('\n')



print('\tExclusive, cross-pollination-resistant method:')

print('n_keyphrases:')

print(conditions_keyphrases_metadata_2['n_keyphrases'])

print('n_occurences:')

print(conditions_keyphrases_metadata_2['n_occurences'])
conditions_keyphrases
# A dictionary which will contain a frequency of keyphrases occuring within each condition (i.e. what fraction of cases within each condition had a given keyphrase)

conditions_keyphrases_freq = conditions_keyphrases.copy()

for condition in conditions_keyphrases_freq:

    for keyphrase in conditions_keyphrases_freq[condition]:

        conditions_keyphrases_freq[condition][keyphrase] = 0

        

conditions_counter = Counter() # Counter for number of individual images displaying signs of each condition



for i, row in data.iterrows():

    keyphrases_L = row['Left-Diagnostic Keywords'].replace('，',',').split(',')

    keyphrases_R = row['Right-Diagnostic Keywords'].replace('，',',').split(',')

    for condition in conditions:

        # If at least one eye of this patient has the following condition

        if row[condition]==1:

            # If any keyphrase characteristic of this condition was found in the diagnostic keyphrases for the left eye...

            if any (keyphrase in keyphrases_L for keyphrase in conditions_keyphrases_freq[condition]):

                # ...then we can assume that this eye displays symptoms of this condition. We add 1 to the counter for cases of this condition.

                conditions_counter[condition] += 1

                # For any keyphrase used to describe it...

                for keyphrase in keyphrases_L:

                    # ...we check if it is associated with this condition. If it is, we add 1 to the counter for occurences of keyphrases for this condition.

                    if keyphrase in conditions_keyphrases_freq[condition]:

                        conditions_keyphrases_freq[condition][keyphrase] += 1 

            # We repeat the same for the right eye:

            if any (keyphrase in keyphrases_R for keyphrase in conditions_keyphrases_freq[condition]):

                conditions_counter[condition] += 1

                for keyphrase in keyphrases_R:

                    if keyphrase in conditions_keyphrases_freq[condition]:

                        conditions_keyphrases_freq[condition][keyphrase] += 1

    

# To obtain percentages, we divide each keyphrase count by the number of cases of the condition (i.e. individual eyes displaying characteristic symptoms)

for condition in conditions_keyphrases_freq:

    for keyphrase in conditions_keyphrases_freq[condition]:

        conditions_keyphrases_freq[condition][keyphrase] = np.round(conditions_keyphrases_freq[condition][keyphrase]/conditions_counter[condition], 3)



# Sorting:

conditions_keyphrases_freq = {condition: dicsort(conditions_keyphrases_freq[condition]) for condition in conditions_keyphrases_freq}



conditions_keyphrases_freq
all_keyphrases = []

for condition in conditions:

    all_keyphrases += [keyphrase for keyphrase in conditions_keyphrases_freq[condition] if keyphrase not in all_keyphrases]



keyphrases_from_columns = []

for keyphrases_L, keyphrases_R in zip(data['Left-Diagnostic Keywords'].values, data['Right-Diagnostic Keywords'].values):

    keyphrases_from_columns += [keyphrase for keyphrase in keyphrases_L.replace('，',',').split(',') if keyphrase not in keyphrases_from_columns]

    keyphrases_from_columns += [keyphrase for keyphrase in keyphrases_R.replace('，',',').split(',') if keyphrase not in keyphrases_from_columns]



len(all_keyphrases), len(keyphrases_from_columns)
def make_keyphrases_df():

    keyphrases_df = pd.DataFrame( {keyphrase: {condition: 0 for condition in conditions} for keyphrase in keyphrases_from_columns} ).T

    keyphrases_df['Predictive power'] = 0 # Predictive power - how many occurrences of this keyphrase were associated with the most frequent class

    return keyphrases_df



keyphrases_excl = make_keyphrases_df()

keyphrases_incl = make_keyphrases_df()

keyphrases_incl
# We reinitialize both DataFrames to avoid errors

keyphrases_excl = make_keyphrases_df()

keyphrases_incl = make_keyphrases_df()



for i, row in data.iterrows():

    # 1.

    only_one_condition = bool(row[-8:].sum()==1)

    # 2.

    keyphrases_L = row['Left-Diagnostic Keywords'].replace('，',',').split(',')

    keyphrases_R = row['Right-Diagnostic Keywords'].replace('，',',').split(',')

    for condition in conditions:

        # 3.

        if row[condition]==1:

            # Left eye

            if any(keyphrase in keyphrases_L for keyphrase in conditions_keyphrases[condition]):

                for keyphrase in keyphrases_L:

                    # 3.1.

                    keyphrases_incl.loc[keyphrase, condition] += 1

                    # 3.2.

                    if only_one_condition:

                        keyphrases_excl.loc[keyphrase, condition] += 1

            # Right eye:

            if any(keyphrase in keyphrases_R for keyphrase in conditions_keyphrases[condition]):

                for keyphrase in keyphrases_R:

                    # 3.1.

                    keyphrases_incl.loc[keyphrase, condition] += 1

                    # 3.2.

                    if only_one_condition:

                        keyphrases_excl.loc[keyphrase, condition] += 1

            



keyphrases_excl['Predictive power'] = (keyphrases_excl.max(axis=1) / keyphrases_excl.sum(axis=1)).replace(np.NaN, 0)

keyphrases_incl['Predictive power'] = (keyphrases_incl.max(axis=1) / keyphrases_incl.sum(axis=1)).replace(np.NaN, 0)
keyphrases_excl
keyphrases_specificity = pd.DataFrame(

    np.zeros(shape=(len(all_keyphrases), 6)),

    index = all_keyphrases,

    columns = ['condition (excl.)','specificity (excl.)', 'condition (incl.)', 'specificity (incl.)', 'condition specificity agreement','specificity discrepancy'],

    dtype='object'

)



for keyphrase, row in keyphrases_specificity.iterrows():

    row.iloc[0] = conditions[keyphrases_excl.loc[keyphrase,:].argmax()]

    row.iloc[1] = keyphrases_excl.loc[keyphrase, 'Predictive power']

    row.iloc[2] = conditions[keyphrases_incl.loc[keyphrase,:].argmax()]

    row.iloc[3] = keyphrases_incl.loc[keyphrase, 'Predictive power']

    row.iloc[4] = row.iloc[0]==row.iloc[2]

    row.iloc[5] = row.iloc[1]-row.iloc[3]

keyphrases_specificity
keyphrases_incl.loc['normal fundus']
suspicious_rows = []

for i, row in data.query('G==1 or C==1 or M==1 or O==1').iterrows():

    # 1.

    only_one_condition = bool(row[-8:].sum()==1)

    # 2.

    keyphrases_L = row['Left-Diagnostic Keywords'].replace('，',',').split(',')

    keyphrases_R = row['Right-Diagnostic Keywords'].replace('，',',').split(',')

    if ('normal fundus' in keyphrases_L and len(keyphrases_L)!=1) or ('normal fundus' in keyphrases_R and len(keyphrases_R)!=1):

        suspicious_rows.append(row)

    

    

suspicious_rows = pd.DataFrame(suspicious_rows)

suspicious_rows
for condition in conditions:

    print(condition, 'lens dust' in conditions_keyphrases[condition].keys())
keyphrases_specificity.loc['lens dust']
keyphrases_incl.loc['lens dust']
keyphrases_O = keyphrases_incl.copy().loc[list(conditions_keyphrases['O'].keys())]

keyphrases_O['O-fraction'] = keyphrases_O['O'] / data['O'].sum()

keyphrases_O['non-O'] = 0



for i, row in data.query('O!=1').iterrows():

    keyphrases_L = row['Left-Diagnostic Keywords'].replace('，',',').split(',')

    keyphrases_R = row['Right-Diagnostic Keywords'].replace('，',',').split(',')

    for keyphrase in keyphrases_O.index:

        if keyphrase in keyphrases_L or keyphrase in keyphrases_R:

            keyphrases_O.loc[keyphrase, 'non-O']+=1



keyphrases_O
try:

    keyphrases_O.drop(['lens dust', 'low image quality'], axis=0, inplace=True)

except Exception as e:

    pass

keyphrases_O.sort_values(by='O-fraction', axis=0, inplace=True, ascending=False)

keyphrases_O
keyphrases_O['O-fraction'].sum()
keyphrases_O_chosen = list(keyphrases_O.iloc[:20].index)

print(keyphrases_O.iloc[:20,-2].sum()) # O-fraction is the penultimate column
diagnostic_keyphrases = {

    'N' : ['normal fundus'],

    'D' : ['nonproliferative retinopathy', 'non proliferative retinopathy', 'proliferative retinopathy'],

    'G' : ['glaucoma'],

    'C' : ['cataract'],

    'A' : ['age-related macular degeneration'],

    'H' : ['hypertensive'],

    'M' : ['myopi'],

    'O' : keyphrases_O_chosen

}



# We reinitialize both DataFrames to avoid errors

keyphrases_excl_new = make_keyphrases_df()

keyphrases_incl_new = make_keyphrases_df()



for i, row in data.iterrows():

    # 1.

    only_one_condition = bool(row[-8:].sum()==1)

    # 2.

    keyphrases_L = row['Left-Diagnostic Keywords'].replace('，',',').split(',')

    keyphrases_R = row['Right-Diagnostic Keywords'].replace('，',',').split(',')

    for condition in conditions:

        # 3.

        if row[condition]==1:

            # Left eye

            if any(keyphrase in keyphrases_L for keyphrase in diagnostic_keyphrases[condition]):

                for keyphrase in keyphrases_L:

                    # 3.1.

                    keyphrases_incl_new.loc[keyphrase, condition] += 1

                    # 3.2.

                    if only_one_condition:

                        keyphrases_excl_new.loc[keyphrase, condition] += 1

            # Right eye:

            if any(keyphrase in keyphrases_R for keyphrase in diagnostic_keyphrases[condition]):

                for keyphrase in keyphrases_R:

                    # 3.1.

                    keyphrases_incl_new.loc[keyphrase, condition] += 1

                    # 3.2.

                    if only_one_condition:

                        keyphrases_excl_new.loc[keyphrase, condition] += 1

            



keyphrases_excl_new['Predictive power'] = (keyphrases_excl_new.max(axis=1) / keyphrases_excl_new.sum(axis=1)).replace(np.NaN, 0)

keyphrases_incl_new['Predictive power'] = (keyphrases_incl_new.max(axis=1) / keyphrases_incl_new.sum(axis=1)).replace(np.NaN, 0)
keyphrases_incl_new.loc['normal fundus']
fundi_images = {condition:[] for condition in conditions}



for i, row in data.iterrows():

    image_L = row['Left-Fundus']

    image_R = row['Right-Fundus']

    if row['N']==1:

        fundi_images['N'] += [image_L, image_R]

        continue

    

    # This time there is no need for splitting the keyphrases from strings into lists of keyphrases 

    keyphrases_L = row['Left-Diagnostic Keywords']

    keyphrases_R = row['Right-Diagnostic Keywords']



    diagnosed_conditions = []

    for condition in conditions[1:]:

        if row[condition]==1:

            diagnosed_conditions.append(condition)

            

    if 'normal fundus' in keyphrases_L:

        fundi_images['N'].append(image_L)

        for condition in diagnosed_conditions:

            fundi_images[condition].append(image_R)

        continue

    if 'normal fundus' in keyphrases_R:

        fundi_images['N'].append(image_R)

        for condition in diagnosed_conditions:

            fundi_images[condition].append(image_L)

        continue

    

    for condition in diagnosed_conditions:

        if any(keyphrase in keyphrases_L for keyphrase in diagnostic_keyphrases[condition]):

            fundi_images[condition].append(image_L)

        if any(keyphrase in keyphrases_R for keyphrase in diagnostic_keyphrases[condition]):

            fundi_images[condition].append(image_R)

            
for condition in conditions:

    print(condition, len(fundi_images[condition]), data[condition].sum())
np.sum([len(fundi_images[condition]) for condition in conditions])
fundi_images_rev = {}

for condition in conditions:

    for im in fundi_images[condition]:

        if im not in fundi_images_rev:

            fundi_images_rev[im] = [condition]

        else:

            fundi_images_rev[im] = sorted(fundi_images_rev[im]+[condition])

            

conditions_correlation = pd.DataFrame(np.zeros(shape=(7,7)), columns=conditions[1:], index=conditions[1:], dtype=np.int32)



for im in fundi_images_rev:

    if len(fundi_images_rev[im])>1:

        for i in range(len(fundi_images_rev[im])):

            for ii in range(len(fundi_images_rev[im])):

                if i!=ii:

                    

                    conditions_correlation.loc[fundi_images_rev[im][i],fundi_images_rev[im][ii]] += 1

    

conditions_correlation
print(data.query('D==1 & O==1').index.shape[0])

print(data.query('D==1 & H==1').index.shape[0])

print(data.query('G==1 & O==1').index.shape[0])
print('Total individual images: ', len(fundi_images_rev))
images_per_condition_1 = {}

for condition in conditions:

    images_per_condition_1[condition] = len(fundi_images[condition])

images_per_condition_1
keyphrases_O_full = list(keyphrases_O.index)

diagnostic_keyphrases['O'] = keyphrases_O_full



fundi_images = {condition:[] for condition in conditions}



for i, row in data.iterrows():

    image_L = row['Left-Fundus']

    image_R = row['Right-Fundus']

    if row['N']==1:

        fundi_images['N'] += [image_L, image_R]

        continue

    

    # This time there is no need for splitting the keyphrases from strings into lists of keyphrases 

    keyphrases_L = row['Left-Diagnostic Keywords']

    keyphrases_R = row['Right-Diagnostic Keywords']



    diagnosed_conditions = []

    for condition in conditions[1:]:

        if row[condition]==1:

            diagnosed_conditions.append(condition)

            

    if 'normal fundus' in keyphrases_L:

        fundi_images['N'].append(image_L)

        for condition in diagnosed_conditions:

            fundi_images[condition].append(image_R)

        continue

    if 'normal fundus' in keyphrases_R:

        fundi_images['N'].append(image_R)

        for condition in diagnosed_conditions:

            fundi_images[condition].append(image_L)

        continue

    

    for condition in diagnosed_conditions:

        if any(keyphrase in keyphrases_L for keyphrase in diagnostic_keyphrases[condition]):

            fundi_images[condition].append(image_L)

        if any(keyphrase in keyphrases_R for keyphrase in diagnostic_keyphrases[condition]):

            fundi_images[condition].append(image_R)

            
images_per_condition_2 = {}

for condition in conditions:

    images_per_condition_2[condition] = len(fundi_images[condition])
print(images_per_condition_1)

print(images_per_condition_2)
fundi_images_rev = {}

for condition in conditions:

    for im in fundi_images[condition]:

        if im not in fundi_images_rev:

            fundi_images_rev[im] = [condition]

        else:

            fundi_images_rev[im] = sorted(fundi_images_rev[im]+[condition])

            

conditions_correlation = pd.DataFrame(np.zeros(shape=(7,7)), columns=conditions[1:], index=conditions[1:], dtype=np.int32)



for im in fundi_images_rev:

    if len(fundi_images_rev[im])>1:

        for i in range(len(fundi_images_rev[im])):

            for ii in range(len(fundi_images_rev[im])):

                if i!=ii:

                    conditions_correlation.loc[fundi_images_rev[im][i],fundi_images_rev[im][ii]] += 1



print('Total number of individual images: ', len(fundi_images_rev))

conditions_correlation
# First, let's give these two dictionaries less mouthful handles



con2img = fundi_images

img2con = fundi_images_rev 



imgdata_columns = ['Image', 'Patient Age', 'Patient Sex', *conditions]

imgdata = []



for i, row in data.iterrows():

    image_L = row['Left-Fundus']

    image_R = row['Right-Fundus']

    if image_L in img2con:

        image_conditions = [int(condition in img2con[image_L]) for condition in conditions]

        imgdata.append([image_L, row['Patient Age'], row['Patient Sex'], *image_conditions])

    if image_R in img2con:

        image_conditions = [int(condition in img2con[image_R]) for condition in conditions]

        imgdata.append([image_R, row['Patient Age'], row['Patient Sex'], *image_conditions])



imgdata = pd.DataFrame(imgdata, columns=imgdata_columns)

imgdata['Patient Sex'] = imgdata['Patient Sex'].apply(lambda x:0  if x=='Female' else 1) # encode sex: 'Female'=>0, 'Male'=>1

imgdata