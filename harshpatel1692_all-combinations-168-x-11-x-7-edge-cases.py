# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import sys

import itertools

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
class_map = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

train_dataset = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')
train_dataset.head()
class_map.head()
# Breaking down a grapheme

# Result corresponds to individual components



print(list('র্যো'))



print(list('র্অ্য')) 



print(list('ত্ব'))



# Character '্' acts as a binder to join root with consonant/vowel
# Collecting mappings of components

binder = '্'

root_mapper = dict(zip(class_map[class_map.component_type=='grapheme_root'].label.to_list(), class_map[class_map.component_type=='grapheme_root'].component.to_list()))

vowel_mapper = dict(zip(class_map[class_map.component_type=='vowel_diacritic'].label.to_list(), class_map[class_map.component_type=='vowel_diacritic'].component.to_list()))

consonant_mapper = dict(zip(class_map[class_map.component_type=='consonant_diacritic'].label.to_list(), class_map[class_map.component_type=='consonant_diacritic'].component.to_list()))

cover_edge_cases = True
# Abra Kadabra

final_output = []

for combination in list(itertools.product(range(168),range(11), range(7))): # create tuple of every combinations and iterate over

    output = ''

    root = combination[0]

    vowel = combination[1]

    consonant = combination[2]

    

    root_label = root_mapper[root]

    vowel_label = vowel_mapper[vowel]

    consonant_label = consonant_mapper[consonant]

    

    # Ignore root with label 0 and 1

    if root in [0, 1] and vowel==0 and consonant==0:

        output = root_label

        

    elif root in [0, 1] and (vowel!=0 or consonant!=0):

        pass

    

    elif root != 0:

        if consonant == 0 and vowel==0:

            output = list(root_label)

            output = ''.join(output)

            

        elif consonant ==  0 and vowel!=0:

            output = list(root_label) + list(vowel_label)

            output = ''.join(output)

            

        elif consonant in [1,2,3]:   

            if consonant==1 and vowel==0:  #High priority on right. preceedes vowel position

                output = list(root_label) + list(consonant_label)

                output = ''.join(output)

            elif consonant==1 and vowel!=0:  #High priority on right. preceedes vowel position

                output = list(root_label)+ list(vowel_label) + list(consonant_label)

                output = ''.join(output)



            elif consonant==2 and vowel==0: #High priority on left.

                output = list(consonant_label) + list(root_label)

                output = ''.join(output)

                

                #Edge Case Handling regarding consonant==2 র‍্যা র্দ্র র্ত্রী র্ত্রে

                if cover_edge_cases:

                    final_output.append([root, vowel, consonant, output])

                    output = list(consonant_label) + list(root_label) + [binder] + list(consonant_label[0])

                    output = ''.join(output)

                



            elif consonant==2 and vowel!=0: #High priority on left.

                output = list(consonant_label) + list(root_label) + list(vowel_label)

                output = ''.join(output)

                

                #Edge Case Handling regarding consonant==2 র‍্যা র্দ্র র্ত্রী র্ত্রে

                if cover_edge_cases:

                    final_output.append([root, vowel, consonant, output])

                    output = list(consonant_label) + list(root_label) + [binder] + list(consonant_label[0]) + list(vowel_label)

                    output = ''.join(output)



            elif consonant==3 and vowel==0: #Split priority Left and Right rest in middle.

                output = list(consonant_label[0])+ [binder] + list(root_label) + [binder] + list(consonant_label[-1])

                output = ''.join(output)    



            elif consonant==3 and vowel!=0: #Split priority Left and Right rest in middle.

                output = list(consonant_label[0])+ [binder] + list(root_label) + [binder] + list(consonant_label[-1]) + list(vowel_label)

                output = ''.join(output)

                

        elif consonant in [4,5,6]:  

            if consonant in [4, 5, 6] and vowel==0: #Soft Right priority

                output = list(root_label) + list(consonant_label)

                output = ''.join(output) 



            elif consonant in [4, 5, 6] and vowel!=0: #Soft Right priority

                output = list(root_label) + list(consonant_label) + list(vowel_label)

                output = ''.join(output) 

            

    else:

        print(combination)

        break

        

    if output:

        final_output.append([root, vowel, consonant, output])
full_dataset = pd.DataFrame(final_output, columns=['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme'])

full_dataset['image_id'] = full_dataset.index

full_dataset['image_id'] = 'Train_' + full_dataset['image_id'].astype(str)

cols = full_dataset.columns.tolist()

cols = cols[-1:] + cols[:-1]

full_dataset = full_dataset[cols]
full_dataset.head()
len(full_dataset) # 168 x 11 x 7 = 12784 + 1826 (Edge Cases)
for char in train_dataset.grapheme.unique():

    if char not in full_dataset.grapheme.tolist():

        print('Missing Grapheme: ', char)

        print('Unicode Components: ',list(char))
full_dataset.to_csv('BengaliAllCombinationsGrapheme.csv', index=None)