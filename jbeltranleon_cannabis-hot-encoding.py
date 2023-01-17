
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

cannabis = pd.read_excel('../input/Cannabis and OCP Values Chemicals.xlsx')
cannabis.head()
from itertools import chain

# list_comprehension
all_labels = np.unique(
                        list(
                            chain(
                                *cannabis['Effects'].map(lambda x: x.split(',')).tolist()
                            )
                        )
                    )
len(all_labels)
all_labels = [x for x in all_labels if len(x)>0]
len(all_labels)
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        # Add a column for each decease
        cannabis[c_label] = cannabis['Effects'].map(lambda finding: 1.0 if c_label in finding else 0)

print(cannabis.shape)
cannabis.head()
# list_comprehension
all_labels = np.unique(
                        list(
                            chain(
                                *cannabis['Flavor'].map(lambda x: x.split(',')).tolist()
                            )
                        )
                    )
len(all_labels)
all_labels = [x for x in all_labels if len(x)>0]
len(all_labels)
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        # Add a column for each decease
        cannabis[c_label] = cannabis['Flavor'].map(lambda finding: 1.0 if c_label in finding else 0)

print(cannabis.shape)
cannabis.head()
cannabis.to_excel('Cannabis and OCP Values Chemicals Hot Encoding.xlsx')
