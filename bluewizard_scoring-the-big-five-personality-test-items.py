import numpy as np

import pandas as pd

df = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
df.head()
positively_keyed = ['EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',

                    'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 

                    'EST8', 'EST9', 'EST10',

                    'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',

                    'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10', 

                    'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 

                    'OPN10']



negatively_keyed = ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',

                    'EST2', 'EST4',

                    'AGR1', 'AGR3', 'AGR5', 'AGR7', 

                    'CSN2', 'CSN4', 'CSN6', 'CSN8', 

                    'OPN2', 'OPN4', 'OPN6']
df.loc[:, negatively_keyed] = 6 - df.loc[:, negatively_keyed]
df.head()