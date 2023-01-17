import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#cont = pd.read_csv("https://raw.githubusercontent.com/edlich/eternalrepo/master/DS-WAHLFACH/dsm-beuth-edl-demodata-dirty.csv")
cont = pd.read_csv("../input/dsm-beuth-edl-demodata-dirty1.csv")
cont
nd = cont.dropna(axis=0, thresh=1) # drop rows that has more then one NaN
nd = nd.drop(['id', 'full_name'], axis=1) # get rid of ID axis as not needed and sparse; get rid off full name as redundant and non atomic
nd = nd.drop_duplicates() # get rid of duplicates
nd['age'] = nd['age'].replace( 'old', pd.to_numeric(nd['age'], errors='coerce').max() ) # replace columnes with 'old' with max age
nd['age'] = pd.to_numeric(nd['age'], errors='coerce').abs() # now make negative values positive

# keep empty email and gender around
nd = nd.fillna('') # replace NaN with empty string

nd