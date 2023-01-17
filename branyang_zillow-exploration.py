

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
prop = pd.read_csv('../input/properties_2016.csv',

                   dtype={'hashottuborspa': bool,

                          'propertycountylandusecode': str,

                          'propertyzoningdesc': str,

                          'fireplaceflag': bool,

                          'taxdelinquencyflag': bool

                   },

                   true_values=['True', 'Y'])
prop.describe()
expl = pd.read_excel('../input/zillow_data_dictionary.xlsx')

print(expl)
prop[['hashottuborspa', 'fireplaceflag', 'taxdelinquencyflag']].describe()
prop.columns[[32,34]]