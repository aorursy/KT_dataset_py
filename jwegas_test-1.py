import numpy as np
import pandas as pd
import os
import re
import nltk
from __future__ import division
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('english')
df_attr = pd.read_csv('../input/attributes.csv')
df_attr['IsFuture'] = df_attr['name'].apply(lambda x: 0 if isinstance(x, str) and re.match('Bullet+', x) != None else 1)
df_feautures = df_attr[df_attr['IsFuture'] == 1][['product_uid', 'name', 'value']]

df_feautures['name'] = df_feautures['name'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df_feautures['value'] = df_feautures['value'].apply(lambda x: x.lower() if isinstance(x, str) else x)
# Dimension - size
re_dimension_name = r'\((in|ft).?\)|gauge'
re_dimension_value = r'[0-9]+.+(in|ft).?'

# Weight
re_weight_name = r'\((lb).?\)'
re_weight_value = r'[0-9]+.+(lb).?'

# Color
re_color = r'color+'

# Material
re_material = r'material'

# Indoor / Outdoor
re_inoutdoor = r'indoor\/outdoor'

#length+|weight+|height+|width+|depth+|
# DIMENSION
df_feautures['IsDimension'] = df_feautures[['name','value']].apply(lambda x:\
                                            1 if   isinstance(x[0], str) and re.search(re_dimension_name, x[0]) != None\
                                                or isinstance(x[1], str) and re.search(re_dimension_value, x[1]) != None\
                                            else 0, axis=1)
print ("IsDimension is calculated!")

# WEIGHT
df_feautures['IsWeight'] = df_feautures[['name','value']].apply(lambda x:\
                                            1 if   isinstance(x[0], str) and re.search(re_weight_name, x[0]) != None\
                                                or isinstance(x[1], str) and re.search(re_weight_value, x[1]) != None\
                                            else 0, axis=1)
print ("IsWeight is calculated!")

# COLOR
df_feautures['IsColor'] = df_feautures['name'].apply(lambda x:\
                                                1 if   isinstance(x, str) and re.search(re_color, x) != None\
                                                else 0)
print ("IsColor is calculated!")

# MATERIAL
df_feautures['IsMaterial'] = df_feautures['name'].apply(lambda x:\
                                                    1 if   isinstance(x, str) and re.search(re_material, x) != None\
                                                    else 0)
print ("IsMaterial is calculated!")

# INDOOR / OUTDOOR
df_feautures['IsInOutDoor'] = df_feautures['name'].apply(lambda x:\
                                                    1 if   isinstance(x, str) and re.search(re_inoutdoor, x) != None\
                                                    else 0)
print ("IsInOutDoor is calculated!")

