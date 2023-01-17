import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
ds = pd.read_csv('../input/mushrooms.csv')
def convert_class_to_num(c):

    if c == 'p':

        return 0

    elif c == 'e':

        return 1

    else:

        return c



def convert_cap_shape_to_num(cs):

    if cs == 'b':

        return 0

    elif cs == 'c':

        return 1

    elif cs == 'x':

        return 2

    elif cs == 'f':

        return 3

    elif cs == 'k':

        return 4

    elif cs == 's':

        return 5

    else:

        return cs

    

def convert_cap_surface_to_num(cs):

    if cs == 'f':

        return 0

    elif cs == 'g':

        return 1

    elif cs == 'y':

        return 2

    elif cs == 's':

        return 3

    else:

        return cs

    

def convert_bruises_to_num(cs):

    if cs == 'f':

        return 0

    elif cs == 't':

        return 1

    else:

        return cs

    

def convert_cap_color_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'b':

        return 1

    elif cs == 'c':

        return 2

    elif cs == 'g':

        return 3

    elif cs == 'r':

        return 4

    elif cs == 'p':

        return 5

    elif cs == 'u':

        return 6

    elif cs == 'e':

        return 7

    elif cs == 'w':

        return 8

    elif cs == 'y':

        return 9

    else:

        return cs

    

def convert_odor_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'a':

        return 1

    elif cs == 'l':

        return 2

    elif cs == 'c':

        return 3

    elif cs == 'y':

        return 4

    elif cs == 'f':

        return 5

    elif cs == 'm':

        return 6

    elif cs == 'p':

        return 7

    elif cs == 's':

        return 8

    else:

        return cs

    

def convert_gill_attachment_to_num(cs):

    if cs == 'a':

        return 0

    elif cs == 'd':

        return 1

    elif cs == 'f':

        return 2

    elif cs == 'n':

        return 3

    else:

        return cs



def convert_gill_spacing_to_num(cs):

    if cs == 'c':

        return 0

    elif cs == 'w':

        return 1

    elif cs == 'd':

        return 2

    else:

        return cs



def convert_gill_size_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'b':

        return 1

    else:

        return cs    



def convert_gill_color_to_num(cs):

    if cs == 'k':

        return 0

    elif cs == 'n':

        return 1

    elif cs == 'b':

        return 2

    elif cs == 'h':

        return 3

    elif cs == 'g':

        return 4

    elif cs == 'r':

        return 5

    elif cs == 'o':

        return 6

    elif cs == 'p':

        return 7

    elif cs == 'u':

        return 8

    elif cs == 'e':

        return 9

    elif cs == 'w':

        return 10

    elif cs == 'y':

        return 11

    else:

        return cs

    

def convert_stalk_shape_to_num(cs):

    if cs == 't':

        return 0

    elif cs == 'e':

        return 1

    else:

        return cs  

    

def convert_stalk_root_to_num(cs):

    if cs == '?':

        return 0

    elif cs == 'b':

        return 1

    elif cs == 'c':

        return 2

    elif cs == 'u':

        return 3

    elif cs == 'e':

        return 4

    elif cs == 'z':

        return 5

    elif cs == 'r':

        return 6

    else:

        return cs



def convert_stalk_surface_to_num(cs):

    if cs == 'f':

        return 0

    elif cs == 'y':

        return 1

    elif cs == 'k':

        return 2

    elif cs == 's':

        return 3

    else:

        return cs

    

def convert_stalk_color_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'b':

        return 1

    elif cs == 'c':

        return 2

    elif cs == 'g':

        return 3

    elif cs == 'o':

        return 4

    elif cs == 'p':

        return 5

    elif cs == 'e':

        return 6

    elif cs == 'w':

        return 7

    elif cs == 'y':

        return 8

    else:

        return cs

    

def convert_veil_type_to_num(cs):

    if cs == 'p':

        return 0

    elif cs == 'u':

        return 1

    else:

        return cs



def convert_veil_color_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'o':

        return 1

    elif cs == 'w':

        return 2

    elif cs == 'y':

        return 3

    else:

        return cs



def convert_ring_num_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'o':

        return 1

    elif cs == 't':

        return 2

    else:

        return cs



def convert_ring_type_to_num(cs):

    if cs == 'n':

        return 0

    elif cs == 'c':

        return 1

    elif cs == 'e':

        return 2

    elif cs == 'f':

        return 3

    elif cs == 'l':

        return 4

    elif cs == 'p':

        return 5

    elif cs == 's':

        return 6

    elif cs == 'z':

        return 7

    else:

        return cs

    

def convert_spore_print_color_to_num(cs):

    if cs == 'k':

        return 0

    elif cs == 'n':

        return 1

    elif cs == 'b':

        return 2

    elif cs == 'h':

        return 3

    elif cs == 'r':

        return 4

    elif cs == 'o':

        return 5

    elif cs == 'u':

        return 6

    elif cs == 'w':

        return 7

    elif cs == 'y':

        return 8

    else:

        return cs

    

def convert_population_to_num(cs):

    if cs == 'a':

        return 0

    elif cs == 'c':

        return 1

    elif cs == 'n':

        return 2

    elif cs == 's':

        return 3

    elif cs == 'v':

        return 4

    elif cs == 'y':

        return 5

    else:

        return cs

    

def convert_habitat_to_num(cs):

    if cs == 'g':

        return 0

    elif cs == 'l':

        return 1

    elif cs == 'm':

        return 2

    elif cs == 'p':

        return 3

    elif cs == 'u':

        return 4

    elif cs == 'w':

        return 5

    elif cs == 'd':

        return 6

    else:

        return cs

    

def processing_data(ds):

    ds['class'] = ds['class'].map(convert_class_to_num)

    ds['cap-shape'] = ds['cap-shape'].map(convert_cap_shape_to_num)

    ds['cap-surface'] = ds['cap-surface'].map(convert_cap_surface_to_num)

    ds['bruises'] = ds['bruises'].map(convert_bruises_to_num)

    ds['cap-color'] = ds['cap-color'].map(convert_cap_color_to_num)

    ds['odor'] = ds['odor'].map(convert_odor_to_num)

    ds['gill-attachment'] = ds['gill-attachment'].map(convert_gill_attachment_to_num)

    ds['gill-spacing'] = ds['gill-spacing'].map(convert_gill_spacing_to_num)

    ds['gill-size'] = ds['gill-size'].map(convert_gill_size_to_num)

    ds['gill-color'] = ds['gill-color'].map(convert_gill_color_to_num)

    ds['stalk-shape'] = ds['stalk-shape'].map(convert_stalk_shape_to_num)

    ds['stalk-root'] = ds['stalk-root'].map(convert_stalk_root_to_num)

    ds['stalk-surface-above-ring'] = ds['stalk-surface-above-ring'].map(convert_stalk_surface_to_num)

    ds['stalk-surface-below-ring'] = ds['stalk-surface-below-ring'].map(convert_stalk_surface_to_num)

    ds['stalk-color-above-ring'] = ds['stalk-color-above-ring'].map(convert_stalk_color_to_num)

    ds['stalk-color-below-ring'] = ds['stalk-color-below-ring'].map(convert_stalk_color_to_num)

    ds['veil-color'] = ds['veil-color'].map(convert_veil_color_to_num)

    ds['ring-number'] = ds['ring-number'].map(convert_ring_num_to_num)

    ds['ring-type'] = ds['ring-type'].map(convert_ring_type_to_num)

    ds['spore-print-color'] = ds['spore-print-color'].map(convert_spore_print_color_to_num)

    ds['population'] = ds['population'].map(convert_population_to_num)

    ds['habitat'] = ds['habitat'].map(convert_habitat_to_num)

    ds = ds.drop(['veil-type'], axis=1)

    return ds
ds = processing_data(ds)
input_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',

       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',

       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',

       'stalk-surface-below-ring', 'stalk-color-above-ring',

       'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',

       'spore-print-color', 'population', 'habitat']

output_cols = ['class']

x_train = ds[input_cols]

y_train = ds[output_cols]
LR = LogisticRegression()

LR.fit(x_train, y_train.values.ravel())

LR.score(x_train, y_train)*100
DT = DecisionTreeClassifier()

DT.fit(x_train, y_train)

DT.score(x_train, y_train)*100
rf = RandomForestClassifier()

rf.fit(x_train, y_train.values.ravel())



rf.score(x_train, y_train)*100