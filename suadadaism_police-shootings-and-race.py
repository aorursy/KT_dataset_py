unarmed = reduced[reduced['arms_category'] == 'Unarmed']

armed = reduced[reduced['arms_category'] == 'Armed']

create_scaled_bars(unarmed,armed,'race')
#percentage difference of black people being shot when armed vs unarmed

print(armed[armed['race'] == 'Black'].shape[0]/armed.shape[0] - unarmed[unarmed['race'] == 'Black'].shape[0]/unarmed.shape[0])

#percentage difference of white people being shot when armed vs unarmed

print(armed[armed['race'] == 'White'].shape[0]/armed.shape[0] - unarmed[unarmed['race'] == 'White'].shape[0]/unarmed.shape[0])

#quickly calculating Z score if being armed makes a Black person more likely to be shot and killed

p_unarmed = unarmed[unarmed['race'] == 'Black'].shape[0]/unarmed.shape[0]

p_armed = armed[armed['race'] == 'Black'].shape[0]/armed.shape[0]

z = (p_armed - p_unarmed)/np.sqrt((p_unarmed * (1-p_unarmed))/armed[armed['race'] == 'Black'].shape[0])

print(z)
ax = famd.plot_row_coordinates(descriptor_data, 

                               ax =None, 

                               figsize=(6,6), 

                               x_component=0, 

                               y_component=1,

                              ellipse_outline=False,

                              ellipse_fill=True,

                              show_points=True)
description_breakdown(descriptor_data[(scaled_desc_data[0] > 18) & (scaled_desc_data[0] < 30)])
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

!pip install prince

import prince

%matplotlib inline
data = pd.read_csv("../input/us-police-shootings/shootings.csv")

data.head()
def create_bars(df,col):

    key, counts = np.unique(df[col], return_counts = True)

    fig, ax = plt.subplots(figsize=(12,9))

    ax.bar(height=counts, x=key)

    ax.set_xticklabels(labels=key, rotation=90)

    for _, spine, in ax.spines.items():

        spine.set_visible(False)

    plt.show()
create_bars(data,'race')
create_bars(data,'age')
create_bars(data, 'arms_category')
create_bars(data,'threat_level')
create_bars(data,'state')
create_bars(data,'flee')
create_bars(data,'manner_of_death')
create_bars(data,'body_camera')
#data = data.drop('id', axis='columns')

descriptor_columns = ['age','gender','race']

descriptor_data = data[descriptor_columns]

descriptor_data.head()


famd = prince.FAMD(n_components =2, n_iter = 3,copy = True,check_input = True, engine = 'auto', random_state =42)
famd = famd.fit(descriptor_data)
scaled_desc_data = famd.row_coordinates(descriptor_data)

scaled_desc_data.head()
descriptor_data[(scaled_desc_data[1] >70) ]
def description_breakdown(df):

    print(df.describe())

    fig, ax = plt.subplots(1, 3, figsize=(18,10))

    

        

    key, counts = np.unique(df['age'], return_counts = True)

    ax[0].bar(height=counts, x=key)

    for _, spine, in ax[0].spines.items():

        spine.set_visible(False)

        

    key, counts = np.unique(df['race'], return_counts = True)

    ax[1].bar(height=counts, x=key)

    ax[1].set_xticklabels(labels=key,rotation=90)

    for _, spine in ax[1].spines.items():

        spine.set_visible(False)

        

    key, counts = np.unique(df['gender'], return_counts = True)

    ax[2].bar(height = counts, x= key)

    ax[2].set_xticklabels(labels=key, rotation = 90)

    for _, spine in ax[2].spines.items():

        spine.set_visible(False)

    plt.show()
description_breakdown(descriptor_data[(scaled_desc_data[0] > 80) ])
ax = famd.plot_row_coordinates(descriptor_data, 

                               ax =None, 

                               figsize=(6,6), 

                               x_component=0, 

                               y_component=1,

                              ellipse_outline=False,

                              ellipse_fill=True,

                              show_points=True)

description_breakdown(descriptor_data[(scaled_desc_data[0] < 40) ])
description_breakdown(descriptor_data[(scaled_desc_data[0] < 120) & (scaled_desc_data[0] > 60) ])
description_breakdown(descriptor_data[(scaled_desc_data[0] > 120) ])
description_breakdown(descriptor_data[(scaled_desc_data[0] < 60) & (scaled_desc_data[0] > 40) ])
def description_breakdown_extended(df):

    print(df.describe())

    fig, ax = plt.subplots(1, 4, figsize=(18,10))

    

        

    key, counts = np.unique(df['age'], return_counts = True)

    ax[0].bar(height=counts, x=key)

    for _, spine, in ax[0].spines.items():

        spine.set_visible(False)

        

    key, counts = np.unique(df['race'], return_counts = True)

    ax[1].bar(height=counts, x=key)

    ax[1].set_xticklabels(labels=key,rotation=90)

    for _, spine in ax[1].spines.items():

        spine.set_visible(False)

        

    key, counts = np.unique(df['gender'], return_counts = True)

    ax[2].bar(height = counts, x= key)

    ax[2].set_xticklabels(labels=key, rotation = 90)

    for _, spine in ax[2].spines.items():

        spine.set_visible(False)

        

    key, counts = np.unique(df['signs_of_mental_illness'], return_counts = True)

    ax[3].bar(height=counts, x = key)

    for _, spine in ax[3].spines.items():

        spine.set_visible(False)

    plt.show()
data['signs_of_mental_illness'] = data['signs_of_mental_illness'].astype(str)
#data = data.drop('id', axis='columns')

descriptor_columns = ['age','gender','race','signs_of_mental_illness']

descriptor_data = data[descriptor_columns]



famd = prince.FAMD(n_components =2, n_iter = 3,copy = True,check_input = True, engine = 'auto', random_state =42)

famd = famd.fit(descriptor_data)

scaled_desc_data = famd.row_coordinates(descriptor_data)



ax = famd.plot_row_coordinates(descriptor_data, 

                               ax =None, 

                               figsize=(6,6), 

                               x_component=0, 

                               y_component=1,

                              ellipse_outline=False,

                              ellipse_fill=True,

                              show_points=True)

#ax.get_figure().savefig('images/famd_row_coordinates.svg')
description_breakdown_extended(descriptor_data[(scaled_desc_data[0] > 25) & (scaled_desc_data[0] < 30)])
def create_scaled_bars(df,df2,col):

    key, counts = np.unique(df[col], return_counts = True)

    total = sum(counts)

    fig, ax = plt.subplots(figsize=(12,9))

    ax.bar(label = 'unarmed',height=list(map(lambda x : x/total, counts)), x=key)

    ax.set_xticklabels(labels=key, rotation=90)

    for _, spine, in ax.spines.items():

        spine.set_visible(False)

        

        

    key, counts = np.unique(df2[col], return_counts = True)

    total = sum(counts)

    ax.bar(label = 'armed', alpha = 0.3, color = 'r', height=list(map(lambda x : x/total, counts)), x=key)

    plt.legend(loc='upper left', prop={'size':26})

    ax.set_ylabel('percentages of total fatality')

    ax.set_xlabel('race')

    plt.show()

    

   
def is_armed(val):

    if val != 'Unarmed':

        return 'Unarmed'

    return 'Armed'



columns = ['race','arms_category']

reduced = data[columns]



reduced = reduced[reduced['arms_category'] != 'Unknown']

reduced['arms_category'] = reduced['arms_category'].apply(lambda x: is_armed(x))