# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import altair as alt

full_clean_data = pd.read_csv('../input/altair-interactive/Circle.csv', parse_dates=['Date'])
countries= ['Bagerhat','Bandarban','Barguna','Barishal','Bhola','Bogura','Brahmmanbaria','Chandpur','Chapai Nawabganj','Chattogram','Chuadanga','Cox s Bazar','Cumilla','Dhaka','Dinajpur','Faridpur','Feni','Gaibandha','Gazipur','Gopalganj','Habiganj','Jamalpur','Jashore','Jhalakhati','Jhenaidah','Joypurhat','Khagrachhari','Khulna','Kishoreganj','Kurigram','Kushtia','Lakshmipur','Lalmonirhat','Madaripur','Magura','Manikganj','Meherpur','Moulvibazar','Munshiganj','Mymensing','Naogaon','Narail','Narayanganj','Narsingdi','Natore','Netrokona','Nilphamari','Noakhali','Pabna','Panchagarh','Patuakhali','Pirojpur','Rajbari','Rajshahi','Rangamati','Rangpur','Satkhira','Shariatpur','Sherpur','Sirajganj','Sunamganj','Sylhet','Tangail','Thakurgaon']
selected_data = full_clean_data[full_clean_data['District'].isin(countries)]
interval = alt.selection_interval()

circle = alt.Chart(selected_data).mark_circle().encode(

    x='Date',

    y='District',

    color=alt.condition(interval, 'District', alt.value('lightgray')),

    size=alt.Size('Cases:Q',

        scale=alt.Scale(range=[0, 700]),

        legend=alt.Legend(title='Daily new cases')

    ) 

).properties(

    width=700,

    height=700,

    selection=interval

)

bars = alt.Chart(selected_data).mark_bar().encode(

    y='District',

    color='District',

    x='sum(Cases):Q'

).properties(

    width=500

).transform_filter(

    interval

)

circle & bars
data=circle & bars
data.save('chart.html', embed_options={'renderer':'svg'})