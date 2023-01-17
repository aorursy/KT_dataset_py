# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel(r'/kaggle/input/the-holy-quran/Dataset-Verse-by-Verse.xlsx')

df.head()
len(df)


df1 = df.groupby(['Juz']).agg({

    'SurahNo': ['nunique', 'unique'],

    'SurahNameEnglish': ['nunique', 'unique'],

    'ArabicWordCount': 'sum',

    'ArabicLetterCount': 'sum',

    

})#.to_excel(r'test.xlsx')#.sum()#.plot(kind='bar')



df1
df1 = df1.reset_index()

df1.columns = ['_'.join(col).strip() for col in df1.columns.values]

df1
df1.columns
dft1 = df1.melt(id_vars=['Juz_', 'SurahNo_nunique', 'SurahNo_unique', 'SurahNameEnglish_nunique',

       'SurahNameEnglish_unique'], value_vars=['ArabicWordCount_sum', 'ArabicLetterCount_sum'])

dft1
chart = alt.Chart(prediction_table2, title='Simulated (attainable) and predicted yield ').mark_bar(

    opacity=1,

    ).encode(

    column = alt.Column('date:O', spacing = 5, header = alt.Header(labelOrient = "bottom")),

    x =alt.X('variable', sort = ["Actual_FAO", "Predicted", "Simulated"],  axis=None),

    y =alt.Y('value:Q'),

    color= alt.Color('variable')

).configure_view(stroke='transparent')



chart.display()
dft1
chart = Chart(df).mark_bar().encode(

   column=Column('Genre', 

                 axis=Axis(axisWidth=1.0, offset=-8.0, orient='bottom'),

                 scale=Scale(padding=4.0)),

   x=X('Gender', axis=False),

   y=Y('Rating', axis=Axis(grid=False)),

   color=Color('Gender', scale=Scale(range=['#EA98D2', '#659CCA']))

).configure_facet_cell(

    strokeWidth=0.0,

)



chart.display()
import altair as alt

from altair import *



alt.Chart(dft1).mark_bar(opacity=1).encode(

   column='Juz_',

             

    x=X('variable:O'),#, axis=Axis(grid=False)),

    y=Y('value:Q'),#, axis=Axis(grid=False)),

    color='variable:N',



).configure_view(

    strokeWidth=0.0,

)



# .properties(width=30, height=400)
import altair as alt

from altair import *



alt.Chart(dft1).mark_bar(opacity=1).encode(

   column='Juz_',

             

    x='variable:O',

    y='value:Q',

    color='variable:N',



).configure_view(

    strokeWidth=0.0,

)



# .properties(width=30, height=400)