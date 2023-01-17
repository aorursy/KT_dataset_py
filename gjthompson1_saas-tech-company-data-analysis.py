import re

import pandas as pd

import numpy as np

from scipy import stats



import seaborn as sns

import matplotlib.pyplot as plt



sns.set(color_codes=True)

plt.rcParams['figure.figsize'] = [10, 10]



df = pd.read_csv('../input/latka_web_data.csv', index_col=False)

df['financials.arr_min.millions'] = df['financials.arr_min']/1000000

df['years_in_business'] = 2018-df['year_founded']

df['hours_sleep'] = pd.to_numeric(df['podcast.guest.hours_sleep'], errors='coerce')

df['years_in_business_cat'] = pd.cut(

    df['years_in_business'], 

    bins = [0, 1, 2, 4, 8, 100], 

    labels=['0-1','1-2','2-4','4-8','8+']

)

df['age_cat'] = pd.cut(

    df['podcast.guest.age'], 

    bins = [20, 30, 40, 50, 100], 

    labels=['(20-30]','(30-40]','(40-50]','50+']

)

df['number_of_kids_cat'] = pd.cut(

    df['podcast.guest.kids'], 

    bins = [0, 1, 2, 3, 4, 10], 

    labels=['0','1','2','3','4+']

)

df['hours_sleep_cat'] = pd.cut(

    df['hours_sleep'], 

    bins = [0, 3, 6, 7, 8, 9, 11], 

    labels=['(0-3]','(3-6]','(6-7]','(7-8]', '(8-9]', '9+']

)



df['married'] = df['podcast.guest.married'].apply(lambda x: 

    'True' if x in ['Yes', 'Married', 'Engaged', 'yes', 'ye', 'Ye'] else 'False'

)



df['likes_popular_ceo'] = df['podcast.guest.favorite_ceo_name'].apply(lambda x:

    True if x in ['Elon Musk', 'Jeff Bezos', 'Steve Jobs', 'Marc Benioff', 'Mark Zuckerberg'] else False

)
Y = 'financials.arr_min.millions'

X = 'num_employees'



df_subset = df.dropna(subset=['financials.arr_min', 'num_employees'])

df_subset = df_subset[df_subset['num_employees'] < 2000]



slope, intercept, r_value, p_value, std_err = stats.linregress(df_subset[X], df_subset[Y])



ax = sns.regplot(x=X, y=Y, data=df_subset, color='b', 

 line_kws={

     'label':"{0} = {1:.2f}*{2} + {3:.2f}, p-value: {4:.6f}".format(

         "Y", 

         slope,

         "X",

         intercept,

         p_value

        )

    }

)



ax.legend()

plt.show()
Y = 'financials.arr_min.millions'

X = 'years_in_business'



df_subset = df.dropna(subset=[Y, X])



slope, intercept, r_value, p_value, std_err = stats.linregress(df_subset[X], df_subset[Y])



ax = sns.regplot(x=X, y=Y, data=df_subset, color='b', 

 line_kws={

     'label':"{0} = {1:.2f}*{2} + {3:.2f}, p-value: {4:.6f}".format(

         "Y", 

         slope,

         "X",

         intercept,

         p_value

        )

    }

)



ax.legend()

plt.show()
Y = 'financials.arr_min.millions'

X = 'married'



df_subset = df.dropna(subset=[Y, X])



df_subset = df_subset[df_subset['financials.arr_min.millions'] < 100]



sns.catplot(x="age_cat", y=Y, hue=X, kind="box", data=df_subset, height=8, aspect=1)

plt.show()



sns.catplot(x="years_in_business_cat", y=Y, hue=X, kind="box", data=df_subset, height=8, aspect=1)

plt.show()
Y = 'financials.arr_min.millions'

X = 'number_of_kids_cat'



df_subset = df.dropna(subset=[Y, X])



df_subset = df_subset[df_subset['financials.arr_min.millions'] < 100]

sns.catplot(x="age_cat", y=Y, hue=X, kind="box", data=df_subset, height=8, aspect=1)

plt.show()



df_subset = df_subset[df_subset['financials.arr_min.millions'] < 100]

sns.catplot(x="years_in_business_cat", y=Y, hue=X, kind="box", data=df_subset, height=8, aspect=1)

plt.show()
Y = 'financials.arr_min.millions'

X = 'hours_sleep_cat'



df_subset = df.dropna(subset=[Y, X])



df_subset = df_subset[df_subset['financials.arr_min.millions'] < 100]

sns.catplot(x="age_cat", y=Y, hue=X, kind="box", data=df_subset, height=8, aspect=1)

plt.show()



df_subset = df_subset[df_subset['financials.arr_min.millions'] < 100]

sns.catplot(x="years_in_business_cat", y=Y, hue=X, kind="box", data=df_subset, height=8, aspect=1)

plt.show()
Y = 'financials.arr_min.millions'

X = 'likes_popular_ceo'

df_subset = df.dropna(subset=[Y, X])

df_subset = df_subset[df_subset['financials.arr_min.millions'] < 100]

sns.catplot(x=X, y=Y, kind="box", data=df_subset, height=8, aspect=1)

plt.show()
df['move_fast'] = df['podcast.guest.advice'].apply(lambda x: 

    True if re.search(('more risk|start|sooner'), str(x)) is not None else False

)



Y = 'financials.arr_min.millions'

X = 'move_fast'

sns.catplot(x=X, y=Y, kind="box", data=df, height=8, aspect=1)

plt.show()