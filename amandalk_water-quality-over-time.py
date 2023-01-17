import datetime



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
sns.set(rc={'figure.figsize':(15, 6)})



sns.set_style('white', {

    'axes.spines.left': False,

    'axes.spines.bottom': False,

    'axes.spines.right': False,

    'axes.spines.top': False

})



pd.set_option('display.max_rows', None)
df = pd.read_csv('/kaggle/input/sp-beaches-water-quality/sp_beaches.csv', parse_dates=['Date'])
df.head()
cities = np.unique(df['City'])



for city in cities:

    beaches = np.unique(df[df['City'] == city]['Beach'])

    

    rows, cols = (len(beaches) + 2) // 3, 3

    fig = plt.figure(figsize=(15, rows * 2.4 + 1.2))

    

    fig.suptitle(city)

    fig.subplots_adjust(hspace=0.4)

    fig.tight_layout()



    for i, beach in enumerate(list(beaches)):

        ax = fig.add_subplot(rows, cols, i+1)

        

        df_beach = df[(df['Beach'] == beach) & (df['City'] == city)]

    

        sns.lineplot(x='Date', y='Enterococcus', data=df_beach, ax=ax)

        

        ax.set_title(beach)

        ax.set_xlim(datetime.date(2012, 1, 1), datetime.date(2021, 12, 31))

        ax.set_ylim(0, 1000)

        

        if i // 3 + 1 < rows:

            ax.set_xlabel('')

            ax.set_xticklabels([])

        else:

            ax.set_xlabel('Year')

            ax.set_xticks(pd.date_range(start='1/1/2012', end='1/1/2021', freq='2Y'))

            ax.set_xticklabels(np.arange(2013, 2020, 2))

            

        if i % 3 != 0:

            ax.set_ylabel('')

            ax.set_yticklabels([])

    

    plt.show()
