import pandas as pd

covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

covid.head(2)
def func(n):                  # ekhane 'n' hocche func er input

    if 0 <= n < 10 :

        return 'Green Zone' 

    elif 10 <= n < 20 :

        return 'Yellow Zone'

    else :

        return 'Red Zone'
covid['Zone'] = covid.Confirmed.map(func)
covid.head()
covid['Confirmed'] = covid.Confirmed.map(func)
covid.head()