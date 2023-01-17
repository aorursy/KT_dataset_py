import pandas

import matplotlib.pyplot as plt

%matplotlib inline
t = pandas.read_csv("../input/Donald.csv", encoding='ISO-8859-1')

h = pandas.read_csv("../input/Clinton.csv", encoding='ISO-8859-1')

t1 = pandas.read_csv("../input/Trump.csv", encoding='ISO-8859-1')

h1 = pandas.read_csv("../input/Hillary.csv", encoding='ISO-8859-1')
t_total= [t, t1]

trump = pandas.concat(t_total)

trump.head()
h_total= [h, h1]

hillary = pandas.concat(h_total)

hillary.head()
sum(trump.likes)
sum(hillary.likes)
hillary_network = hillary.groupby('network').sum()['likes']

trump_network = trump.groupby('network').sum()['likes']
print(hillary_network , trump_network)
hillary_network.plot.bar()
trump_network.plot.bar()
hillary_timeline = hillary.groupby('timestamp')

trump_network = trump.groupby('network')
split_month = lambda x : x.split()[0].split('/')[0]

split_year = lambda x : x.split()[0].split('/')[2]



month = hillary.timestamp.apply(split_month)

year = hillary.timestamp.apply(split_year)

hillary['month']= month

hillary['year']= year

hillary.head()
# def filter_year(x) :

#     if x['year'] >14:

#         return x



# hillary_year_month = hillary['year'].filter( regex = '\b(15|16)\b')

hillary_year_month = hillary.groupby(['year','month'])['likes'].sum()

hillary_year_month.sort()

hillary_year_month.plot.bar()

hillary_year_month.head()
month = trump.timestamp.apply(split_month)

year = trump.timestamp.apply(split_year)

trump['month']= month

trump['year']= year

trump_year_month = trump.groupby(['year','month'])['likes'].sum()

trump_year_month.sort()

trump_year_month.plot.bar()

trump_year_month.head()
hillary_author = hillary.groupby(['author'])

len(hillary_author)
trump_author = trump.groupby(['author'])

len(trump_author)