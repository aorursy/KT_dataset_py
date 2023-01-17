import pandas as pd

a = pd.read_csv('globalterrorismdb_0617dist.csv', encoding='latin-1')

a.head()



b = a.groupby('country_txt').count()

b.to_csv('out1.csv', sep=',')



c = a.groupby(['iyear']).size()

c.to_csv('out2.csv', sep=',')



d = a.groupby(['country_txt']).size()

d.to_csv('out3.csv', sep=',')



e = a.groupby(['iyear', 'country_txt']).size().reset_index(name='counts')

e.to_csv('out4.csv', sep=',')



f = a.groupby(['attacktype1_txt']).size()

f.to_csv('out6.csv', sep=',')



g = a.groupby(['attacktype1_txt', 'targtype1_txt', 'targsubtype1_txt']).size().reset_index(name='counts')

g.to_csv('out5.csv', sep=',')