import pandas as pd
pd.set_option('precision', 3)

B2011 = pd.read_csv('/kaggle/input/bger6b2018/1B_2011.csv')
B2012 = pd.read_csv('/kaggle/input/bger6b2018/1B_2012.csv')
B2013 = pd.read_csv('/kaggle/input/bger6b2018/1B_2013.csv')
B2014 = pd.read_csv('/kaggle/input/bger6b2018/1B_2014.csv')
B2015 = pd.read_csv('/kaggle/input/bger6b2018/1B_2015.csv')
B2016 = pd.read_csv('/kaggle/input/bger6b2018/1B_2016.csv')
B2017 = pd.read_csv('/kaggle/input/bger6b2018/1B_2017.csv')
B2018 = pd.read_csv('/kaggle/input/bger6b2018/1B_2018.csv')
B2019 = pd.read_csv('/kaggle/input/bger6b2018/1B_2019.csv')
B2020 = pd.read_csv('/kaggle/input/bger6b2018/1B_2020.csv')
B2011b = pd.read_csv('/kaggle/input/bger6b2018/6B_2011.csv')
B2012b = pd.read_csv('/kaggle/input/bger6b2018/6B_2012.csv')
B2013b = pd.read_csv('/kaggle/input/bger6b2018/6B_2013.csv')
B2014b = pd.read_csv('/kaggle/input/bger6b2018/6B_2014.csv')
B2015b = pd.read_csv('/kaggle/input/bger6b2018/6B_2015.csv')
B2016b = pd.read_csv('/kaggle/input/bger6b2018/6B_2016.csv')
B2017b = pd.read_csv('/kaggle/input/bger6b2018/6B_2017.csv')
B2018b = pd.read_csv('/kaggle/input/bger6b2018/6B_2018.csv')
B2019b = pd.read_csv('/kaggle/input/bger6b2018/6B_2019.csv')
B2020b = pd.read_csv('/kaggle/input/bger6b2018/6B_2020.csv')



df = pd.concat([B2011, B2012, B2013, B2014, B2015, B2016, B2017, B2018, B2019, B2020, \
                      B2011b, B2012b, B2013b, B2014b, B2015b, B2016b, B2017b, B2018b, B2019b, B2020b])

rv_count = df.groupby('Rechtsvertreter').Verfahrensnummer.count().reset_index()
rv_count = rv_count.rename(columns={'Verfahrensnummer': 'tot'})
rv_count = rv_count.sort_values('tot', ascending=False)

# unvertretene und anonymisierte aussortieren
rv_count = rv_count[rv_count.Rechtsvertreter != 'unvertreten']
rv_count = rv_count[rv_count.Rechtsvertreter != 'A.________']

# einfachnennungen aussortieren
rv_count = rv_count[rv_count.tot > 1]

rv_gut_count = df[df.Verfahrensergebnis.str.contains('(teilweise Gutheissung|Gutheissung)', regex=True)]
rv_gut_count = rv_gut_count.groupby('Rechtsvertreter').Verfahrensnummer.count().reset_index()
rv_gut_count = rv_gut_count.rename(columns={'Verfahrensnummer': 'tot_gut'})
rv_gut_count = rv_gut_count.sort_values('tot_gut', ascending=False)

rv_gut_count = rv_gut_count[rv_gut_count.Rechtsvertreter != 'unvertreten']
rv_gut_count = rv_gut_count[rv_gut_count.tot_gut > 1]

rv_gut_count = pd.merge(rv_count, rv_gut_count, how='outer')

rv_gut_count = rv_gut_count.fillna(0)
rv_gut_count['%'] = rv_gut_count.tot_gut / rv_gut_count.tot * 100
rv_gut_count = rv_gut_count.sort_values(['%', 'tot_gut'], ascending=False)


# nicht erfolgreiche ausblenden
rv_gut_count = rv_gut_count[rv_gut_count.tot_gut > 0]

# alle unter 9 Beschwerden (mind. durchschn. 1 Beschwerde pro Jahr) ausblenden
rv_gut_count = rv_gut_count[rv_gut_count.tot >= 9]

rv_gut_count = rv_gut_count.reset_index(drop=True)

# cut top20
rv_gut_count = rv_gut_count.head(20)

rv_gut_count = rv_gut_count.rename(columns={'tot': 'Total Beschwerden', 'tot_gut':'Total Gutheissungen'})
rv_gut_count.style.background_gradient(subset=['%'], cmap='BuGn')
platz1 = df[df.Rechtsvertreter.str.contains('Daniel Kinzer') & df.Verfahrensergebnis.str.contains('(teilweise Gutheissung|Gutheissung)', regex=True)]
platz1 = platz1.sort_values('Verfahrensergebnis', ascending=False)
platz1.style
platz2 = df[df.Rechtsvertreter.str.contains('Jean-Pierre Garbade') & df.Verfahrensergebnis.str.contains('(Gutheissung|teilweise Gutheissung)', regex=True)]
platz2 = platz2.sort_values('Verfahrensergebnis', ascending=False)
platz2.style
platz3 = df[df.Rechtsvertreter.str.contains('Gaétan Droz') & df.Verfahrensergebnis.str.contains('(Gutheissung|teilweise Gutheissung)', regex=True)]
platz3 = platz3.sort_values('Verfahrensergebnis', ascending=False)
platz3.style