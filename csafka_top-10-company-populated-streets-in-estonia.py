import pandas as pd

import matplotlib.pyplot as p
df = pd.read_csv('../input/companies2016-10-05.csv', header=0)
df.head()
df.describe()
df.info()
df = df.dropna(subset=['asukoht_ettevotja_aadressis'])
df[ df['asukoht_ettevotja_aadressis'].isnull()]
def get_street(address):

    street = address.split(' ')

    if len(street) > 1:

        del street[-1]

    return ' '.join(street)
df['street'] = df.asukoht_ettevotja_aadressis.apply(lambda x: get_street(x) )
p.style.use('ggplot')
df['street'].value_counts()[:10].plot(kind='barh', color="Red")