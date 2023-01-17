import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!ls
data = pd.read_csv('../input/flavors_of_cacao.csv')
data.head()
data['Cocoa\nPercent'] = data['Cocoa\nPercent'].apply(lambda x: float(x.split('%')[0]))/100
data.head()
data.columns[data.isnull().any()].tolist
print(data.corr())
plt.hist(data['Rating'])
plt.hist(data['Cocoa\nPercent'])
plt.plot(data['Rating'], data['Cocoa\nPercent'],'o')
data['Specific Bean Origin\nor Bar Name'][data['Rating']==5]
data['Company\xa0\n(Maker-if known)'][data['Rating']==5]
amedei=data['Rating'][data['Company\xa0\n(Maker-if known)']=='Amedei']
print(amedei)
print('Average rating for Amedei is: ',amedei.mean())
data[data['Specific Bean Origin\nor Bar Name'].isin(['Chuao', 'Toscano Black'])]
chuao=data['Rating'][data['Specific Bean Origin\nor Bar Name']=='Chuao']
toscano=data['Rating'][data['Specific Bean Origin\nor Bar Name']=='Toscano Black']
print('Average rating for chocolates produced from Chuao bean is: ',chuao.mean())
print('Average rating for chocolates produced from Toscano Black bean is: ',toscano.mean())
data['Company\xa0\n(Maker-if known)'][data['Rating']<2]
