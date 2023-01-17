import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

pwt=pd.read_csv('../input/PWT.csv')

#print(pwt.head())

#print(pwt.describe().T)

#
BE=pwt[pwt['country']=='Belgium']

BE.T
import matplotlib.pyplot as plt

plt.figure()

BE=BE.set_index(BE['year'])

BET=pd.DataFrame(BE['labsh'][-5:])

BET.plot.barh

BET.plot()

BET

print('% increase income share to labour is this a tax shift effect ?',(0.629824/0.618982-1)*100)
plt.figure()

BE['procent active population']=BE['emp']/BE['pop']/0.41104213

BE['active population growth']=BE['emp']/ ( 4.341195 )

BE['population growth']=BE['pop']/ ( 10.561436 )

BET=pd.DataFrame(BE[['procent active population','active population growth','population growth']][-10:])

BET.plot.barh



BET.plot()

print('% increase of active population 1.56%',(43.64/42.968-1)*100)

print('activation growth 12% goes faster then immigration 6% one thing going right ')

plt.figure()

BE['income_per_person']=BE['rgdpo']/BE['pop']/33721.979

BE['income_per_active']=BE['rgdpo']/BE['emp']/82040.203

BET=pd.DataFrame(BE[['income_per_person','income_per_active']][-10:])

BET.plot.barh

BET.plot()



print('% increase income from labour an impressive 11%',(91541/87498-1)*100)

print('% increase income per inhabitant is 18% higher;.. amazing how communistic Belgium is working ')

print('labour is not renumerated in Belgium and did not improve under Michel yet very bizar, its not the poor that are getting richter problably more the ambtonary getting higher pensions that is lifting this number...')
plt.figure()

BE['income_per_hour']=(BE['rgdpo']/( BE['avh'] * BE['emp'] ) /52.419536)

BET=pd.DataFrame(BE['income_per_hour'][-10:])

BET.plot.barh

BET.plot()



print('% gross income per hour increases with an impressive 10% no taming of the costcurve of labour ')
plt.figure()

BE['ratio consumption government']=BE['csh_c']/BE['csh_g']

BE['ratio consumption investment']=BE['csh_c']/BE['csh_i']

BET=pd.DataFrame(BE[['ratio consumption government','ratio consumption investment']][-10:])

BET.plot.barh

BET.plot()

print('% the government cookie monster keeps on eating our cake.. -26% available GDP income for private consumption ',(2.5/3.4-1)*100)
plt.figure()

BE['export']=BE['csh_x']*100

BE['import']=-BE['csh_m']*100

BET=pd.DataFrame(BE[['export','import']][-10:])

BET.plot.barh

BET.plot()

print('% not enough taxshift on the consumption to tamper the import, import grows with 15.9%... ',(145/125-1)*100)