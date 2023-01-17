import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

pwt=pd.read_csv('../input/PWT.csv')

#print(pwt.head())

#print(pwt.describe().T)

#
BE=pwt[pwt['country']=='Germany']

BE.T
import matplotlib.pyplot as plt

plt.figure()

BE=BE.set_index(BE['year'])

BET=pd.DataFrame(BE['labsh'][-10:])

BET.plot.barh

BET.plot()

BET

print('% increase income share to labour is this a tax shift effect ?',(0.629824/0.618982-1)*100)
plt.figure()

BE['procent active population']=BE['emp']/BE['pop']/0.477

BE['active population growth']=BE['emp']/38.757

BE['population growth']=BE['pop']/81.246

BET=pd.DataFrame(BE[['procent active population','active population growth','population growth']][-10:])

BET.plot.barh

print(BET)

BET.plot()

print('% increase of active population surpasses increase of population%',(43.64/42.968-1)*100)

print('activation growth 10% and NO population growth ! ')

plt.figure()

BE['income_per_person']=BE['rgdpo']/BE['pop']/33721.979

BE['income_per_active']=BE['rgdpo']/BE['emp']/82040.203

BET=pd.DataFrame(BE[['income_per_person','income_per_active']][-10:])

BET.plot.barh

BET.plot()



print('% increase income from labour an impressive 14%')

print('% increase income per inhabitant is 27% thus also higher;..')

print('german labour surpassed belgian labour, income per person surpasses 40%')
plt.figure()

BE['income_per_hour']=(BE['rgdpo']/( BE['avh'] * BE['emp'] ) /52.419536)

BET=pd.DataFrame(BE['income_per_hour'][-10:])

BET.plot.barh

BET.plot()



print('% gross income per hour increases with an impressive 17% and german labour earns 22% more then belgian labour')
plt.figure()

BE['ratio consumption government']=BE['csh_c']/BE['csh_g']

BE['ratio consumption investment']=BE['csh_c']/BE['csh_i']

BET=pd.DataFrame(BE[['ratio consumption government','ratio consumption investment']][-10:])

BET.plot.barh

BET.plot()

print('% also an impressive government cookie monster effect, but not so bad as the belgian story... -22% available GDP income for private consumption ',(3.2/4.1-1)*100)
plt.figure()

BE['export']=BE['csh_x']*100

BE['import']=-BE['csh_m']*100

BET=pd.DataFrame(BE[['export','import']][-10:])

BET.plot.barh

BET.plot()

print('export higher then import, and still the motor of the german economy. The consumption follows the export',(60/55-1)*100)