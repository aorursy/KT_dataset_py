import numpy as np
import pandas as pd
!ls ../input/avocado-prices/
avocado = pd.read_csv("../input/avocado-prices/avocado.csv")
avocado.head()
#avocado['Date'].head()
avocado['AveragePrice'].value_counts()
avocado_year_bool = avocado['year']<=2016
idx = avocado_year_bool.sum()
avocado_test = avocado[:idx]
avocado_answer = avocado[idx:]
print(len(avocado),len(avocado_test),len(avocado_answer))
def year_average_price(y):
    avo_sum = {}
    date_cnt = {}
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        avo_sum[avocado['Date'][i]] = 0
        date_cnt[avocado['Date'][i]] = 0
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        avo_sum[avocado['Date'][i]] += avocado['AveragePrice'][i]
        date_cnt[avocado['Date'][i]] += 1
    for k,v in date_cnt.items():
        avo_sum[k] /= v
    return sorted(avo_sum.items(),key=lambda x:x[0])
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

fig, ax = plt.subplots()
avosum2015 = year_average_price(2015)
as5x = []
as5y = []
for a,b in avosum2015:
    as5x.append(a[5:])
    as5y.append(b)
ax.plot(as5x,as5y,label='2015')
avosum2016 = year_average_price(2016)
as6x = []
as6y = []
for a,b in avosum2016:
    as6x.append(a[5:])
    as6y.append(b)
ax.plot(as6x,as6y,label='2016')
avosum2017 = year_average_price(2017)
as7x = []
as7y = []
for a,b in avosum2017:
    as7x.append(a[5:])
    as7y.append(b)
ax.plot(as7x,as7y,label='2017')
ax.legend()
loc = plticker.MultipleLocator(base=15)
ax.xaxis.set_major_locator(loc)
def year_average_bags(y):
    bag_sum = {}
    date_cnt = {}
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        bag_sum[avocado['Date'][i][5:]] = 0
        date_cnt[avocado['Date'][i][5:]] = 0
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        bag_sum[avocado['Date'][i][5:]] += avocado['Total Bags'][i]
        date_cnt[avocado['Date'][i][5:]] += 1
    for k,v in date_cnt.items():
        bag_sum[k] /= v
    return sorted(bag_sum.items(),key=lambda x:x[0])
fig, ax = plt.subplots()
bagsum2015 = year_average_bags(2015)
bs5x = []
bs5y = []
for a,b in bagsum2015:
    bs5x.append(a)
    bs5y.append(b)
ax.plot(bs5x,bs5y,label='2015')
bagsum2016 = year_average_bags(2016)
bs6x = []
bs6y = []
for a,b in bagsum2016:
    bs6x.append(a)
    bs6y.append(b)
ax.plot(bs6x,bs6y,label='2016')
bagsum2017 = year_average_bags(2017)
bs7x = []
bs7y = []
for a,b in bagsum2017:
    bs7x.append(a)
    bs7y.append(b)
ax.plot(bs7x,bs7y,label='2017')
ax.legend()
loc = plticker.MultipleLocator(base=15)
ax.xaxis.set_major_locator(loc)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(as5x,as5y,'C0',label='avocado price')
ax2.plot(bs5x,bs5y,'C1',label='number of bags')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')
loc = plticker.MultipleLocator(base=15)
ax1.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_locator(loc)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(as6x,as6y,'C0',label='avocado price')
ax2.plot(bs6x,bs6y,'C1',label='number of bags')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')
loc = plticker.MultipleLocator(base=15)
ax1.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_locator(loc)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(as7x,as7y,'C0',label='avocado')
ax2.plot(bs7x,bs7y,'C1',label='number of bags')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')
loc = plticker.MultipleLocator(base=15)
ax1.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_locator(loc)
ss5,ds5 = 0,0
for i in range(0,len(avosum2015)-1):
    x = avosum2015[i+1][1]-avosum2015[i][1]
    y = bagsum2015[i+1][1]-bagsum2015[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        ss5 += 1
    else:
        ds5 += 1
plt.pie([ss5,ds5],labels=['same sign','different sign'],autopct='%1.1f%%')
ss6,ds6 = 0,0
for i in range(0,len(avosum2016)-1):
    x = avosum2016[i+1][1]-avosum2016[i][1]
    y = bagsum2016[i+1][1]-bagsum2016[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        ss6 += 1
    else:
        ds6 += 1
plt.pie([ss6,ds6],labels=['same sign','different sign'],autopct='%1.1f%%')
ss7,ds7 = 0,0
for i in range(0,len(avosum2017)-1):
    x = avosum2017[i+1][1]-avosum2017[i][1]
    y = bagsum2017[i+1][1]-bagsum2017[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        ss7 += 1
    else:
        ds7 += 1
plt.pie([ss7,ds7],labels=['same sign','different sign'],autopct='%1.1f%%')
plt.pie([ss5+ss6+ss7,ds5+ds6+ds7],labels=['same sign','different sign'],autopct='%1.1f%%')
ss,ds = 0,0
for i in range(0,len(avosum2015)-2):
    x = avosum2015[i+2][1]-avosum2015[i][1]
    y = bagsum2015[i+2][1]-bagsum2015[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        ss += 1
    else:
        ds += 1
for i in range(0,len(avosum2016)-2):
    x = avosum2016[i+2][1]-avosum2016[i][1]
    y = bagsum2016[i+2][1]-bagsum2016[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        ss += 1
    else:
        ds += 1
for i in range(0,len(avosum2017)-2):
    x = avosum2017[i+2][1]-avosum2017[i][1]
    y = bagsum2017[i+2][1]-bagsum2017[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        ss += 1
    else:
        ds += 1
plt.pie([ss,ds],labels=['same sign','different sign'],autopct='%1.1f%%')
def year_average_volume(y):
    avo_sum = {}
    date_cnt = {}
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        avo_sum[avocado['Date'][i]] = 0
        date_cnt[avocado['Date'][i]] = 0
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        avo_sum[avocado['Date'][i]] += avocado['Total Volume'][i]
        date_cnt[avocado['Date'][i]] += 1
    for k,v in date_cnt.items():
        avo_sum[k] /= v
    return sorted(avo_sum.items(),key=lambda x:x[0])
fid,ax = plt.subplots()
avovlm2015 = year_average_volume(2015)
av5x = []
av5y = []
for a,b in avovlm2015:
    av5x.append(a[5:])
    av5y.append(b)
ax.plot(av5x,av5y,label='2015')
avovlm2016 = year_average_volume(2016)
av6x = []
av6y = []
for a,b in avovlm2016:
    av6x.append(a[5:])
    av6y.append(b)
ax.plot(av6x,av6y,label='2016')
avovlm2017 = year_average_volume(2017)
av7x = []
av7y = []
for a,b in avovlm2017:
    av7x.append(a[5:])
    av7y.append(b)
ax.plot(av7x,av7y,label='2017')
plt.legend()
loc = plticker.MultipleLocator(base=15)
ax.xaxis.set_major_locator(loc)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(as5x,as5y,'C0',label='2015-price')
ax2.plot(av5x,av5y,'C1',label='2015-volume')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')
loc = plticker.MultipleLocator(base=15)
ax1.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_locator(loc)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(as6x,as6y,'C0',label='2016-price')
ax2.plot(av6x,av6y,'C1',label='2016-volume')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')
loc = plticker.MultipleLocator(base=15)
ax1.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_locator(loc)
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()
ax1.plot(as7x,as7y,'C0',label='2017-price')
ax2.plot(av7x,av7y,'C1',label='2017-volume')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')
loc = plticker.MultipleLocator(base=15)
ax1.xaxis.set_major_locator(loc)
ax2.xaxis.set_major_locator(loc)
a,b = 0,0
for i in range(0,len(avosum2015)-1):
    x = avosum2015[i+1][1]-avosum2015[i][1]
    y = avovlm2015[i+1][1]-avovlm2015[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        a += 1
    else:
        b += 1
for i in range(0,len(avosum2016)-1):
    x = avosum2016[i+1][1]-avosum2016[i][1]
    y = avovlm2016[i+1][1]-avovlm2016[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        a += 1
    else:
        b += 1
for i in range(0,len(avosum2017)-1):
    x = avosum2017[i+1][1]-avosum2017[i][1]
    y = avovlm2017[i+1][1]-avovlm2017[i][1]
    if (x<0 and y<0) or (x>0 and y>0):
        a += 1
    else:
        b += 1
plt.pie([a,b],labels=['same sign','different sign'],autopct='%1.1f%%')
def year_average_price_type(y,t):
    avo_sum = {}
    date_cnt = {}
    idx_to_date = {}
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        if avocado['type'][i] != t:
            continue
        avo_sum[avocado['Unnamed: 0'][i]] = 0
        date_cnt[avocado['Unnamed: 0'][i]] = 0
        idx_to_date[avocado['Unnamed: 0'][i]] = avocado['Date'][i]
    for i in range(len(avocado)):
        if avocado['year'][i] != y:
            continue
        if avocado['type'][i] != t:
            continue
        avo_sum[avocado['Unnamed: 0'][i]] += avocado['AveragePrice'][i]
        date_cnt[avocado['Unnamed: 0'][i]] += 1
    for k,v in date_cnt.items():
        avo_sum[k] /= v
    return avo_sum
avopri_15_0 = year_average_price_type(2015,0)
avopri_15_1 = year_average_price_type(2015,1)
avopri_16_0 = year_average_price_type(2016,0)
avopri_16_1 = year_average_price_type(2016,1)
avopri_17_0 = year_average_price_type(2017,0)
avopri_17_1 = year_average_price_type(2017,1)
plt.plot(list(avopri_15_0.keys()),list(avopri_15_0.values()),label='2015-convenical')
plt.plot(list(avopri_16_0.keys()),list(avopri_16_0.values()),label='2016-convenical')
plt.plot(list(avopri_17_0.keys()),list(avopri_17_0.values()),label='2017-convenical')
plt.plot(list(avopri_15_1.keys()),list(avopri_15_1.values()),label='2015-organic')
plt.plot(list(avopri_16_1.keys()),list(avopri_16_1.values()),label='2016-organic')
plt.plot(list(avopri_17_1.keys()),list(avopri_17_1.values()),label='2017-organic')
plt.legend()
a,b = 0,0
for i in range(0,len(avopri_15_0)-1):
    x = avopri_15_0[i+1]-avopri_15_0[i]
    y = avopri_15_1[i+1]-avopri_15_1[i]
    if (x<0 and y<0) or (x>0 and y>0):
        a += 1
    else:
        b += 1
plt.pie([a,b],labels=['same sign','different sign'],autopct='%1.1f%%')
a,b = 0,0
for i in range(0,len(avopri_16_0)-1):
    x = avopri_16_0[i+1]-avopri_16_0[i]
    y = avopri_16_1[i+1]-avopri_16_1[i]
    if (x<0 and y<0) or (x>0 and y>0):
        a += 1
    else:
        b += 1
plt.pie([a,b],labels=['same sign','different sign'],autopct='%1.1f%%')
a,b = 0,0
for i in range(0,len(avopri_17_0)-1):
    x = avopri_17_0[i+1]-avopri_17_0[i]
    y = avopri_17_1[i+1]-avopri_17_1[i]
    if (x<0 and y<0) or (x>0 and y>0):
        a += 1
    else:
        b += 1
plt.pie([a,b],labels=['same sign','different sign'],autopct='%1.1f%%')
a,b,c,d = 0,0,0,0
for i in range(0,len(avopri_15_0)-1):
    x = avopri_15_0[i+1]-avopri_15_0[i]
    y = avopri_15_1[i+1]-avopri_15_1[i]
    if x>0 and y<0:
        if bagsum2015[i+1]-bagsum2015[i] > 0:
            a += 1
        else:
            b += 1
    elif x<0 and y>0:
        if bagsum2015[i+1]-bagsum2015[i] > 0:
            c += 1
        else:
            d += 1
print(a,b,c,d)
plt.pie([a,b],labels=['bags increase','bags decrease'],autopct='%1.1f%%')

plt.pie([c,d],labels=['bags increase','bags decrease'],autopct='%1.1f%%')
a,b,c,d = 0,0,0,0
for i in range(0,len(avopri_16_0)-1):
    x = avopri_16_0[i+1]-avopri_16_0[i]
    y = avopri_16_1[i+1]-avopri_16_1[i]
    if x>0 and y<0:
        if bagsum2016[i+1]-bagsum2016[i] > 0:
            a += 1
        else:
            b += 1
    elif x<0 and y>0:
        if bagsum2016[i+1]-bagsum2016[i] > 0:
            c += 1
        else:
            d += 1
print(a,b,c,d)
plt.pie([a,b],labels=['bags increase','bags decrease'],autopct='%1.1f%%')

plt.pie([c,d],labels=['bags increase','bags decrease'],autopct='%1.1f%%')
a,b,c,d = 0,0,0,0
for i in range(0,len(avopri_17_0)-1):
    x = avopri_17_0[i+1]-avopri_17_0[i]
    y = avopri_17_1[i+1]-avopri_17_1[i]
    if x>0 and y<0:
        if bagsum2017[i+1]-bagsum2017[i] > 0:
            a += 1
        else:
            b += 1
    elif x<0 and y>0:
        if bagsum2017[i+1]-bagsum2017[i] > 0:
            c += 1
        else:
            d += 1
print(a,b,c,d)
plt.pie([a,b],labels=['bags increase','bags decrease'],autopct='%1.1f%%')

plt.pie([c,d],labels=['bags increase','bags decrease'],autopct='%1.1f%%')
avocado['XLarge Bags'].value_counts()
avocado['region'].value_counts()

avocado['type'] = avocado['type'].map({'conventional':0,'organic':1}).astype(int)
avocado['region'] = avocado['region'].map({'Spokane':0,'HartfordSpringfield':1,'SouthCentral':2,'BuffaloRochester':3,'CincinnatiDayton':4,'DallasFtWorth':5,'GrandRapids':6,'HarrisburgScranton':7,'Charlotte':8,'Tampa':9,'Louisville':10,'Southeast':11,'Portland':12,'RaleighGreensboro':13,'MiamiFtLauderdale':14,'Sacramento':15,'SanDiego':16,'Philadelphia':17,'Houston':18,'NorthernNewEngland':19,'Roanoke':20,'Detroit':21,'StLouis':22,'NewOrleansMobile':23,'PhoenixTucson':24,'Boston':25,'Boise':26,'TotalUS':27,'LasVegas':28,'Seattle':29,'Nashville':30,'SanFrancisco':31,'Denver':32,'Pittsburgh':33,'Indianapolis':34,'West':35,'Syracuse':36,'BaltimoreWashington':37,'Midsouth':38,'LosAngeles':39,'Columbus':40,'Orlando':41,'Northeast':42,'Plains':43,'GreatLakes':44,'SouthCarolina':45,'Albany':46,'Atlanta':47,'Jacksonville':48,'California':49,'Chicago':50,'RichmondNorfolk':51,'NewYork':52,'WestTexNewMexico':53}).astype(int)
avocado.head()
avocado_test = avocado[:idx]
avocado_answer = avocado[idx:]

y_avocado = avocado_test['AveragePrice']
X_avocado = avocado_test.drop('AveragePrice',axis=1).drop('Date',axis=1).drop('Total Volume',axis=1)
X_ans = avocado_answer.drop('AveragePrice',axis=1).drop('Date',axis=1).drop('Total Volume',axis=1)
from sklearn                        import metrics, svm
from sklearn.linear_model           import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

lab_enc = preprocessing.LabelEncoder()
y_avocado = lab_enc.fit_transform(y_avocado)

clf = LogisticRegression(penalty='l2',solver='sag',random_state=0)
clf.fit(X_avocado,y_avocado)

y_pred = clf.predict(X_ans)
xaxis = np.arange(0,len(y_avocado))
data = avocado
data['AveragePrice'] = lab_enc.fit_transform(data['AveragePrice'])
data.head()
def show2017():
    data_sum_t = {}
    data_sum_s = {}
    data_cnt_d = {}
    for i in range(0,len(data)):
        if data['year'][i] == 2017:
            data_sum_t[data['Unnamed: 0'][i]] = 0
            data_sum_s[data['Unnamed: 0'][i]] = 0
            data_cnt_d[data['Unnamed: 0'][i]] = 0
    idx = 0
    for i in range(0,len(data)):
        if data['year'][i] == 2017:
            data_sum_t[data['Unnamed: 0'][i]] += y_avocado[idx]
            data_sum_s[data['Unnamed: 0'][i]] += y_pred[idx]
            data_cnt_d[data['Unnamed: 0'][i]] += 1
            idx += 1
    data_ave_t = data_sum_t
    data_ave_s = data_sum_s
    for k,v in data_cnt_d.items():
        data_ave_t[k] /= v
        data_ave_s[k] /= v

    plt.plot(list(data_ave_t.keys()),list(data_ave_t.values()),label='Theoretical value')
    plt.plot(list(data_ave_s.keys()),list(data_ave_s.values()),label='predictical value')
    plt.legend()
show2017()
y_avocado = avocado_test['AveragePrice']
X_avocado = avocado_test.drop('AveragePrice',axis=1).drop('Date',axis=1)
X_ans = avocado_answer.drop('AveragePrice',axis=1).drop('Date',axis=1)
y_avocado = lab_enc.fit_transform(y_avocado)

clf = LogisticRegression()
clf = LogisticRegression(penalty='l2',solver='sag',random_state=0)
clf.fit(X_avocado,y_avocado)
y_pred = clf.predict(X_ans)
data = avocado
data['AveragePrice'] = lab_enc.fit_transform(data['AveragePrice'])
show2017()