# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
tr = pd.read_csv('../input/train.csv')
te = pd.read_csv('../input/test.csv')

t1 = tr.copy()
t2 = te.copy()
y = t1['price_range']
print(type(y))
print(tr.columns)
print(te.columns)
del t2['id']
del t1['price_range']
t = pd.concat([t1,t2], axis = 0)
print(t.columns)
print(t.isnull().sum())
temp1 = list(tr['battery_power'].copy())
temp2 = temp1.copy()
print(max(temp1))
print(min(temp1))
def temp_battery(a):
    b = a
    if(b < 600):
        c = 1
    elif(b >= 600 and b < 900 ):
        c = 2
    elif(b >= 900 and b < 1200):
        c = 3
    elif(b >= 1200 and b < 1500):
        c = 4
    elif(b >= 1500 and b < 1800):
        c = 5
    else:
        c = 6
    return c
print(t['battery_power'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,0:1] = temp_battery(temp1[i])
print(t['battery_power'].head(5))
temp2 = list(tr['clock_speed'].copy())
print(max(temp2))
print(min(temp2))
def temp_clock(a):
    b = a
    if(b < 0.5):
        c = 1
    elif(b >= 0.5 and b < 1.0 ):
        c = 2
    elif(b >= 1 and b < 1.5):
        c = 3
        
    elif(b >= 1.5 and b < 2):
        c = 4
    elif(b >= 2 and b < 2.5):
        c = 5
    else:
        c = 6
    return c
print(t['clock_speed'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,2:3] = temp_clock(temp2[i])
print(t['clock_speed'].head(5))
temp3 = list(tr['fc'].copy())
print(max(temp3))
print(min(temp3))
def temp_fc(a):
    b = a
    if(b < 3):
        c = 1
    elif(b >= 6 and b < 3 ):
        c = 2
    elif(b >= 6 and b < 9):
        c = 3
    elif(b >= 9 and b < 12):
        c = 4
    elif(b >= 12 and b < 15):
        c = 5
    elif(b >= 15 and b < 18):
        c = 6
    else:
        c = 7
    return c
print(t['fc'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,4:5] = temp_fc(temp3[i])
print(t['fc'].head(5))
temp4 = list(tr['int_memory'].copy())
print(max(temp4))
print(min(temp4))
def temp_intmemory(a):
    b = a
    if(b < 4):
        c = 1
    elif(b >= 4 and b < 8 ):
        c = 2
    elif(b >= 8  and b < 16):
        c = 3
    elif(b >= 16 and b < 32):
        c = 4
    elif(b >= 32 and b < 64):
        c = 5
    else:
        c = 6
    return c
print(t['int_memory'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,6:7] = temp_intmemory(temp4[i])
print(t['int_memory'].head(5))
print(tr.columns)
temp5 = list(tr['m_dep'].copy())
print(max(temp5))
print(min(temp5))
def temp_dep(a):
    b = a
    if(b < 0.2):
        c = 1
    elif(b >= 0.2 and b < 0.4 ):
        c = 2
    elif(b >= 0.4  and b < 0.6):
        c = 3
    elif(b >= 0.6 and b < 0.8):
        c = 4
    else:
        c = 5
    return c
print(t['m_dep'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,7:8] = temp_dep(temp5[i])
print(t['m_dep'].head(5))
temp6 = list(tr['mobile_wt'].copy())
print(max(temp6))
print(min(temp6))
def temp_weight(a):
    b = a
    if(b < 100):
        c = 1
    elif(b >= 100 and b < 130 ):
        c = 2
    elif(b >= 130  and b < 160):
        c = 3
    elif(b >= 160 and b < 190):
        c = 4
    else:
        c = 5
    return c
print(t['mobile_wt'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,8:9] = temp_weight(temp6[i])
print(t['mobile_wt'].head(5))
temp7 = list(tr['n_cores'].copy())
print(max(temp7))
print(min(temp7))
def temp_ncores(a):
    b = a
    if(b < 2):
        c = 1
    elif(b >= 2 and b < 4 ):
        c = 2
    elif(b >= 4  and b < 8):
        c = 3
    else:
        c = 4
    return c
print(tr.columns)
print(t['n_cores'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,9:10] = temp_ncores(temp7[i])
print(t['n_cores'].head(5))
print(tr.columns)
temp8 = list(tr['pc'].copy())
print(max(temp8))
print(min(temp8))
def temp_pc(a):
    b = a
    if(b < 4):
        c = 1
    elif(b >= 4 and b < 8 ):
        c = 2
    elif(b >= 8  and b < 12):
        c = 3
    elif(b >= 12 and b < 16 ):
        c = 4
    elif(b >= 16  and b < 20):
        c = 5
    else:
        c = 6
    return c
print(tr.columns)
print(t['pc'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,10:11] = temp_pc(temp8[i])
print(t['pc'].head(5))
print(tr.columns)
temp9 = list(tr['px_height'].copy())
print(max(temp9))
print(min(temp9))
def temp_pxheight(a):
    b = a
    if(b < 300):
        c = 1
    elif(b >= 600 and b < 300 ):
        c = 2
    elif(b >= 600  and b < 900):
        c = 3
    elif(b >= 900 and b < 1200 ):
        c = 4
    elif(b >= 1200  and b < 1500):
        c = 5
    elif(b >= 1500  and b < 1800):
        c = 5
    else:
        c = 6
    return c
print(tr.columns)
print(t['px_height'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,11:12] = temp_pxheight(temp9[i])
print(t['px_height'].head(5))
print(tr.columns)
temp10 = list(tr['px_width'].copy())
print(max(temp10))
print(min(temp10))
def temp_pxwidth(a):
    b = a
    if(b < 300):
        c = 1
    elif(b >= 300 and b < 600 ):
        c = 2
    elif(b >= 600  and b < 900):
        c = 3
    elif(b >= 900 and b < 1200 ):
        c = 4
    elif(b >= 1200  and b < 1500):
        c = 5
    elif(b >= 1500  and b < 1800):
        c = 5
    else:
        c = 6
    return c
print(tr.columns)
print(t['px_width'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,12:13] = temp_pxwidth(temp10[i])
print(t['px_width'].head(5))
print(tr.columns)
temp11 = list(tr['ram'].copy())
print(max(temp11))
print(min(temp11))
def temp_ram(a):
    b = a
    if(b < 512):
        c = 1
    elif(b >= 512 and b < 1024 ):
        c = 2
    elif(b >= 1024  and b < 2048):
        c = 3
    else:
        c = 4
    return c
print(tr.columns)
print(t['ram'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,13:14] = temp_ram(temp11[i])
print(t['ram'].head(5))
print(tr.columns)
temp12 = list(tr['sc_h'].copy())
print(max(temp12))
print(min(temp12))
def temp_sch(a):
    b = a
    if(b < 6):
        c = 1
    elif(b >= 6 and b < 9 ):
        c = 2
    elif(b >= 9  and b < 12):
        c = 3
    elif(b >= 12  and b < 15):
        c = 4
    elif(b >= 15  and b < 18):
        c = 5
    else:
        c = 6
    return c
print(tr.columns)
print(t['sc_h'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,14:15] = temp_sch(temp12[i])
print(t['sc_h'].head(5))
print(tr.columns)
temp13 = list(tr['sc_w'].copy())
print(max(temp13))
print(min(temp13))
def temp_scw(a):
    b = a
    if(b < 6):
        c = 1
    elif(b >= 6 and b < 9 ):
        c = 2
    elif(b >= 9  and b < 12):
        c = 3
    elif(b >= 12  and b < 15):
        c = 4
    elif(b >= 15  and b < 18):
        c = 5
    else:
        c = 6
    return c
print(tr.columns)
print(t['sc_w'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,15:16] = temp_scw(temp13[i])
print(t['sc_w'].head(5))
print(tr.columns)
temp14 = list(tr['talk_time'].copy())
print(max(temp14))
print(min(temp14))
def temp_talk(a):
    b = a
    if(b < 3):
        c = 1
    elif(b >= 6 and b < 3 ):
        c = 2
    elif(b >= 6 and b < 9):
        c = 3
    elif(b >= 9 and b < 12):
        c = 4
    elif(b >= 12 and b < 15):
        c = 5
    elif(b >= 15 and b < 18):
        c = 6
    else:
        c = 7
    return c
print(tr.columns)
print(t['talk_time'].head(5))
for i in range(0,len(temp1)):
    t.loc[i,16:17] = temp_scw(temp14[i])
print(t['talk_time'].head(5))
print(t.head(5))
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import matplotlib.pyplot as plt
var = list(t.columns)
X = t.copy()
corr = X.corr()
sm.graphics.plot_corr(corr)
plt.show()

thresh = 15
for i in np.arange(0,len(var)):
    vif = [variance_inflation_factor(X[var].values, ix) for ix in range(X[var].shape[1])]
    maxloc = vif.index(max(vif))
    print(i)
    if max(vif) > thresh:
        x1 = var[maxloc]
        del X[x1]
        var = list(X.columns)
    else:
        break
corr = X.corr()
sm.graphics.plot_corr(corr)
plt.show()
X1 = X.iloc[0:2000,:]
X_val = X.iloc[2000:,:]
y = list(y)
print(type(y))
X_train,X_test,y_train,y_test = train_test_split(X1,y, test_size = 0.2, random_state = 2)
lo = LogisticRegression(C = 10000)
lo.fit(X_train,y_train)
print(r2_score(lo.predict(X_train),y_train))
print(r2_score(lo.predict(X_test),y_test))
de = DecisionTreeClassifier()
de.fit(X_train,y_train)
print(r2_score(de.predict(X_train),y_train))
print(r2_score(de.predict(X_test),y_test))

re = RandomForestClassifier(n_estimators = 100, max_depth= 200, max_leaf_nodes = 12)
re.fit(X_train,y_train)
print(r2_score(re.predict(X_train),y_train))
print(r2_score(re.predict(X_test),y_test))
sv = SVC( gamma = 'auto', kernel = 'linear', random_state = 2017, C = 10000)
sv.fit(X_train,y_train)
print(r2_score(sv.predict(X_train),y_train))
print(r2_score(sv.predict(X_test),y_test))

