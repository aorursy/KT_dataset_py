# Loading Important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import date, timedelta, datetime
# Function to Load India Data 
def load_country_data():
    df = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
    return df
# Load Data
df = load_country_data()
df.head()
df[df["State/UnionTerritory"]=="Telengana"][-12:]
# Listing all the states
df11 = df["State/UnionTerritory"].unique()
print(type(df11))
df11=df11[:-2]
print(df11)
print(len(df11))
def split_state(name):
    s = df[(df["State/UnionTerritory"] == name)]
    return s.iloc[-45:]
dfS = {}
for x in df11:
    dfS[x] = split_state(x)
    
def split_state(name):
    s = df[(df["State/UnionTerritory"] == name)]
    return s
dfSS = {}
for x in df11:
    dfSS[x] = split_state(x)
# print(dfSS)
#Growth rate calculation 
import math
def growth_fac(a):
    ev = a[0][-1]
    bv = a[0][0]
    if ev-bv == 0 or ev == 0 or bv ==0:
        return 0
    e = math.log(ev/bv)
    gf = (e/len(a[0]))
    print(gf, len(a[0]))
    return gf*100
a=[[762, 803, 814, 848, 905, 968, 1000, 1046, 1102, 1175, 1236, 1326, 1415, 1511, 1568, 1661, 1749, 1809, 1888, 1943, 2008, 2110, 2152, 2152, 2306, 2439, 2640, 2841, 3048, 3174, 3341, 3452, 3559, 3708, 3820, 3963, 4095, 4257, 4438, 4634, 4862, 4995, 5199, 5371, 5616]]
growth_fac(a)
dfSS['Kerala']
def split(state, s1, e1, c):
    try:
        print(state,s1,e1)
        x = dfSS[state]
        sd = x[x["Date"]==s1]["Sno"].tolist()
        sd1 = int(sd[0])
        ed = x[x["Date"]==e1]["Sno"].tolist()
        ed1 = int(ed[0])
        cu=[];de=[];co=[]
        for i in range(sd1, ed1):
            if not (x[x["Sno"]==float(i)]).empty: 
                a= x[x["Sno"]==float(i)].values.tolist()
                cu.append(a[0][-3])
                de.append(a[0][-2])
                co.append(a[0][-1])
        cu_gf = growth_fac([cu]);de_gf = growth_fac([de]);co_gf = growth_fac([co])
    except:
        c+=1
        print("Exception", c)
        return 0,0,0,c        
    return cu_gf, de_gf, co_gf, c

CUGF0={};DEGF0={};CONGF0={};count=0
CUGF1={};DEGF1={};CONGF1={};
CUGF2={};DEGF2={};CONGF2={};
CUGF3={};DEGF3={};CONGF3={};
CUGF4={};DEGF4={};CONGF4={};
CUGF5={};DEGF5={};CONGF5={};
phase0={};phase1={};phase2={};phase3={};phase4={};phase5={};
for i in df11:
    # Phases dates data in form of 2 seperate list 
    # sd -> Start Date; ed -> End Date
    sd=["23/03/20","15/04/20","04/05/20","18/05/20","01/06/20","01/07/20"]
    ed=["14/04/20","03/05/20","17/05/20","31/05/20","30/06/20","24/07/20"]
    # Storing the data according to the phases
    for j in range(len(sd)):
        k=i
        if j==0:
            CUGF0[k], DEGF0[k], CONGF0[k], exceptionCount = split(i,sd[j],ed[j],count)
        if j==1:
            CUGF1[k], DEGF1[k], CONGF1[k], exceptionCount = split(i,sd[j],ed[j],count)
        if j==2:
            CUGF2[k], DEGF2[k], CONGF2[k], exceptionCount = split(i,sd[j],ed[j],count)
        if j==3:
            CUGF3[k], DEGF3[k], CONGF3[k], exceptionCount = split(i,sd[j],ed[j],count)
        if j==4:
            CUGF4[k], DEGF4[k], CONGF4[k], exceptionCount = split(i,sd[j],ed[j],count)
        if j==5:
            CUGF5[k], DEGF5[k], CONGF5[k], exceptionCount = split(i,sd[j],ed[j],count)

CGF={};DGF={};ConGF={}
for i in dfS:
    cured=[];deaths=[];confirmed=[]
    x=dfS[i]["State/UnionTerritory"].iloc[0]
    cured.append(dfS[i]["Cured"].values.tolist())
    deaths.append(dfS[i]["Deaths"].values.tolist())
    confirmed.append(dfS[i]["Confirmed"].values.tolist())
    
    CGF[x] = growth_fac(cured);DGF[x] = growth_fac(deaths);ConGF[x] = growth_fac(confirmed)
#     print(len(CGF),len(ConGF))
#     print(CGF,DGF,ConGF); print();print()
print(CGF)
print();print()
print(DGF)
print();print()
print(ConGF)
print("Cured Growth Rates in States During Various Phases: \n")
for key, value in CUGF0.items():
    print(key, ' : ', value)
print("\n---------------------------END-----------------------------------\n")
print("Death Growth Rates in States During Various Phases: \n")
for key, value in DEGF0.items():
    print(key, ' : ', value)
print("\n---------------------------END-----------------------------------\n")
print("Confirmed Growth Rates in States During Various Phases: \n")
for key, value in CONGF0.items():
    print(key, ' : ', value)
print("\n---------------------------END-----------------------------------\n")
print(len(CUGF0))
print("exceptionCount:",exceptionCount)
print("Recent Cured Growth Rates in the states:\n")
for key, value in CGF.items():
    print(key, ' : ', value)
print("\n---------------------------END-----------------------------------\n")

print("Recent Death Growth Rates in the states:\n")
for key, value in DGF.items():
    print(key, ' : ', value)
print("\n---------------------------END-----------------------------------\n")

print("Recent Confirmed Growth Rates in the states:\n")
for key, value in ConGF.items():
    print(key, ' : ', value)
print("\n---------------------------END-----------------------------------\n")

# CGF,DGF,ConGF - Recent Trends
# CUGF, DEGF, CONGF - Phase Trends
# Correlations as measure of relation
cured = pd.DataFrame(columns=["state","cured"])
for key, value in CGF.items():
    cured = cured.append({"state": key, "cured":value},ignore_index=True)
#     print(key, ' : ', value)
print(cured.head())
phase1Cured = pd.DataFrame(columns=["state","cured"])
for key, value in CUGF0.items():
    phase1Cured = phase1Cured.append({"state": key, "cured":value},ignore_index=True)
print(phase1Cured.head())
cured.corrwith(phase1Cured, axis = 0)
# print(CGF["Kerala"])
# correlation = df13.corr()

# Data Preparation
cured = pd.DataFrame(CGF,index=[0])
death = pd.DataFrame(DGF,index=[0])
confirmed = pd.DataFrame(ConGF,index=[0])
print(death)
df2 = cured.append(death)
df2 = df2.append(confirmed)
print(df2)

cured0 = pd.DataFrame(CUGF0,index=[0])
death0 = pd.DataFrame(DEGF0,index=[0])
confirmed0 = pd.DataFrame(CONGF0,index=[0])
dfp0 = cured0.append(death0)
dfp0 = dfp0.append(confirmed0)
print(dfp0)


cured1 = pd.DataFrame(CUGF1,index=[0])
death1 = pd.DataFrame(DEGF1,index=[0])
confirmed1 = pd.DataFrame(CONGF1,index=[0])
dfp1 = cured1.append(death1)
dfp1 = dfp1.append(confirmed1)
print(dfp1)


cured2 = pd.DataFrame(CUGF2,index=[0])
death2 = pd.DataFrame(DEGF2,index=[0])
confirmed2 = pd.DataFrame(CONGF2,index=[0])
dfp2 = cured2.append(death2)
dfp2 = dfp2.append(confirmed2)
print(dfp2)


cured3 = pd.DataFrame(CUGF3,index=[0])
death3 = pd.DataFrame(DEGF3,index=[0])
confirmed3 = pd.DataFrame(CONGF3,index=[0])
dfp3 = cured3.append(death3)
dfp3 = dfp3.append(confirmed3)
print(dfp3)


cured4 = pd.DataFrame(CUGF4,index=[0])
death4 = pd.DataFrame(DEGF4,index=[0])
confirmed4 = pd.DataFrame(CONGF4,index=[0])
dfp4 = cured4.append(death4)
dfp4 = dfp4.append(confirmed4)
print(dfp4)


cured5 = pd.DataFrame(CUGF5,index=[0])
death5 = pd.DataFrame(DEGF5,index=[0])
confirmed5 = pd.DataFrame(CONGF5,index=[0])
dfp5 = cured5.append(death5)
dfp5 = dfp5.append(confirmed5)
print(dfp5)


# for key, value in CGF.items():
#     cured = cured.append({"state": key, "cured":value},ignore_index=True)

# Correlation calculations
corr0 = df2.corrwith(dfp0, axis = 0)
corr1 = df2.corrwith(dfp1, axis = 0)
corr2 = df2.corrwith(dfp2, axis = 0)
corr3 = df2.corrwith(dfp3, axis = 0)
corr4 = df2.corrwith(dfp4, axis = 0)
corr5 = df2.corrwith(dfp5, axis = 0)
# print(corr0,corr1,corr2,corr3,corr4,corr5)
dfCorr0 = corr0.reset_index()
dfCorr1 = corr1.reset_index()
dfCorr2 = corr2.reset_index()
dfCorr3 = corr3.reset_index()
dfCorr4 = corr4.reset_index()
dfCorr5 = corr5.reset_index()

dfCorr0 = dfCorr0.rename(columns={'index': 'States', 0: 'Phase1'})
dfCorr1 = dfCorr1.rename(columns={'index': 'States', 0: 'Phase2'})
dfCorr2 = dfCorr2.rename(columns={'index': 'States', 0: 'Phase3'})
dfCorr3 = dfCorr3.rename(columns={'index': 'States', 0: 'Phase4'})
dfCorr4 = dfCorr4.rename(columns={'index': 'States', 0: 'Phase5'})
dfCorr5 = dfCorr5.rename(columns={'index': 'States', 0: 'Phase6'})
print(dfCorr0.head())
print(dfCorr1.head())
print(dfCorr2.head())
print(dfCorr3.head())
print(dfCorr4.head())
print(dfCorr5.head())

# print(dfCorr0.info)

# print(dfCorr0)
# print(type(dfCorr1))
# print(type(corr0))
# Final Correlation Analysis Results
df21 = pd.merge(dfCorr0,dfCorr1,on="States")
df21 = pd.merge(df21,dfCorr2,on="States")
df21 = pd.merge(df21,dfCorr3,on="States")
df21 = pd.merge(df21,dfCorr4,on="States")
df21 = pd.merge(df21,dfCorr5,on="States")
df21
df22 = df21.set_index("States")
m = df22.idxmax(axis=1, skipna=True)
m
n = df22.max(axis=1)
n
# Minimum growth usuage and 0 calculation
p1 = pd.DataFrame(CUGF0,index=[0])
p2 = pd.DataFrame(CUGF1,index=[0])
p3 = pd.DataFrame(CUGF2,index=[0])
p4 = pd.DataFrame(CUGF3,index=[0])
p5 = pd.DataFrame(CUGF4,index=[0])
p6 = pd.DataFrame(CUGF5,index=[0])
dfppCGF = p1.append(p2)
dfppCGF = dfppCGF.append(p3)
dfppCGF = dfppCGF.append(p4)
dfppCGF = dfppCGF.append(p5)
dfppCGF = dfppCGF.append(p6)
print(dfppCGF)

p1 = pd.DataFrame(DEGF0,index=[0])
p2 = pd.DataFrame(DEGF1,index=[0])
p3 = pd.DataFrame(DEGF2,index=[0])
p4 = pd.DataFrame(DEGF3,index=[0])
p5 = pd.DataFrame(DEGF4,index=[0])
p6 = pd.DataFrame(DEGF5,index=[0])
dfppDGF = p1.append(p2)
dfppDGF = dfppDGF.append(p3)
dfppDGF = dfppDGF.append(p4)
dfppDGF = dfppDGF.append(p5)
dfppDGF = dfppDGF.append(p6)
print(dfppDGF)

p1 = pd.DataFrame(CONGF0,index=[0])
p2 = pd.DataFrame(CONGF1,index=[0])
p3 = pd.DataFrame(CONGF2,index=[0])
p4 = pd.DataFrame(CONGF3,index=[0])
p5 = pd.DataFrame(CONGF4,index=[0])
p6 = pd.DataFrame(CONGF5,index=[0])
dfppCONGF = p1.append(p2)
dfppCONGF = dfppCONGF.append(p3)
dfppCONGF = dfppCONGF.append(p4)
dfppCONGF = dfppCONGF.append(p5)
dfppCONGF = dfppCONGF.append(p6)
print(dfppCONGF)



dd = dfS[23][["Date","Cured"]].copy()
dd["dif"] = dd.Cured.diff()
dd1 = dd.set_index("Date")
dd1
plot_df(dd1, x=dd1.index, y=dd1["dif"], title='Past Cured trend analysis.')    
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
d1 = dfS[0].set_index("Date")
plot_df(df, x=d1.index, y=d1["Cured"], title='Past Cured trend analysis.')    
d1 = dfS[0].set_index("Date")
plt.figure(figsize=(40,10))
plt.xlabel('Dates')
plt.ylabel('Active')
plt.plot(d1['Confirmed'])
plt.grid(True)
plt.show()
from matplotlib import pyplot
from pandas.plotting import lag_plot
d1 = dfS[0].set_index("Date")
plt.figure(figsize=(40,10))
plt.xlabel('Dates')
plt.ylabel('Active')
lag_plot(d1['Confirmed'])
pyplot.show()
