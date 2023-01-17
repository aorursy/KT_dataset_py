import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#Importing dataset
df_educ = pd.read_csv("../input/elsect_summary.csv")
#First look at the dataset
print(df_educ.shape)
print(df_educ.nunique())
# Looking Data types and nulls 
print(df_educ.info())
#How our data appears
df_educ.head()
plt.figure(figsize = (12,12))
plt.subplot(221)
g1 = sns.kdeplot(df_educ['TOTAL_REVENUE'], color='g')
g1 = sns.kdeplot(df_educ['TOTAL_EXPENDITURE'],color='r')
g1 = sns.kdeplot(df_educ['ENROLL'].dropna(), color='y')

plt.subplot(222)
g2 = sns.kdeplot(df_educ['FEDERAL_REVENUE'], color='g')
g2 = sns.kdeplot(df_educ['STATE_REVENUE'],color='r')
g2 = sns.kdeplot(df_educ['LOCAL_REVENUE'],color='y')

plt.subplot(223)
g3 = sns.kdeplot(df_educ['INSTRUCTION_EXPENDITURE'].dropna(), color='g')
g3 = sns.kdeplot(df_educ['SUPPORT_SERVICES_EXPENDITURE'],color='r')

plt.subplot(224)
g4 = sns.kdeplot(df_educ['OTHER_EXPENDITURE'].dropna(),color='y')
g4 = sns.kdeplot(df_educ['CAPITAL_OUTLAY_EXPENDITURE'],color='b')


plt.subplots_adjust(wspace = 0.4, hspace = 0.2,top = 0.9)
plt.show()
print('MIN TOTAL REVENUE')
print(df_educ["TOTAL_REVENUE"].min())
print('MEAN TOTAL REVENUE')
print(round(df_educ["TOTAL_REVENUE"].mean(),2))
print('MEDIAN TOTAL REVENUE')
print(df_educ["TOTAL_REVENUE"].median())
print('MAX TOTAL REVENUE')
print(df_educ["TOTAL_REVENUE"].max())
print('STD TOTAL REVENUE')
print(round(df_educ["TOTAL_REVENUE"].std(),2))

df_educ['AMOUNT_FINAL'] = df_educ["TOTAL_REVENUE"] - df_educ["TOTAL_EXPENDITURE"]

plt.figure(figsize = (8,5))
g = sns.kdeplot(df_educ['AMOUNT_FINAL'],color='b')
plt.show()
df_educ['REGION'] = np.nan

df_educ.loc[df_educ.STATE.isin(['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 
                                'Rhode Island','Vermont','New Jersey', 'New York',
                                'Pennsylvania']), 'REGION'] = 'Northweast'

df_educ.loc[df_educ.STATE.isin(['Illinois', 'Indiana', 'Michigan', 'Ohio','Wisconsin','Iowa', 'Kansas', 
                                'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 
                                'South Dakota']), 'REGION'] = 'Midwest'

df_educ.loc[df_educ.STATE.isin(['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 
                                'South Carolina', 'Virginia','District of Columbia','West Virginia',
                                'Alabama', 'Kentucky', 'Mississippi', 'Tennessee', 
                                'Arkansas', 'Louisiana', 'Oklahoma', 'Texas']), 'REGION'] = 'South'

df_educ.loc[df_educ.STATE.isin(['Arizona','Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 
                                'Utah','Wyoming','Alaska', 'California', 'Hawaii', 'Oregon',
                                'Washington']), 'REGION'] = 'West'
plt.figure(figsize = (15,6))

plt.subplot(2,2,1)
g = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=df_educ[df_educ['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("AMOUNT FINAL Northweast", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Net Income", fontsize=12)

plt.subplot(2,2,2)
g1 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=df_educ[df_educ['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("AMOUNT FINAL Midwest", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Net Income", fontsize=12)

plt.subplot(2,2,3)
g2 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=df_educ[df_educ['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("AMOUNT FINAL by South Region", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Net Income", fontsize=12)

plt.subplot(2,2,4)
g3 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=df_educ[df_educ['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("AMOUNT FINAL by West Region", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Net Income", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)

plt.show()
fig, ax = plt.subplots(1,3, sharex=False, sharey=True, 
                       figsize = (15,28))
ax = ax.flatten()

g = sns.boxplot(x="TOTAL_REVENUE",y="STATE", data=df_educ,ax=ax[0])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("TOTAL REVENUE BY STATE", fontsize=15)
g.set_xlabel("Revenue(log)", fontsize=12)
g.set_ylabel("States", fontsize=12)

g1 = sns.boxplot(x="TOTAL_EXPENDITURE",y="STATE", data=df_educ,ax=ax[1])
g1.set_xticklabels(g.get_xticklabels(),rotation=90)
g1.set_title("TOTAL EXPENDITURE BY STATE", fontsize=15)
g1.set_xlabel("Expenditure(log)", fontsize=12)
g1.set_ylabel("States", fontsize=12)

g2 = sns.boxplot(x="AMOUNT_FINAL",y="STATE",data=df_educ,ax=ax[2])
g2.set_xticklabels(g.get_xticklabels(),rotation=90)
g2.set_title("REV x EXPEND FINAL AMOUNT", fontsize=15)
g2.set_xlabel("Final Amount(US)", fontsize=12)
g2.set_ylabel("States", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)

plt.show()
state_amount_final = df_educ.groupby("STATE")['AMOUNT_FINAL'].mean()
print("Total of States with positive final Result")
print((state_amount_final > 0).sum())
year_amount_final = df_educ.groupby("YEAR")['AMOUNT_FINAL'].mean()

print("Positive final results since 92")
print((year_amount_final > 0).sum())

plt.figure(figsize = (15,6))
g = sns.barplot(x="YEAR", y="AMOUNT_FINAL",data=df_educ)
g.set_title("Final Result REVENUE X EXPENDITURE", fontsize=15)
g.set_xlabel("Years ", fontsize=12)
g.set_ylabel("Net Income", fontsize=12)

plt.show()
df_educ["President"] = np.nan

df_educ.loc[df_educ["YEAR"] <= 2000,"President"] = 'Bill Clinton'
df_educ.loc[(df_educ["YEAR"] > 2000) & (df_educ["YEAR"] <= 2009),"President"] = 'George Bush'
df_educ.loc[(df_educ["YEAR"] > 2009),"President"] = 'Barack Obama'

print(df_educ['President'].value_counts())
Bill = df_educ[df_educ["President"] == "Bill Clinton"]
Bush = df_educ[df_educ["President"] == "George Bush"]
Obama = df_educ[df_educ["President"] == "Barack Obama"]

plt.figure(figsize = (16,6))
plt.subplot(121)
g1 = sns.kdeplot(Bill['FEDERAL_REVENUE'], color='g')
g1 = sns.kdeplot(Bush['FEDERAL_REVENUE'], color='r')
g1 = sns.kdeplot(Obama['FEDERAL_REVENUE'], color='b')

plt.subplot(122)
g1 = sns.distplot(Bill['ENROLL'].dropna(), color='g')
g1 = sns.distplot(Bush['ENROLL'].dropna(), color='r')
g1 = sns.distplot(Obama['ENROLL'].dropna(), color='b')

plt.show()
plt.figure(figsize = (10,6))
g = sns.kdeplot(Bill['AMOUNT_FINAL'], color='g')
g = sns.kdeplot(Bush['AMOUNT_FINAL'], color='r')
g = sns.kdeplot(Obama['AMOUNT_FINAL'], color='b')
plt.figure(figsize = (15,30))

plt.subplot(6,2,1)
g = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bill[Bill['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("AMOUNT FINAL Northweast Bill Era", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,2)
g1 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bill[Bill['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("AMOUNT FINAL Midwest Bill Era", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,3)
g2 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bill[Bill['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("AMOUNT FINAL by South Region Bill Era", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,4)
g3 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bill[Bill['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("AMOUNT FINAL by West Region Bill Era", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,5)
g = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bush[Bush['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("AMOUNT FINAL Northweast Bush Era", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,6)
g1 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bush[Bush['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("AMOUNT FINAL Midwest Bush Era", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,7)
g2 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bush[Bush['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("AMOUNT FINAL by South Region Bush Era", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,8)
g3 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Bush[Bush['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("AMOUNT FINAL by West Region Bush Era", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,9)
g = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Obama[Obama['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("AMOUNT FINAL Northweast Obama Era", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,10)
g1 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Obama[Obama['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("AMOUNT FINAL Midwest Obama Era", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,11)
g2 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Obama[Obama['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("AMOUNT FINAL by South Region Obama Era", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Net Income", fontsize=12)

plt.subplot(6,2,12)
g3 = sns.barplot(x="STATE", y="AMOUNT_FINAL",
                data=Obama[Obama['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("AMOUNT FINAL by West Region Obama Era", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Net Income", fontsize=12)


plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)

plt.show()
fig, ax = plt.subplots(1, 3, sharex=False, sharey=True, 
                       squeeze=False, figsize = (15,28))
ax = ax.flatten()

g = sns.barplot(x="ENROLL",y="STATE", data=Bill,ax=ax[0])
g.set_title("ENROLLs Bill Clinton Era", fontsize=15)
g.set_xlabel("ENROLL COUNT", fontsize=12)
g.set_ylabel("STATES", fontsize=12)

g1 = sns.barplot(x="ENROLL",y="STATE", data=Bush,ax=ax[1])
g1.set_title("ENROLLs Bush Era", fontsize=15)
g1.set_xlabel("ENROLL COUNT ", fontsize=12)


g2 = sns.barplot(x="ENROLL",y="STATE", data=Obama,ax=ax[2])
g2.set_title("ENROLLs Obama Era", fontsize=15)
g2.set_xlabel("ENROLL COUNT ", fontsize=12)


plt.show()
df_educ['revenue_per_student'] = df_educ['TOTAL_REVENUE'] / len(df_educ['ENROLL'])
df_educ['expend_per_student'] = df_educ['TOTAL_EXPENDITURE'] /  len(df_educ['ENROLL'])

Bill = df_educ[df_educ["President"] == "Bill Clinton"]
Bush = df_educ[df_educ["President"] == "George Bush"]
Obama = df_educ[df_educ["President"] == "Barack Obama"]

plt.figure(figsize = (15,30))

plt.subplot(6,2,1)
g = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bill[Bill['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Revenue per capita Northweast Bill Era", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,2)
g1 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bill[Bill['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Revenue per capita Midwest Bill era", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,3)
g2 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bill[Bill['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Revenue per capita South Region Bill Era", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,4)
g3 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bill[Bill['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Revenue per capita West Region Bill Era", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,5)
g = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bush[Bush['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Revenue per capita Northweast Bush Era", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,6)
g1 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bush[Bush['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Revenue per capita Midwest Bush Era", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,7)
g2 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bush[Bush['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Revenue per capita South Region Bush Era", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,8)
g3 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Bush[Bush['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Revenue per capita West Region Bush Era", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,9)
g = sns.barplot(x="STATE", y="revenue_per_student",
                data=Obama[Obama['REGION'] == "Northweast"])
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Revenue per capita Northweast Obama Era", fontsize=15)
g.set_xlabel("", fontsize=12)
g.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,10)
g1 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Obama[Obama['REGION'] == "Midwest"])
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
g1.set_title("Revenue per capita Midwest Obama Era", fontsize=15)
g1.set_xlabel("", fontsize=12)
g1.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,11)
g2 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Obama[Obama['REGION'] == "South"])
g2.set_xticklabels(g2.get_xticklabels(),rotation=45)
g2.set_title("Revenue per capita South Region Obama Era", fontsize=15)
g2.set_xlabel("", fontsize=12)
g2.set_ylabel("Revue per Student", fontsize=12)

plt.subplot(6,2,12)
g3 = sns.barplot(x="STATE", y="revenue_per_student",
                data=Obama[Obama['REGION'] == "West"])
g3.set_xticklabels(g3.get_xticklabels(),rotation=45)
g3.set_title("Revenue per capita West Region Obama Era", fontsize=15)
g3.set_xlabel("", fontsize=12)
g3.set_ylabel("Revue per Student", fontsize=12)

plt.subplots_adjust(wspace = 0.2, hspace = 0.7,top = 0.9)

plt.show()
plt.figure(figsize =( 22,15))

g = sns.lmplot(x="YEAR",y='revenue_per_student', 
           data=df_educ, col="President",sharey=True)
plt.show()

