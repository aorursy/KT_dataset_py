import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



sns.set(style="white", color_codes=True)



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# ST,    STATE

# SEX    1 Make, 2 Female

# PWGTP  (Weighting of record. Very important)

# AGEP   Age

# MSP    Married, spouse present/spouse absent

# FOD1P  Recoded field of degree - first entry (College)

# PERNP  Total person's earnings

# ADJINC Use ADJINC to adjust PERNP to constant dollars.

# SCHL   Highest education allocation flag

# SCH    School enrollment

# ESR    Employment status recode

# WKHP   Usual hours worked per week past 12 months allocation flag

# RAC1P  Recoded detailed race code 1) White  2) Black ... see below

# FOD1P  Field of Degree

# OCCP   Occupational Code

# CIT    Citizenship status

# CITWP  Year of naturalization write-in

# COW    Class of Worker

             



# Careful -- you don't want to select every column in the Dataset. That's too much.    

fields = ["PUMA", "ST", "SEX","PWGTP","AGEP", "MSP","FOD1P","PERNP","ADJINC","SCHL",

          "SCH","ESR","WKHP","RAC1P","FOD1P","OCCP","CIT",'CITWP','COW']



# Two files...each file is about 1.5G in size.

data_a=pd.read_csv("../input/ss15pusa.csv",skipinitialspace=True, usecols=fields)

data_b=pd.read_csv("../input/ss15pusb.csv",skipinitialspace=True, usecols=fields)
# Concatinate -- we'll work with everything.

d=pd.concat([data_a,data_b])



# Get the correct income in 2015 dollars

d['INCOME']=d['ADJINC']*d['PERNP']/1000000



del d['ADJINC']





d.head()
# We only want certain occupations

# No managers.. 110: 'MGR-COMPUTER AND INFORMATION SYSTEMS MANAGERS'

my_occpD = {1005: 'CMM-COMPUTER AND INFORMATION RESEARCH SCIENTISTS',

 1006: 'CMM-COMPUTER SYSTEMS ANALYSTS',

 1007: 'CMM-INFORMATION SECURITY ANALYSTS',

 1010: 'CMM-COMPUTER PROGRAMMERS',

 1020: 'CMM-SOFTWARE DEVELOPERS, APPLICATIONS AND SYSTEMS SOFTWARE',

 1030: 'CMM-WEB DEVELOPERS',

 1050: 'CMM-COMPUTER SUPPORT SPECIALISTS',

 1060: 'CMM-DATABASE ADMINISTRATORS',

 1105: 'CMM-NETWORK AND COMPUTER SYSTEMS ADMINISTRATORS',

 1106: 'CMM-COMPUTER NETWORK ARCHITECTS',

 1107: 'CMM-COMPUTER OCCUPATIONS, ALL OTHER',

 1200: 'CMM-ACTUARIES',

 1220: 'CMM-OPERATIONS RESEARCH ANALYSTS',

 1240: 'CMM-MISCELLANEOUS MATHEMATICAL SCIENCE OCCUPATIONS,'}

my_occp=list(my_occpD.keys())



#my_occp=[110,1005,1006,1010,1030,1050,1060,1105,1106,1107]

d=d[(d['OCCP'].isin(my_occp))]



d = d[['PUMA', 'ST', 'PWGTP', 'AGEP', 'CIT', 'CITWP', 'COW', 'SCH', 'SCHL',

       'SEX', 'WKHP', 'ESR', 'FOD1P', 'MSP', 'OCCP', 'PERNP', 'RAC1P',

       'INCOME']]

numberOfRows=d['PWGTP'].sum()



cols=['ST', 'AGEP', 'CIT', 'CITWP', 'COW', 'SCH', 'SCHL',

       'SEX', 'WKHP', 'ESR', 'FOD1P', 'MSP', 'OCCP', 'PERNP', 'RAC1P',

       'INCOME']

I=0

A=np.zeros((numberOfRows,len(cols)),dtype=np.int64)

def f(t):

    global A,I

    z=[int(i) for i in t]

    idx= int(t[0])

    

    for i in range(0,idx):

        A[I]=z

        I+=1



d.fillna(-1, inplace=True) # Can't have NaN when we go to int       

d[cols].apply(f,axis=1);

A=A.astype(int)



d = pd.DataFrame(A,columns = cols)

d = d[d['INCOME']>=0]  # 

d = d[d['AGEP']>15]

d.head()
def percentile(n):

    def percentile_(x):

        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n

    return percentile_



g_us = d[(d['INCOME']>0) & (d['AGEP']<65) & (d['CITWP']==-1)].groupby(['AGEP']).INCOME.agg([percentile(75),percentile(50),percentile(35)])

g_us = g_us.reset_index()

# We're sorting on 75th percentile

g_us.sort_values(by=['AGEP'],ascending=True,inplace=True)

g_us.head()







g_f = d[(d['INCOME']>0) &  (d['AGEP']<65) &(d['CITWP']!=-1)].groupby(['AGEP']).INCOME.agg([percentile(75),percentile(50),percentile(35)])

g_f = g_f.reset_index()

# We're sorting on 75th percentile

g_f.sort_values(by=['AGEP'],ascending=True,inplace=True)

g_f.head()










from matplotlib.ticker import FuncFormatter



fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False) 



#ax.plot(d[(d['CITWP']==-1)][['AGEP','INCOME']].head() ,color = 'blue')

ax.plot(g_us['AGEP'],g_us['percentile_50'],color='blue')

#ax.plot(g_us['AGEP'],g_us['percentile_50'],color='grey')

#ax.plot(g_us['AGEP'],g_us['percentile_35'],color='silver')



ax.plot(g_f['AGEP'],g_f['percentile_50'],color='black')

#ax.plot(g_f['AGEP'],g_f['percentile_50'],color='grey')

#ax.plot(g_f['AGEP'],g_f['percentile_35'],color='silver')







ax.set_title("Computer Workers: Foreign vs U.S. Born\n2015 Median Income", fontsize=12, color='darkslateblue')

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '${:,}'.format(int(y))))



ax.grid(b=True, which='major', color='wheat', linestyle='-')



plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")    

 

plt.text(50, 65000, "U.S. Born", fontsize=10, ha="center",color="blue")   

plt.text(65, 107000, "Foreign Born", fontsize=10, ha="center",color="black") 





#plt.text(55,  -1000, "Foreign Born", fontsize=10, ha="center",color="black")

#plt.text(30, 431900, "Other Males", fontsize=10, ha="center",color="green") 

label = ax.set_xlabel('Age', fontsize = 9)

ax.xaxis.set_label_coords(.5, -0.065)





plt.text(-1, -23000, "Data source: 2015 American Community Survey"    

       "\nAuthor: Mike Chirico (mchirico@gmail.com)"    

       "\nNote: These are all U.S. Citizens."    

       "  This does not include H1B1 visa.", fontsize=8)    



plt.show()
g_us=d[(d['CITWP']==-1)].groupby(['AGEP'])['ST'].agg(['count'])

g_us=g_us.reset_index()





g_f=d[(d['CITWP']>-1)].groupby(['AGEP'])['ST'].agg(['count'])

g_f=g_f.reset_index()





from matplotlib.ticker import FuncFormatter



fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False) 



#ax.plot(d[(d['CITWP']==-1)][['AGEP','INCOME']].head() ,color = 'blue')

ax.plot(g_us['AGEP'],g_us['count'],color='blue')

ax.plot(g_f['AGEP'],g_f['count'],color='black')





ax.set_title("2015 Population of Computer Workers\n(Foreign vs U.S. Born)", fontsize=12, color='darkslateblue')

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,}'.format(int(y))))



ax.grid(b=True, which='major', color='wheat', linestyle='-')



plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")    

 

plt.text(71, 15000, "U.S. Born", fontsize=10, ha="center",color="blue")   

plt.text(60,  4800, "Foreign Born", fontsize=10, ha="center",color="black") 





#plt.text(55,  -1000, "Foreign Born", fontsize=10, ha="center",color="black")

#plt.text(30, 431900, "Other Males", fontsize=10, ha="center",color="green") 

label = ax.set_xlabel('Age', fontsize = 9)

ax.xaxis.set_label_coords(.5, -0.065)



plt.text(-1, -6000, "Data source: 2015 American Community Survey"    

       "\nAuthor: Mike Chirico (mchirico@gmail.com)"    

       "\nNote: These are all U.S. Citizens."    

       "  This does not include H1B1 visa.", fontsize=8)    

plt.show()