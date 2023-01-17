

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



sns.set(style="white", color_codes=True)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# ST,    STATE

# SEX    1 Male, 2 Female

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

             



# Careful -- you don't want to select every column in the Dataset. That's too much.    

fields = ["PUMA", "ST", "SEX","PWGTP","AGEP", "MSP","FOD1P","PERNP","ADJINC","SCHL",

          "SCH","ESR","WKHP","RAC1P"]



# Two files...each file is about 1.5G in size.

data_a=pd.read_csv("../input/ss15pusa.csv",skipinitialspace=True, usecols=fields)

data_b=pd.read_csv("../input/ss15pusb.csv",skipinitialspace=True, usecols=fields)
# Concatinate -- we'll work with everything.

d=pd.concat([data_a,data_b])



# Get the correct income in 2015 dollars

d['INCOME']=d['ADJINC']*d['PERNP']/1000000



# Take a look

d.head()
# We don't need these

del data_a, data_b
# For example how many males are in the US, age 30?

# SEX:  Male = 1,  Female = 2

d[(d['AGEP']==30) & (d['SEX'] == 1)]['PWGTP'].sum()
# Quick Check

# How many people in the US, age 0 to 99?

#   Note, AGEP only goes to 99.



# The number should be around 320 million... This

# is just to demonstrate the importance of using PWGTP in your

# calculations.



total_population = d[(d['AGEP']>=0) & (d['AGEP']<100)]['PWGTP'].sum()



# This is just for displaying ','

"{:,}".format(total_population)
# Okay, since I'm curious, how many people 90 or older?

total_population = d[(d['AGEP']>=90)]['PWGTP'].sum()

"{:,}".format(total_population)
# Graph the population by Race. Only take Males



white=[]

black=[]

other=[]



for i in range(0,100):

    white.append(d[(d['AGEP']==i) & (d['SEX']==1) & (d['RAC1P']==1)]['PWGTP'].sum())

    black.append(d[(d['AGEP']==i) & (d['SEX']==1) & (d['RAC1P']==2)]['PWGTP'].sum())

    other.append(d[(d['AGEP']==i) & (d['SEX']==1) & (d['RAC1P'] >2)]['PWGTP'].sum())
from matplotlib.ticker import FuncFormatter



fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False) 



ax.plot(white,color = 'blue')

ax.plot(black,color = 'black')

ax.plot(other,color = 'green')

ax.set_title("2015 US Population\n(Age and Race)", fontsize=12, color='darkslateblue')

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,}'.format(int(y))))



ax.grid(b=True, which='major', color='wheat', linestyle='-')



plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")    

 

plt.text(71, 1400000, "White Males", fontsize=10, ha="center",color="blue")   

plt.text(60, 351900, "Black Males", fontsize=10, ha="center",color="black") 

plt.text(30, 431900, "Other Males", fontsize=10, ha="center",color="green") 

plt.show()
# Graph the population by Race. Female.



whiteF=[]

blackF=[]

otherF=[]



for i in range(0,100):

    whiteF.append(d[(d['AGEP']==i) & (d['SEX']==2) & (d['RAC1P']==1)]['PWGTP'].sum())

    blackF.append(d[(d['AGEP']==i) & (d['SEX']==2) & (d['RAC1P']==2)]['PWGTP'].sum())

    otherF.append(d[(d['AGEP']==i) & (d['SEX']==2) & (d['RAC1P'] >2)]['PWGTP'].sum())
from matplotlib.ticker import FuncFormatter



fig, ax = plt.subplots()

ax.spines["top"].set_visible(False)    

ax.spines["bottom"].set_visible(False)    

ax.spines["right"].set_visible(False)    

ax.spines["left"].set_visible(False) 



ax.plot(white,color = 'blue')

ax.plot(black,color = 'black')

ax.plot(other,color = 'green')



ax.plot(whiteF,color = 'turquoise')

ax.plot(blackF,color = 'grey')

ax.plot(otherF,color = 'green')





ax.set_title("2015 US Population\n(Age and Race)", fontsize=12, color='darkslateblue')

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,}'.format(int(y))))



ax.grid(b=True, which='major', color='wheat', linestyle='-')



plt.tick_params(axis="both", which="both", bottom="off", top="off",    

                labelbottom="on", left="off", right="off", labelleft="on")    

 

plt.text(55, 1400000, "White Male", fontsize=10, ha="center",color="blue")   

plt.text(79, 1500000, "White Female", fontsize=10, ha="center",color="turquoise")



plt.text(40, 150900, "Black Male", fontsize=10, ha="center",color="black") 

plt.text(60, 350900, "Black Female", fontsize=10, ha="center",color="grey") 







plt.text(30, 431900, "Other", fontsize=10, ha="center",color="green") 

plt.show()
# Clean up

del white,black,other,whiteF,blackF
k=d[(d['AGEP']>=20) & (d['AGEP']<=60) & (d['SEX']==1) & (d['RAC1P']==1)][['PWGTP','AGEP','SCH','INCOME']]

numberOfRows=k['PWGTP'].sum()



I=0

A=np.zeros((numberOfRows+1,3))

def f(t):

    global A,I

    z=[t[1],t[2],t[3]]

    idx= int(t[0])

    

    for i in range(0,idx):

        A[I]=z

        I+=1

    

k.apply(f,axis=1);

df = pd.DataFrame(A,columns = ['AGEP','SCH','INCOME'])






plt.style.use('fivethirtyeight')



plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Ubuntu'

plt.rcParams['font.monospace'] = 'Ubuntu Mono'

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 10

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 10

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['legend.fontsize'] = 10

plt.rcParams['figure.titlesize'] = 12





a=df[(df['AGEP']>=20) &(df['AGEP']<30)]['INCOME']

b=df[(df['AGEP']>=30) &(df['AGEP']<40)]['INCOME']

c=df[(df['AGEP']>=40) &(df['AGEP']<55)]['INCOME']

sns.distplot(a,bins=100,  hist=False,  label="20 to 30 (Age in years)"  );

sns.distplot(b,bins=100,  hist=False,  label="30 to 40"  );

g=sns.distplot(c,bins=100,  hist=False,  label="40 to 55"  );

#ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,}'.format(int(x))))

sns.despine()

g.tick_params(labelsize=12,labelcolor="black")

#ax.xaxis.set_major_formatter(tck.FormatStrFormatter('{:,}'))

#x.xaxis.set_major_formatter(x_format)

plt.xlim(0, 200000)

plt.xticks([50000,100000,150000,200000], ['$50K', '$100K', '$120K', '$150K','$200K'])

plt.title("Income Distribution\nby Age, White Males", fontname='Ubuntu', fontsize=14,

            fontstyle='italic', fontweight='bold')



plt.legend();


plt.style.use('fivethirtyeight')



plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Ubuntu'

plt.rcParams['font.monospace'] = 'Ubuntu Mono'

plt.rcParams['font.size'] = 10

plt.rcParams['axes.labelsize'] = 10

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 10

plt.rcParams['xtick.labelsize'] = 8

plt.rcParams['ytick.labelsize'] = 8

plt.rcParams['legend.fontsize'] = 10

plt.rcParams['figure.titlesize'] = 12





a=df[(df['AGEP']>=20) &(df['AGEP']<30)]['INCOME']

b=df[(df['AGEP']>=30) &(df['AGEP']<40)]['INCOME']

c=df[(df['AGEP']>=40) &(df['AGEP']<55)]['INCOME']

sns.distplot(a,bins=100,  hist=False,  label="20 to 30 (Age in years)"  );

sns.distplot(b,bins=100,  hist=False,  label="30 to 40"  );

g=sns.distplot(c,bins=100,  hist=False,  label="40 to 55"  );

#ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:,}'.format(int(x))))

sns.despine()

g.tick_params(labelsize=12,labelcolor="black")

#ax.xaxis.set_major_formatter(tck.FormatStrFormatter('{:,}'))

#x.xaxis.set_major_formatter(x_format)

plt.xlim(0, 200000)

plt.xticks([50000,100000,150000,200000], ['$50K', '$100K', '$120K', '$150K','$200K'])

plt.title("Income Distribution\nby Age, White Males", fontname='Ubuntu', fontsize=14,

            fontstyle='italic', fontweight='bold')



plt.legend();