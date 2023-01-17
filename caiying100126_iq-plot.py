import pandas as pd

import numpy as np

import random

import scipy.stats as st

import scipy as sc

from scipy.stats import norm

import matplotlib.pyplot as plt

import math



tt_pop=5867920 #total population in Singapore

#calculate the Sheetsingapore-"singapore" column--number of people in a certain group of IQ，round the numbers,get the"rounded"column in Sheet"singapore"

sd0 = tt_pop*(st.norm(100,15).cdf(20.5) - st.norm(100,15.68).cdf(0))

sd0=int(sd0)



sd1 = tt_pop*(st.norm(100,15).cdf(35.5) - st.norm(100,15).cdf(0))

sd1=int(sd1)



sd2 = tt_pop*(st.norm(100,15).cdf(50.5) - st.norm(100,15).cdf(35.5))

sd2=int(sd2)



sd3 = tt_pop*(st.norm(100,15).cdf(70.5) - st.norm(100,15).cdf(50.5))

sd3=int(sd3)



sd4 = tt_pop*(st.norm(100,15).cdf(85.5) - st.norm(100,15).cdf(70.5))

sd4=int(sd4)



sd5 = tt_pop*(st.norm(100,15).cdf(100) - st.norm(100,15).cdf(85.5))

sd5=int(sd5)



sd6 = tt_pop*(st.norm(100,15).cdf(114.5) - st.norm(100,15).cdf(100))

sd6=int(sd6)



sd7 = tt_pop*(st.norm(100,15).cdf(129.5) - st.norm(100,15).cdf(114.5))

sd7=int(sd7)



sd8 = tt_pop*(st.norm(100,15).cdf(144.5) - st.norm(100,15).cdf(129.5))

sd8=int(sd8)



sd9 = tt_pop*(st.norm(100,15).cdf(159.5) - st.norm(100,15).cdf(144.5))

sd9=int(sd9)



sd10 = tt_pop*(st.norm(100,15).cdf(169.5) - st.norm(100,15).cdf(159.5))

sd10=int(sd10)



sd11 = tt_pop*(st.norm(100,15).cdf(179.5) - st.norm(100,15).cdf(169.5))

sd11=int(sd11)



sd12 = tt_pop*(1-st.norm(100,15).cdf(179.5))

sd12=int(sd12)



#get the rounded numbers in column "5867920"in sheet"singapore_2"

rsd0=sd0+sd1        #50

rsd1=round(sd2,-2)  #2800

rsd2=round(sd3,-4)  #140000

rsd3=round(sd4,-4)  #830000

rsd4=round(sd5,-6)  #2000000

rsd5=round(sd6,-6)  #2000000

rsd6=round(sd7,-4)  #830000

rsd7=round(sd8,-4)  #140000

rsd8=round(sd9,-2)  #8600

rsd9=round(sd10,-2) #200

rsd10=sd11+sd12     #10



#get the sheet"singapore2" 100%,which is related to the sheet"singapore"non-overlapping%

vpsd0=sd0/tt_pop+sd1/tt_pop

vpsd1=sd2/tt_pop

vpsd2=sd3/tt_pop

vpsd3=sd4/tt_pop

vpsd4=sd5/tt_pop

vpsd5=sd6/tt_pop

vpsd6=sd7/tt_pop

vpsd7=sd8/tt_pop

vpsd8=sd9/tt_pop

vpsd9=sd10/tt_pop

vpsd10=sd11/tt_pop+sd12/tt_pop

sum_vpsd=vpsd0+vpsd1+vpsd2+vpsd3+vpsd4+vpsd5+vpsd6+vpsd7+vpsd8+vpsd9+vpsd10



#transfer the 100% into required format,"rp"means required percentage

rp0='{:.5%}'.format(vpsd0)     #0.00085%

rp1='{:.3%}'.format(vpsd1)     #0.047%

rp2='{:.1%}'.format(vpsd2)     #2.4%

rp3='{:.0%}'.format(vpsd3)     #14%

rp4='{:.0%}'.format(vpsd4)     #33%

rp5='{:.0%}'.format(vpsd5)     #33%

rp6='{:.0%}'.format(vpsd6)     #14%

rp7='{:.1%}'.format(vpsd7)     #2.3%

rp8='{:.2%}'.format(vpsd8)     #0.15%

rp9='{:.4%}'.format(vpsd9)     #0.0035%

rp10='{:.5%}'.format(vpsd10)   #0.00017%

rp_sum_vpsd='{:.0%}'.format(sum_vpsd)  #100%



#define the third column:singapore IQ classfication

column3=['Singapore I.Q. Estimates 2019-06-30','Severe General Learning Disabilities','Moderate General Learning Disabilities','Mild General Learning Disabilities','Below Normal','Lower Normal','Upper Normal','Bright','Gifted','Highly Gifted','Exceptionally Gifted','Profoundly Gifted']



#create the dataframe2 which is sheet"singapore2" in IQCart.xls

d={'column1':[rp_sum_vpsd,rp0,rp1,rp2,rp3,rp4,rp5,rp6,rp7,rp8,rp9,rp10],

   'column2':[tt_pop,rsd0,rsd1,rsd2,rsd3,rsd4,rsd5,rsd6,rsd7,rsd8,rsd9,rsd10]}

df2=pd.DataFrame(data=d)



d1={'column3':['Singapore I.Q. Estimates 2019-06-30','Severe General Learning Disabilities','Moderate General Learning Disabilities','Mild General Learning Disabilities','Below Normal','Lower Normal','Upper Normal','Bright','Gifted','Highly Gifted','Exceptionally Gifted','Profoundly Gifted']}

df3=pd.DataFrame(data=d1)



df4=pd.DataFrame.join(df2, df3, on=None, how='left', lsuffix='', rsuffix='', sort=False)

print(df4)



#export to Excel file

df4.to_excel('IQ normal distribution chart.xls')



#print the bar chart

data = [math.log(rsd0),math.log(rsd1),math.log(rsd2),math.log(rsd3),math.log(rsd4),math.log(rsd5),math.log(rsd6),math.log(rsd7),math.log(rsd8),math.log(rsd9),math.log(rsd10)]

labels = ['Severe General Learning Disabilities', 'Moderate General Learning Disabilities', 'Mild General Learning Disabilities', 'Below Normal', 'Lower Normal','Upper Normal','Bright','Gifted','Highly Gifted','Exceptionally Gifted','Profoundly Gifted']

plt.xlabel('General Term')

plt.ylabel('log(rounded_number)')

plt.title(r'Histogram of IQ in Singapore')

plt.xticks(rotation=90)

plt.bar(range(len(data)), data, tick_label=labels)

plt.show()










