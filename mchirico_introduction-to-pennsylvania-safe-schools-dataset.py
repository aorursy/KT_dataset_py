

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Quick listing of the files

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Reading in the data

d = pd.read_csv("../input/school.csv")
# List of counties in Pennsylvania

# d['County'].value_counts()



# Crimes by Law Enforcement 

# 'No. of Incidents Involving Local Law Enforcement' by County

t = d[(d['LEA Type'] == 'County') & (d['Year']==2016)][['Year','County','Year','No. of Incidents Involving Local Law Enforcement']]

t.sort_values(by='No. of Incidents Involving Local Law Enforcement',ascending=False)
# Note, you might want to look at the above 

# by percentage of student population

t = d[(d['LEA Type'] == 'County') & (d['Year']==2016)][['Year','County','Year','No. of Incidents Involving Local Law Enforcement','Enrollment']]

t['Percent Student Population'] = t['No. of Incidents Involving Local Law Enforcement']/t['Enrollment']



t.sort_values(by='Percent Student Population',ascending=False).head(20)
# Percentage of populate impacted only

# considering HS an JH

g=d[(d['School Name'].str.contains(' HS', na=False)) |

 d['School Name'].str.contains(' SHS', na=False)|

 d['School Name'].str.contains(' JHS', na=False)

 ].groupby(['County']).agg({'No. of Incidents Involving Local Law Enforcement':sum,

                           'Enrollment':sum})

g=g.reset_index()

g['Percent Student Population'] = g['No. of Incidents Involving Local Law Enforcement']/g['Enrollment']

g.sort_values(by='Percent Student Population',ascending=False).head(20)
# Same as above, but using 'Sexual Harassment'

# Percentage of populate impacted only

# considering HS an JH

g=d[(d['School Name'].str.contains(' HS', na=False)) |

 d['School Name'].str.contains(' SHS', na=False)|

 d['School Name'].str.contains(' JHS', na=False)

 ].groupby(['County']).agg({'Sexual Harassment':sum,

                           'Enrollment':sum})

g=g.reset_index()

g['Percent Student Population'] = g['Sexual Harassment']/g['Enrollment']

g.sort_values(by='Percent Student Population',ascending=False).head(50)
d['LEA Type'].value_counts()
# List of all schools in Montgomery, PA

# Note, you need to use 'LEA Type School'

d[(d['County']=='Montgomery') & (d['LEA Type']=='School')][['School Name']].drop_duplicates()
# Take a look at one school

d[(d['School Name']=="Cheltenham HS")][['Year','Offenders','Enrollment','No. of Incidents Involving Local Law Enforcement']]
# Calculate percentage of offenders field

def f(x):

    if x[1] > 0:

        return (x[0]/x[1]*100)



d['percentOffenders'] = d[(d['LEA Type']=='School')][['Offenders','Enrollment']].apply(f, axis=1)

d['percentSexHarass'] = d[(d['LEA Type']=='School')][['Sexual Harassment','Enrollment']].apply(f, axis=1)



# For year 2016, which school in Montgomery county had the highest percentage of offenders?

#

t = d[(d['County']=='Montgomery') & (d['LEA Type']=='School') & (d['Year']==2016)][['School Name','percentOffenders','Enrollment','Offenders','No. of Incidents Involving Local Law Enforcement']].drop_duplicates()

t.sort_values(by='percentOffenders',ascending=False)
# High School with greatest percentage of offenders

t = d[(d['School Name'].str.contains(' HS', na=False)) & (d['LEA Type']=='School') & (d['Year']==2016)][['School Name','County','percentOffenders','Enrollment','Offenders','No. of Incidents Involving Local Law Enforcement']].drop_duplicates()

t.sort_values(by='percentOffenders',ascending=False)
t.sort_values(by='percentOffenders',ascending=False).to_csv('hs2016.csv',index=True,header=True)
# High School with greatest percentage of offenders for 2015

t = d[(d['School Name'].str.contains(' HS', na=False)) & (d['LEA Type']=='School') & (d['Year']==2015)][['School Name','County','percentOffenders','Enrollment','Offenders','No. of Incidents Involving Local Law Enforcement']].drop_duplicates()

t.sort_values(by='percentOffenders',ascending=False)

t.sort_values(by='percentOffenders',ascending=False).to_csv('hs2015.csv',index=True,header=True)
# Sexual Harassment

# High School with greatest percentage of offenders for 2016

t = d[(d['School Name'].str.contains(' HS', na=False)) & (d['LEA Type']=='School') & (d['Year']==2016)][['School Name','County','percentOffenders','Enrollment','Offenders','No. of Incidents Involving Local Law Enforcement','Sexual Harassment']].drop_duplicates()

t.sort_values(by='percentOffenders',ascending=False)

t.sort_values(by='percentOffenders',ascending=False).to_csv('hsSexh2016.csv',index=True,header=True)
# Top schools dealing with sexual harrassment by percentages...

t = d[(d['School Name'].str.contains(' HS', na=False)) & (d['LEA Type']=='School') & (d['Year']==2016)][['School Name','percentOffenders','Enrollment','Offenders','No. of Incidents Involving Local Law Enforcement','percentSexHarass','Sexual Harassment']].drop_duplicates()

t.sort_values(by='percentSexHarass',ascending=False)
# Same select as above, but consider all schools...

# Top schools dealing with sexual harrassment by percentages...

t = d[(d['LEA Type']=='School') & (d['Year']==2016)][['School Name','percentOffenders','Enrollment','Offenders','No. of Incidents Involving Local Law Enforcement','percentSexHarass','Sexual Harassment']].drop_duplicates()

t.sort_values(by='percentSexHarass',ascending=False).to_csv('hsSexhAllSchools2016.csv',index=True,header=True)