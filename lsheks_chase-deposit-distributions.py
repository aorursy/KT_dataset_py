# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for graphing



import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/database.csv', parse_dates=[4])

df.columns

df=df.drop(df.index[df['Main Office']==1]) # We don't want to include the total coming from the main branch in our analysis
bins=np.logspace(3,9,100) #make these log bins to get a clearer picture

bins[0]=0

n,bins,patches=plt.hist(df['2016 Deposits'], bins=bins, histtype='bar')

plt.clf()

plt.loglog(bins[1:],n)

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("$", fontsize=20)

plt.ylabel("Frequency", fontsize=20)

plt.title("Distribution of Deposits at Branches", fontsize=24)

total_0_1e4=0

total_1e4_1e5=0

total_1e5_1e6=0

total_1e6_1e7=0

total_1e7_plus=0

balances=df['2016 Deposits']

for b in balances:

    if b<1e4:

        total_0_1e4+=b

    elif b<1e5:

        total_1e4_1e5+=b

    elif b<1e6:

        total_1e5_1e6+=b

    elif b<1e7:

        total_1e6_1e7+=b    

    else:

        total_1e7_plus+=b   

totals=[total_1e4_1e5,total_1e5_1e6,total_1e6_1e7,total_1e7_plus]

labels=["\$10k to \$100k", "\$100k to \$1mil", "\$1mil to \$10mil", ">\$10mil"]

print(totals)

plt.pie(totals, labels=labels,autopct='%1.1f%%', shadow=True, startangle=90, radius=1.)

plt.title("Fraction of total deposits (total=\$%s)" % '{:,d}'.format(sum(totals)), fontsize=18 )
dfpost2010=df[df['Established Date'] > datetime.datetime(2010,1,1,0,0)]

dfpre2010=df[df['Established Date'] <= datetime.datetime(2010,1,1,0,0)]



bins=np.logspace(3,9,100) #make these log bins to get a clearer picture

bins[0]=0

n,bins,patches=plt.hist(dfpost2010['2016 Deposits'], bins=bins, histtype='bar')

n1,bins1,patches1=plt.hist(dfpre2010['2016 Deposits'], bins=bins, histtype='bar')



plt.clf()

plt.loglog(bins[1:],n, label="post-2010")

plt.loglog(bins1[1:],n1, label="pre-2010")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("$", fontsize=20)

plt.ylabel("Frequency", fontsize=20)

plt.legend(fontsize=16)

plt.title(" Deposits at Branches Opened Pre vs. Post-2010", fontsize=24)

print("Median Post-2010=", np.median(dfpost2010['2016 Deposits']))

print("Median Pre-2010=", np.median(dfpre2010['2016 Deposits']))