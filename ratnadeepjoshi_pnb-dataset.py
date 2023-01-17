# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pnb = pd.read_csv('../input/PNB.csv')
pnb.head()
pnb.info()
print('Number of Records:',str(len(pnb)))
print('Total outstanding amount in  lac Rs (1 lac is 0.1 million)', round(pnb.osamt.sum(),2))
print('Number of states',len(pnb.state.unique()))
pnb.osamt.describe()
g = sns.countplot(pnb.state)
g.set_xticklabels(g.get_xticklabels(),rotation=90);
f = plt.gcf();
f.set_size_inches(10,8)
pnb2 = pnb.groupby('state')
pnb.groupby(['state','bkbr']).osamt.sum().reset_index().sort_values(by='osamt',ascending=False)
pnb_summary = pnb2.state.count().to_frame().join(pnb2.osamt.sum().to_frame())
pnb_summary.columns = ['CasesCount','TotalAmt']
pnb_summary.head()
pnb_summary['logTotalAmt']=np.log10(pnb_summary.TotalAmt)
pnb_summary.describe()
pnb3 = pnb.groupby(['state','bkbr'])
pnb_branchwise = pnb3.bkbr.count().to_frame().join(pnb3.osamt.sum().to_frame())
pnb_branchwise.columns =['Branches','TotalAmt']
pnb_branchwise['AmtPerBranch']= round(pnb_branchwise.TotalAmt/pnb_branchwise.Branches,2)
pnb_branchwise.sort_values(by='AmtPerBranch', ascending=False).iloc[:5,:]['TotalAmt']
pnb_branchwise.iloc[pnb_branchwise.AmtPerBranch.values.argmax(),:]
pnb[(pnb['state'] =='MAHARASHTRA') & (pnb['bkbr'].str.contains('Brady'))]
pnb_branchwise.sort_values(by='AmtPerBranch', ascending=False)
def AmountInNBranches(n):
    Total_top_N = pnb_branchwise.sort_values(by='AmtPerBranch', ascending=False).iloc[:n,:]['TotalAmt'].sum()
    GrandTotal = pnb_branchwise.sort_values(by='AmtPerBranch', ascending=False)['TotalAmt'].sum()
    return Total_top_N/GrandTotal
for i in range(1,10):
    print(i,round(AmountInNBranches(i),4))
pnb2.describe()
g = sns.barplot(x=pnb_summary.index,y='TotalAmt',data=pnb_summary)
g.set_xticklabels(g.get_xticklabels(),rotation=90);
f = plt.gcf();
f.set_size_inches(10,8)
sns.boxplot(pnb['osamt'],orient='v');
g =sns.barplot(x='state',y='logTotalAmt',data=pnb_summary.reset_index().sort_values(by='logTotalAmt'))
f = plt.gcf();
g.set_xticklabels(g.get_xticklabels(),rotation=90);
f.set_size_inches(10,8)
sns.kdeplot(np.log10(pnb['osamt']))
