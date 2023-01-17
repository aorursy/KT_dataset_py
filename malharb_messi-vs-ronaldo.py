# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
age = [19,20,21,22,23,24,25,26,27,28,29,30,31,32]
messigoals = [30,51,92,140,197,279,348,394,458,508,565,616,671,697]
ronaldogoals = [23,36,62,102,132,160,213,272,339,403,463,521,575,628]

figgoals,ax = plt.subplots(dpi=250)
ax.plot(age,messigoals,c='crimson',marker='o',markersize=5,label='Messi')
ax.plot(age,ronaldogoals,c='blue',marker='o',markersize=5,label='Ronaldo')
ax.set_ylabel('Goals')
ax.set_xlabel('Age')
ax.set_title('GOALS EVOLUTION BY AGE')


ax.legend()
messiassists = [11,27,46,58,87,120,139,155,187,214,232,254,275,292]
ronaldoassists = [30,39,56,64,73,81,99,116,131,146,168,185,198,209]

figassists,ax2 = plt.subplots(dpi=250)
ax2.plot(age,messiassists,c='crimson',marker='o',markersize=5,label='Messi')
ax2.plot(age,ronaldoassists,marker='o',markersize=5,label='Ronaldo')
ax2.set_xlabel('Age')
ax2.set_ylabel('Assists')
ax2.set_title('ASSISTS EVOLUTION BY AGE')
ax2.legend()
messicgoalsperseason = [51,45,54,41,58,41,60,73,53,47]
ronaldocgoalsperseason = [28,44,42,51,61,51,55,60,53,33]

figgps,ax3 = plt.subplots(dpi=250)
ax3.boxplot([messicgoalsperseason,ronaldocgoalsperseason])
ax3.set_title('AVG. CLUB GOALS PER SEASON SINCE 09/10')
ax3.set_ylabel('Goals')
messiaps = [19,18,16,23,27,14,15,29,23,11]
ronaldoaps = [10,8,12,15,21,14,12,15,15,7]

figaps,ax4 = plt.subplots(dpi=250)
ax4.boxplot([messiaps,ronaldoaps],labels=['Messi','Ronaldo'], showmeans = False)
ax4.set_title('AVG. CLUB ASSISTS PER SEASON SINCE 09/10')
ax4.set_ylabel('Assists')
#messivsronaldo headtohead per season
labels = ['Goals','Assists','G+A',"Ballon d'Ors",'Golden Boots','Team Trophies']
messih2h = [7,10,9,6,6,5]
ronaldoh2h = [3,0,2,4,3,4]
width = 0.35
x = np.arange(len(labels))

figh2hyearly,ax5 = plt.subplots(dpi=250)
ax5.bar(x-width/2,messih2h,width = 0.35, label = 'Messi', color = 'crimson')
ax5.bar(x+width/2,ronaldoh2h,width=0.35, label = 'Ronaldo')

ax5.set_xticks(x)
ax5.set_xticklabels(labels,fontsize=5)
ax5.set_title('AMOUNT OF TIMES PLAYER HAS ENDED A SEASON WITH MORE (SINCE 08/09):', fontsize = 7)
ax5.set_ylabel('No. of times')

ax5.legend()
labels = ['2016/17','2017/18','2018/19']
barcelonagoals = [171,141,138]
messigoals = [54,45,51]
ronaldoteamgoals = [173,148,86]
ronaldogoals = [42,44,28]
width = 0.35

goals = np.array([barcelonagoals,messigoals])

fig, ax = plt.subplots(1,2,figsize = (8,4),dpi=200)
plt.tight_layout()

ax[0].bar([0,1,2],barcelonagoals, width = 0.35, label = 'Team')
ax[0].bar([0,1,2],messigoals,width=0.35, label = 'Messi')
ax[0].set_xticks([0,1,2])
ax[0].set_xticklabels(labels)
ax[0].set_ylabel('Goals')
ax[0].set_title("MESSI'S GOALS AS A PART OF HIS TEAM'S",fontsize = 8)
ax[0].legend()

ax[1].bar([0,1,2],ronaldoteamgoals, width = 0.35, label = 'Team')
ax[1].bar([0,1,2],ronaldogoals,width=0.35,label='Ronaldo')
ax[1].set_xticks(np.arange(len(labels)))
ax[1].set_xticklabels(labels)
ax[1].set_title("RONALDO'S GOALS AS A PART OF HIS TEAM'S",fontsize = 8)
ax[1].legend()
messiteams = [33,5]
ronaldoteams = [29,11]

#plt.pie(messiteams, labels=['Finals played by team','Finals won without a Messi goal (13.2%)'],explode=[0,0.1],colors=['blue','red'],shadow = True)
fig,ax = plt.subplots(1,2,dpi=200, figsize = (14,6))
plt.tight_layout

ax[0].pie(messiteams,labels=['','Finals won without Messi scoring (15.2%)'],startangle=-90,counterclock=False,colors=['white','red'],shadow=False,wedgeprops={'edgecolor':'black','linewidth':2})
ax[0].set_title("Messi's Teams",fontsize=20)
ax[0].text(-0.35,1.1,'Finals Played',fontsize=13)
ax[0].text(1.275,2,'Since 08/09',fontsize=20)


ax[1].pie(ronaldoteams,labels=[' ','Finals won without Ronaldo scoring (37.9%)'],startangle=-90,counterclock=False,colors=['white','red'],wedgeprops={'edgecolor':'black','linewidth':2})
ax[1].set_title("Ronaldo's Teams",fontsize=20)
ax[0].text(2.9,1.1,'Finals Played',fontsize=13)
#datasources:
#messivsronaldo.net
#transfermarkt.com
#whoscored.com
#sofascore.com
#https://michelacosta.com/en/messi-vs-cristiano-ronaldo-finals/
#wikipedia.org
test = input('Enter: ')
