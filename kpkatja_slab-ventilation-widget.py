import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import numpy as np
import ipywidgets as wg
from IPython.display import display
%matplotlib inline
import matplotlib.pyplot as pl_me

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

pd.set_option('display.max_colwidth', 300)

slab_all1 = pd.read_csv('../input/airExchange_comparison80.csv', sep=",", header=None, error_bad_lines=False)
slab_all2 = pd.read_csv('../input/airExchange_comparison100.csv', sep=",", header=None, error_bad_lines=False)
slab_all3 = pd.read_csv('../input/airExchange_comparison120.csv', sep=",", header=None, error_bad_lines=False)
slab_all4 = pd.read_csv('../input/airExchange_comparison140.csv', sep=",", header=None, error_bad_lines=False)

def setdate(slab):
    slab.iloc[0,0]='Date'
    slab.columns = slab.iloc[0]
    slab = slab.iloc[1:]
    slab['Date']=pd.to_datetime(slab['Date'])

setdate(slab_all1)
setdate(slab_all2)
setdate(slab_all3)
setdate(slab_all4)
slab_all1=slab_all1.iloc[1:]
slab_all2=slab_all2.iloc[1:]
slab_all3=slab_all3.iloc[1:]
slab_all4=slab_all4.iloc[1:]

slab_all=slab_all1

#Slider values 
a_min = 1    # the minimial value of the paramater a
a_max = 10   # the maximal value of the paramater a
a_init = 1 
b_min = 80    # the minimial value of the paramater b
b_max = 140   # the maximal value of the paramater b
b_init = 80




#Handling the mold condition presentation
dates = np.array(slab_all['Date'], dtype='datetime64[D]')
dates = list(map(pd.to_datetime, dates))

#Handling the water content presentation
mc_val=slab_all.iloc[[1,60,120,175],[8*0+3,8*0+4,8*0+5,8*0+6]].rename(columns={'MC10':125,'MC20':350,'MC30':650,'MC40':875}).transpose().rename(columns={2:'02020',61:'02025',121:'02030',176:'02035'})
for x in range(1, 10):
    mc_val1=slab_all.iloc[[1,60,120,175],[8*x+3,8*x+4,8*x+5,8*x+6]].rename(columns={'MC1'+str(x):125,'MC2'+str(x):350,'MC3'+str(x):650,'MC4'+str(x):875}).transpose().rename(columns={2:str(x)+'2020',61:str(x)+'2025',121:str(x)+'2030',176:str(x)+'2035'})
    mc_val=pd.merge(mc_val,mc_val1,how='outer', left_index=True, right_index=True)
    
thickness = np.array([125,350,650,875],dtype=float)

#Interactive plotting
def update(a, b):    
    #Postprocessing the initial data
    if b==80:
        slab_all=slab_all1
        mc_str='80'
    else:
        if b==100:
            slab_all=slab_all2
            mc_str='100'
        else:
            if b==120:
                slab_all=slab_all3
                mc_str='120'
            else:
                if b==140:
                    slab_all=slab_all4
                    mc_str='140'
    
    #Handling the water content presentation
    mc_val=slab_all.iloc[[1,60,120,175],[8*0+3,8*0+4,8*0+5,8*0+6]].rename(columns={'MC10':125,'MC20':350,'MC30':650,'MC40':875}).transpose().rename(columns={2:'02020',61:'02025',121:'02030',176:'02035'})
    for x in range(1, 10):
        mc_val1=slab_all.iloc[[1,60,120,175],[8*x+3,8*x+4,8*x+5,8*x+6]].rename(columns={'MC1'+str(x):125,'MC2'+str(x):350,'MC3'+str(x):650,'MC4'+str(x):875}).transpose().rename(columns={2:str(x)+'2020',61:str(x)+'2025',121:str(x)+'2030',176:str(x)+'2035'})
        mc_val=pd.merge(mc_val,mc_val1,how='outer', left_index=True, right_index=True)
    
    #mold
    f, mold = pl_me.subplots(2, sharex=True,figsize=(10,4))
    f.suptitle('Mold condition in reference layers', size=16)
    mold[0].set_title('Flooring atop surface slab')
    mold[1].set_title('EPS layer above the massive concrete slab')
    mold[0].set_ylabel('Hours per month', fontsize=12)
    mold[1].set_ylabel('Hours per month', fontsize=12)
    mold[0].set_ylim([0, 800])
    mold[1].set_ylim([0, 800])
    str1='MoldEPS'+str(a-1)[:1]
    mold_EPS0=np.array(slab_all[str1], dtype=float)
    mold[1].plot(dates, mold_EPS0,label="Ventilation rate " +str(a)[:1])
    str2='MoldFl'+str(a-1)[:1]
    mold_fl0=np.array(slab_all[str2], dtype=float)
    mold[0].plot(dates, mold_fl0,label="Ventilation rate " +str(a)[:1]+", surface slab MC=" +mc_str+"kg/m^3")
    mold[0].legend(loc="upper right", fontsize=12)
    #water content
    str20=str(a-1)[:1]+'2020'
    str25=str(a-1)[:1]+'2025'
    str30=str(a-1)[:1]+'2030'
    str35=str(a-1)[:1]+'2035'
    air20=np.array(mc_val[str20],dtype=float)
    air25=np.array(mc_val[str25],dtype=float)
    air30=np.array(mc_val[str30],dtype=float)
    air35=np.array(mc_val[str35],dtype=float)
    f, mwc = pl_me.subplots(1, sharex=True,figsize=(10,2))
    f.suptitle('Water content in massive slab', size=16)
    mwc.set_ylabel('Slab thickness, mm')
    mwc.set_xlabel('MC, kg/m^3')
    mwc.set_xlim([80,160])
    mwc.plot(air20, thickness,label="2020 ")
    mwc.plot(air25, thickness,label="2025 ")
    mwc.plot(air30, thickness,label="2030 ")
    mwc.plot(air35, thickness,label="2035 ")
    mwc.legend(loc="upper right", fontsize=12)
    
style = {'description_width': 'initial'}



a_slider = wg.FloatSlider(value=a_init,min=a_min,max=a_max, step=1, description="Ventilation rate", width='300px', height='75px')
b_slider = wg.FloatSlider(value=b_init,min=b_min,max=b_max, step=20, description="Initial MC of surface slab", width='300px', height='75px', style=style)
w = interactive(update, a=a_slider, b=b_slider)
display(w)

