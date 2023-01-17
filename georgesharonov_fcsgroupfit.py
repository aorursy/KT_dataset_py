data_folder="/kaggle/input/kv13-straight-titration"

conditions = [['AgTx_1nM-AgTx2-GFP','1nM AgTx2-GFP + AgTx'],
              ['ChTx_3.2nM-AgTx2-GFP','3.2nM AgTx2-GFP + ChTx'],
              ['AgTx2-GFP_100nM-cold-AgTx2','AgTx2-GFP + 100nM AgTx2'],
              ['AgTx2-GFP','AgTx2-GFP'],
              ['ChTx-GFP_316nM-cold-ChTx','ChTx-GFP + 316nM ChTx'],
              ['ChTx_3.2nM-ChTx-GFP','3.2nM ChTx-GFP + ChTx'],
              ['ChTx-GFP','ChTx-GFP']]
concentrations = [['100nM',100], ['10nM',10],['0.1nM',0.1],['0.01nM',0.01],['1nM',1],['0nM',0],
                  ['0.003nM',0.003],['0.03nM',0.03],['0.32nM',0.32],['3.2nM',3.2],['32nM',32]]

x_channel = "FL3"
y_channel = "FL1"
!pip install fcsparser #install required librarry 
#(Internet connection should be "ON", see https://www.kaggle.com/questions-and-answers/36982)
import fcsparser
import pandas as pd
import os

data=pd.DataFrame()

for filename in os.listdir(data_folder):
    meta, data_entry = fcsparser.parse(os.path.join(data_folder, filename))
    for pair in conditions:
        if pair[0] in filename:
            data_entry['Condition']=pair[1]
            filename = filename.replace(pair[0],"") 
            break
    #print(filename)
    for pair in concentrations:
        if pair[0] in filename:
            data_entry['Concentration']=pair[1]
            break 
    data = data.append(data_entry)
data=data.reset_index()
print('Available parameters: \n',data.columns.values)
print("\n Event number for each condition/concentration:")
data.sort_values(['Condition','Concentration'])
data.groupby(['Condition','Concentration']).size()
from numpy import log10
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats 

#USER INPUT 
#A distribution that will be used to reduce data along y_channel
#available distributions can be found at https://docs.scipy.org/doc/scipy/reference/stats.html
distr_name = 'norm' 

#number of segments that will be used to segment data along x_channel
segment_numb = 8

#FITTING 
#this is needed once your data is in linear scale
data['x_channel_log']=data[x_channel].apply(log10)
data['y_channel_log']=data[y_channel].apply(log10)

#required for data reduction according to statistc distribution
distr=getattr(stats, distr_name)

#new pandas dataframe for writing of reduced data in 
reduced_data=pd.DataFrame(columns=['Condition','Concentration','x_channel_log_r','y_channel_log_r'])

#define function that will be used to fit the data
def func(x, slope,intercept):
    return slope*x+intercept

#def func(x,Bottom=2,Top=4,LogEC50=4.5,HillSlope=1):
#    return Bottom+((Top-Bottom)/(1+10**((LogEC50-x)*HillSlope)))


import pandas as pd
from scipy.optimize import curve_fit

results=pd.DataFrame(columns=['Condition','Concentration','popt_a','popt_r'])

available_cond = data['Condition'].unique().tolist()
available_conc = data['Concentration'].unique().tolist()
available_conc.sort() #necessary for correct mapping of curves to the scatterplots (in FacetGrid) on the next step 

count = 0
for cond in available_cond:
    for conc in available_conc: 
        datafile = data.loc[(data['Condition']==cond)&(data['Concentration']==conc)]
        if not datafile.empty:
            
            #divide the data into segments and reduce each segment into a single point with y_channel_log_c value
            x_min = datafile['x_channel_log'].min()
            x_max = datafile['x_channel_log'].max()
            step = (x_max - x_min)/segment_numb
            for i in range(0,segment_numb):
                data_segment = datafile[(datafile['x_channel_log']>(x_min+i*step))&
                                   (datafile['x_channel_log']<(x_min+(i+1)*step))]
                if data_segment.shape[0]>50:
                    stat=distr.fit(data_segment['y_channel_log'])
                    midpoint = x_min+(i+0.5)*step #midlle of the current segment
                    reduced_data.loc[segment_numb*count+i]=[cond,conc,midpoint,stat[-2]]
                            
            #fitting of reduced data
            popt_r, pconv_r = curve_fit(func,
                           reduced_data['x_channel_log_r'][(reduced_data['Condition']==cond)&
                                                         (reduced_data['Concentration']==conc)],
                           reduced_data['y_channel_log_r'][(reduced_data['Condition']==cond)&
                                                         (reduced_data['Concentration']==conc)],maxfev=10000) 
            
            #fitting of all data 
            init_vals = [0,0] # [popt_r[0], popt_r[1]] #initial values can taken from fitting of reduced data
            val_bounds = ([0,-2],[5,10]) #min and max bounds for [slope,intercept] 
            #datafile = datafile.sample(n=1000) #to downsample the data
            popt_a, pconv_a = curve_fit(func,
                            datafile['x_channel_log'],
                            datafile['y_channel_log'], maxfev=100000, p0=init_vals, bounds=val_bounds, method='dogbox') 
                            #use ..datafile[x_channel], datafile[y_channel].. in order to fit data in linear scale
            results.loc[count]=pd.Series({'Condition':cond,'Concentration':conc,
                                      'popt_a':popt_a, 'popt_r':popt_r})
            count = count+1 
            #print(popt_a)
            
results.sort_values(['Condition','Concentration'])
%matplotlib inline
import seaborn as sns
import pylab as plt
import numpy as np

#USER INPUT

#choose type of fitted data that will be used for binding curves: 
#'a' for all-point fittind data and 
#'r' for fitting of reduced data
fit_data='r'

columns ="Concentration"
rows ="Condition" 

#!Specify the order of rows
order=['AgTx2-GFP','AgTx2-GFP + 100nM AgTx2','ChTx-GFP','ChTx-GFP + 316nM ChTx']
#order=['1nM AgTx2-GFP + AgTx','3.2nM ChTx-GFP + ChTx','3.2nM AgTx2-GFP + ChTx']

#ARRAY OF SCATTER PLOTS (INITIAL DATA) 
g2 = sns.FacetGrid(data, col=columns, row=rows, sharex=True, sharey=True, height=4, aspect=.75, 
                   margin_titles=False,
                   row_order=order) #Has to be removed if there is no correct order specified
g2.map(plt.scatter, x_channel, y_channel, s=1, alpha=0.1)
g2.set(yscale='log', xscale='log', xlim=(3000,300000), ylim=(20,200000))
g2.fig.subplots_adjust(wspace=.2, hspace=.2)


popt_ar =''.join(['popt_',fit_data])

for i, axes_row in enumerate(g2.axes):
    ncols, max_ax = max(enumerate(axes_row))
    
    for j, axes_col in enumerate(axes_row):
        row, col = axes_col.get_title().split('|')
        col = col.replace(''.join([columns,' = ']),'').strip()
        row = row.replace(''.join([rows,' = ']),'').strip()
        col_1=float(col) if col.replace('.','',1).isnumeric() else col
        row_1=float(row) if row.replace('.','',1).isnumeric() else row
        #print(col_1,row_1)
        
        ax=g2.axes[i, j]
        
        #Making titles
        if j==ncols:
            ax.text(1.1, 0.5, row_1,
                    horizontalalignment='center', verticalalignment='center',
                    rotation=-90, transform=ax.transAxes, fontsize=20)
        if i==0:
            ax.set_title('{}'.format(col_1), loc='center', fontsize=20)
        else:
            ax.set_title('') 
        
        #Check for results availability
        try:
            ppt = np.array(results.loc[(results[rows]==row_1)&
                                (results[columns]==col_1)][popt_ar].values)[0] 
            #print("ppt success")
        except: 
            #print("ppt exception")
            ax.set_title('')
            #ax.set_visible(False)
            continue
        
        #Plotting data and results of analyses
        try:
            x = np.arange(data[x_channel].min(), data[x_channel].max(), 100)
            curve = pow(10,func(log10(x),*ppt))  #func(x,*ppt) if fitting of linear data was used 
            ax.plot(x,curve,'k--', linewidth=1.0, label=''.join(['Slope = ', str(np.round(ppt[0],2))]))
            ax.legend(loc=3,prop={'size':16})
            ax.grid(b=True, which='major', linestyle='-')
            ax.grid(b=True, which='minor', linestyle='dotted', alpha=0.5)
            if fit_data == 'r':
                ax.scatter(pow(10,reduced_data.loc[(reduced_data[rows]==row_1)&
                                (reduced_data[columns]==col_1)]['x_channel_log_r']),
                           pow(10,reduced_data.loc[(reduced_data[rows]==row_1)&
                                (reduced_data[columns]==col_1)]['y_channel_log_r']),
                           s=15, color="red")         
            #print("ax success")
        except:
            print("ax exception")
            continue
            
            
g2.set_xlabels('')
g2.fig.text(0.5, -0.02, 'Receptor expression', ha='center', size=24)
g2.set_ylabels('')
g2.fig.text(0.0, 0.5, 'Ligand binding', rotation=90, va='center', size=24)
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#USER INPUT

#Choose type of fitted data that will be used for binding curves: 
#'a' for all-point fitted data and 
#'r' for data fitted after reduction
fit_data = 'r' 

binding_datasets=['AgTx2-GFP','ChTx-GFP'] #specific binding datasets
nonspecific_binding_datasets=['AgTx2-GFP + 100nM AgTx2','ChTx-GFP + 316nM ChTx'] #non-specific binding datasets

#SPECIFIC AND NSPECIFIC BINDING FUNCTIONS
def specific_binding_curve(x, Kd, Bmax): 
    return Bmax*x/(x+Kd)
def nonspecific_binding_curve(x, offset, NS=0):
    return offset+NS*x

#DATA CLEANUP
bad_data_index=[17,8] #remove some bad datapoint. Can be done for several points
                      #by repetitive execution of this step with different values.
                      #For recovery of initial data run STEP 3 again 
results=results.drop(results.index[bad_data_index]) 
        
#GRAPH INITIALIZATION
fig, axarr = plt.subplots(nrows=1,ncols=len(binding_datasets), sharex=False, sharey=False, 
                          figsize=(len(binding_datasets)*16,12))
fig.text(0.5, 0.0, 'Concentration, nM', ha='center', size=40) 

#CALCULATIONS
popt_fit_data=''.join(['popt_',fit_data])
for idx, cond in enumerate(binding_datasets):
    #extract binding curves
    try: 
        #calculate linear curve for nonspecific binding popt_ns[0]=background, popt_ns[1]=NS
        ns_x_data=results.loc[results.Condition==nonspecific_binding_datasets[idx]]['Concentration'].values
        ns_y_data=results.loc[results.Condition==nonspecific_binding_datasets[idx]][popt_fit_data].str[0].values
        popt_ns,pconv_ns = curve_fit(lambda x, offset: nonspecific_binding_curve(x, offset, 0), 
                    #force backround slope=0
                    #for inclined background line simply use 'nonspecific_binding_curve' instead of 'lambda x...'
                            ns_x_data,
                            ns_y_data, maxfev=10000)
    except:
        #use value at concentration=0 if nonspecific binding data are not available
        popt_ns = [float(results.loc[results.Condition==cond][popt_fit_data].str[0]),0]
    
    #add a column 'offset' to 'results' table with the offset/background value for each concentration 
    results.loc[results.Condition==cond,'offset'] = \
         nonspecific_binding_curve(results.loc[results.Condition==cond]['Concentration'],*popt_ns) 
 
    #add a column to 'results' table with specific binding data 'SB' by subtracting 
    #offset/background from total binding 'Slope'
    results.loc[results.Condition==cond,'SB'] = results.loc[results.Condition==cond,
                        popt_fit_data].str[0]-results.loc[results.Condition==cond,'offset']

    
    #fit specific binding curve 
    x_data = results.loc[results.Condition==cond,'Concentration'].values
    y_data = results.loc[results.Condition==cond,'SB'].values
    popt_s,pconv_s = curve_fit(specific_binding_curve,
                            x_data,
                            y_data, maxfev=10000)
   
    #caclulate R-squared
    residuals = y_data - specific_binding_curve(x_data, *popt_s)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data-np.mean(y_data))**2)
    R_squared = 1 - (ss_res / ss_tot)
    
    
    #PLOTTING & OUTPUT 
    x = pow(10,np.arange(-3, 2.1, 0.1))
    curve_ns=nonspecific_binding_curve(x,*popt_ns)
    curve_s=specific_binding_curve(x,*popt_s)
    curve_t=curve_ns+curve_s
    Kd=''.join([str(np.round(popt_s[0],3)),"+/-",str(np.round(pconv_s[0,0]**0.5,3))])
    if len(binding_datasets) > 1:
        ax = axarr[idx]
    else:
        ax = axarr   
    
    ax.scatter(x_data, results.loc[results.Condition==cond][popt_fit_data].str[0], 
                    s=150, color='b', edgecolors='none', label='')
    try:
        ax.scatter(ns_x_data,ns_y_data, 
                    s=150, color='g', edgecolors='none', label='')
    except:
        continue
    
    ax.plot(x,curve_t,'b-',label=''.join(['total, $R^2$=', str(np.round(R_squared,3))]),linewidth=4)
    ax.plot(x,curve_ns,'g-',label='offset + background',linewidth=4)
    ax.plot(x,curve_s,'r-',label='specific',linewidth=4)#.join(["specific, \nKd=", Kd]),linewidth=4) 
    ax.set_xscale('log')
    ax.set_xlim([0.001,110])
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_title(binding_datasets[idx], size=40)
    ax.plot([popt_s[0],popt_s[0]],[specific_binding_curve(popt_s[0], *popt_s),0],'r--',
            label=''.join(["Kd=", Kd]),linewidth=4)
    ax.legend(loc=2,prop={'size':28})
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(b=True, which='major', linestyle='-', linewidth=2)
    ax.grid(b=True, which='minor', linestyle='dotted', linewidth=1)
    print(binding_datasets[idx],"Kd =",Kd) 
    results.sort_values(['Condition','Concentration'])
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

#USER INPUT
#experimental parameters: [dataset, labeleg ligand concentration, labeled ligand Kd]  
params = [['1nM AgTx2-GFP + AgTx', 1, 0.195],
          ['3.2nM ChTx-GFP + ChTx', 3.2, 0.855],
          ['3.2nM AgTx2-GFP + ChTx',3.2, 0.195]]

fit_data = 'r'

#COMPETITIVE BINDING FUNCTION
def comp_binding_curve(x, Top=1, Bottom=0, IC50=0.1): 
    return Bottom + (Top-Bottom)/(1+pow(10,log10(x)-log10(IC50))) 
   
#GRAPH INITIALIZATION
fig, axarr = plt.subplots(nrows=1,ncols=len(params), sharex=False, sharey=False, figsize=(len(params)*8,12))
fig.text(0.5, 0.0, 'Concentration, nM', ha='center', size=30) 
x = pow(10,np.arange(-3, 3, 0.1))

#CALCULATIONS
popt_fit_data=''.join(['popt_',fit_data])
for idx, param in enumerate(params):
    #indexes = results.loc[lambda df: df.Condition==param[0],:].index.values.tolist()
    res=results.loc[results.Condition==param[0]]
    x_data=res['Concentration'].values.tolist()
    y_data=res[popt_fit_data].str[0].values.tolist()
    popt,pconv = curve_fit(comp_binding_curve,
                            x_data,
                            y_data, maxfev=10000)
    
    #PLOTTING & OUTPUT
    Kap=popt[2]/(1+param[1]/param[2])
    curve=comp_binding_curve(x,*popt)
    
    if len(params) > 1:
        ax = axarr[idx]
    else:
        ax = axarr
     
    ax.scatter(x_data,y_data,s=150, color='b', edgecolors='none', label='')
    ax.plot(x,curve,'r-',label='fitting',linewidth=4)
    ax.set_xscale('log')
    ax.set_ylim(bottom=popt[1])
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='x', which='minor', labelsize=0)
    ax.set_title(param[0], size=28)
    ax.plot([popt[2],popt[2]],[comp_binding_curve(popt[2], *popt),popt[1]],'k--', 
            label=''.join(['IC50=',str(np.round(popt[2],2))]),linewidth=4)
    ax.plot([0,0],[0,0],linewidth=0,label=''.join(['Kap=',str(np.round(Kap,2))]))
    ax.legend(loc=1,prop={'size':22})
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    locmin = ticker.LogLocator(base=10, subs=(0.2,0.4,0.6,0.8,1), numticks=10) 
    ax.xaxis.set_minor_locator(locmin)     
    ax.grid(b=True, which='major', linestyle='-', linewidth=2)
    ax.grid(b=True, which='minor', linestyle='dotted', linewidth=1)
    print(param[0],': IC50 =',np.round(popt[2],2),'   Kap =', np.round(Kap,4))