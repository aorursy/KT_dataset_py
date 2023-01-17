import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.formula.api as smf



%matplotlib inline 
# World Development Indicators data

wdi = pd.read_csv('../input/wdi-dec-2019/WDIData.csv')
wdi.head(3)
g20 = ['AUS', 'IND', 'ARG', 'FRA', 'CHN', 

       'CAN', 'RUS', 'BRA', 'DEU', 'IDN', 

       'SAU', 'ZAF', 'MEX', 'ITA', 'JPN', 

       'USA', 'TUR', 'GBR', 'KOR']



outputs = ['NV.AGR.TOTL.KD', 'NV.IND.TOTL.KD', 'NV.SRV.TOTL.KD']

cols = ['Agriculture', 'Industry', 'Services']



t20 = '2010'
# Extracting data 

data20 = pd.DataFrame(index=g20, columns=cols).sort_index()



for g in g20:

    for o in range(len(outputs)):

        data20.loc[g][cols[o]] = wdi[wdi['Country Code'] == g][wdi['Indicator Code'] == outputs[o]][t20].tolist()[0]
data20
# Add a blank row for easier alignment of charts 

data20.loc['blank'] = 0
# Setting up the colors, this is important 

colors20 = ['#4FD4DB', '#5EA1A2', '#C6C7E3']
# Setting up how many charts to be shown in each row and column 

nrows21 = 4

ncols21 = round(len(data20.index)/nrows21)



fig21, ax21 = plt.subplots(nrows=nrows21, ncols=ncols21, figsize = (20, 16))

fig21.patch.set_facecolor('xkcd:white')



for i in range(nrows21):

    for j in range(ncols21):

        

        ax21[i][j].pie(

            data20.iloc[i*ncols21+j], 

            colors=colors20, explode=[0, 0, 0], shadow=False, 

            startangle=-270, counterclock=False, 

            wedgeprops={'width':0.6, 'edgecolor':'white', 'linewidth':1}, 

            autopct='%1.2f%%', pctdistance=0.7, 

            labeldistance=1.2, 

            textprops={'size':16}

        )

        

        ax21[i][j].set_ylabel(data20.index.tolist()[i*ncols21+j], fontsize=18, labelpad=0.1)



plt.legend(cols, loc='lower right', fontsize=18)

plt.show()
data22 = data20.drop('blank')

# Pie charts can generate proportions automatically, but other charts cannot 

data22['sum'] = data22[cols[0]] + data22[cols[1]] + data22[cols[2]]
# Setting up the circle 

N22 = len(data22.index)



theta22 = np.linspace(start=0.0, stop=2*np.pi, num=N22, endpoint=False)



angels22 = [n/float(N22)*2*np.pi for n in range(N22)]

angels22.append(angels22[0])  # It has to be a closure 



# For this kind of graphs, tick labels are between but not on tick points 

witdh22 = (2*np.pi)/N22

angels22_2 = [n+witdh22/2 for n in angels22]
# Setting up the petals of the rose 

base_height22 = 2

layer22 = 3  # How many layers of petals 



radii22 = []  # Radius of each layer  

bottom22 = []

data22['base_bottom'] = [0.5]*N22



for i in range(layer22):

    radii22.append(base_height22*data22.iloc[:, i]/data22['sum'])

    

    if i == 0:

        pass

    else:

        data22['base_bottom'] += radii22[-2]

    

    bottom22.append(data22['base_bottom'].tolist())

    

radii22 = [r.tolist() for r in radii22]
fig22, ax22 = plt.subplots(figsize=(10, 10), subplot_kw={'projection':'polar'})

fig22.patch.set_facecolor('xkcd:white')



for r, b, c in zip(radii22, bottom22, colors20):

    ax22.bar(x=theta22, height=r, bottom=b, width=witdh22, color=c, lw=1, edgecolor='white', alpha=1)



ax22.set_xticklabels([])

ax22.set_yticklabels([])

ax22.set_yticks([])

ax22.set_xticks(angels22_2)

ax22.tick_params(axis='x', which='both', grid_linestyle='--',grid_linewidth=0.2, grid_alpha=0.5)

ax22.set_axis_off()



for n in range(N22):

    ax22.text(x=angels22[n], y=bottom22[-1][n]+radii22[-1][n]+0.2, 

              s=data22.index.tolist()[n], 

              ha='center', va='center', fontsize=14

              )



ax22.set_title('Economic Output Composition of G20 Countries in ' + t20, fontsize=20, pad=40)



plt.legend(cols, loc='lower right', bbox_to_anchor=(1.05, -0.1), fontsize=14)

plt.tight_layout()

plt.show()
data23 = data20.drop('blank')
N23 = len(data23.index)



theta23 = np.linspace(start=0.0, stop=2*np.pi, num=N23, endpoint=False)



angels23 = [n/float(N23)*2*np.pi for n in range(N23)]

angels23.append(angels23[0])  # It has to be a closure 



# For this kind of graphs, tick labels are between but not on tick points 

witdh23 = (2*np.pi)/N23

angels23_2 = [n+witdh23/2 for n in angels22]

base_height23 = 10**13

layer23 = 3  # How many layers of petals 



radii23 = []  # Radius of each layer  

bottom23 = []

data23['base_bottom'] = [0.2]*N23



for i in range(layer23):

    radii23.append(data23.iloc[:, i]/base_height23)

    

    if i == 0:

        pass

    else:

        data23['base_bottom'] += radii23[-2]

    

    bottom23.append( data23['base_bottom'].tolist())

    

radii23 = [r.tolist() for r in radii23]
fig23, ax23 = plt.subplots(figsize=(16, 16), subplot_kw={'projection':'polar'})

fig23.patch.set_facecolor('xkcd:white')



for r, b, c in zip(radii23, bottom23, colors20):

    ax23.bar(x=theta23, height=r, bottom=b, width=witdh23, color=c, lw=0.5, edgecolor='white')



ax23.set_xticklabels([])

ax23.set_yticklabels([])

ax23.set_yticks([])

ax23.set_xticks(angels23_2)

ax23.tick_params(axis='x', which='both', grid_linestyle='--',grid_linewidth=0.2, grid_alpha=0.5)

ax23.set_axis_off()



for n in range(N23):

    ax23.text(x=angels23[n], y=bottom23[-1][n]+radii23[-1][n]+0.1, 

              s=data23.index.tolist()[n], 

              ha='center', va='center', fontsize=14)



ax23.text(x=0, y=0, s='G20\nOutputs', ha='center', va='center', fontsize=14)



ax23.set_title('Economic Outputs of G20 Countries in ' + t20, fontsize=20, pad=-170)



plt.legend(cols, loc='lower left', bbox_to_anchor=(0.1, 0.1), fontsize=16)

plt.show()
t_start = 1978

t_end = 2010



t = [t for t in range(t_start, t_end+1)]

tt = [str(t) for t in t]
# China statistical data consisted of two parts with different structures

GpC = pd.read_excel('../input/china-gdp-per-capital-19782010/China GDP per Capita.xlsx', index_col='Province')

dummy = pd.read_excel('../input/china-province-series/China Prinvince Series.xlsx', index_col='Province')
province = GpC.index.tolist()
index=np.arange(len(t)*len(province)).tolist()

variables = ['year', 'province', 'G', 'SE', 'CL']
data = pd.DataFrame(index=index, columns=variables)
idx = 0

for year in t:

    for i in province:

        data['year'].iloc[idx] = year

        data['province'].iloc[idx] = i

        data['G'].iloc[idx] = GpC.loc[i][year]

        for a in variables[3:]:

            data[a].iloc[idx] = dummy.loc[i][a]

        idx += 1
data
# The target data to be collected are the estimated coefficients, the p values, and the r squared 

a = [[], [], []]

p = [[], [], []]

R2 = []



for year in t:

    modeldata = data[data['year'] == year][variables[1:]].set_index('province')

    

    model1 = smf.ols('G ~ SE + CL', data=modeldata.astype(float))

    result1 = model1.fit()

    

    for i in range(len(a)):

        a[i].append(result1.params.tolist()[i])

        p[i].append(result1.pvalues.tolist()[i])

    R2.append(result1.rsquared)



matrix = pd.DataFrame(

    {

        'Year': t, 

        'a0': a[0], 

        'p0': p[0], 

        'a1': a[1], 

        'p1': p[1], 

        'a2': a[2], 

        'p2': p[2],

        'R2': R2

    }

)
matrix
fig11, ax11 = plt.subplots(figsize=(12, 6))

fig11.patch.set_facecolor('xkcd:white')



a_NW = matrix['a0'].tolist()

a_SE = (matrix['a0']+matrix['a1']).tolist()

a_CL = (matrix['a0']+matrix['a2']).tolist()



ax11.plot(t, a_NW, label='Northwest')

ax11.plot(t, a_SE, label='Southeast')

ax11.plot(t, a_CL, label='Coastal')



ax11.legend(['Northwest', 'Southeast', 'Coastal'], fontsize=14)

ax11.grid(axis='y', ls='--')

ax11.set_ylabel('GDP per Capita (Yuan)', fontsize=14, labelpad=10)



ax11.set_title('GDP per Capita of China Provinces (1978 ~ 2010)', fontsize=20)



plt.xlim(t[0], t[-1]+1)

plt.ylim(0, 50000)

plt.xticks(np.arange(t[0], t[-1]+1, 1), rotation=90)



plt.show()
fig12, (ax121, ax122) = plt.subplots(ncols=1, nrows=2, figsize=(12, 8))

fig12.patch.set_facecolor('xkcd:white')



ax121.plot(t, matrix['p2'].tolist(), label='The p-values of b2')

ax121.xaxis.set_ticks(np.arange(t[0], t[-1]+1, 1))

ax121.yaxis.set_ticks(np.arange(0, 0.14, 0.01))

ax121.tick_params(axis='x', rotation=90)

ax121.grid(ls='--', alpha=0.3)

ax121.legend(fontsize=14)

ax121.set_ylabel('p-values', fontsize=14, labelpad=10)



ax122.plot(t, matrix['R2'].tolist(), label='R-squared', color='red')

ax122.xaxis.set_ticks(np.arange(t[0], t[-1]+1, 1))

ax122.tick_params(axis='x', rotation=90)

ax122.grid(ls='--', alpha=0.3)

ax122.legend(fontsize=14)

ax122.set_ylabel('R-squared', fontsize=14, labelpad=10)



plt.show()
ServiceG_CHN = wdi[wdi['Country Code'] == 'CHN'][wdi['Indicator Code'] == 'NV.SRV.TOTL.KD.ZG'][tt].fillna(0).iloc[0].tolist()

AgricultureG_CHN = wdi[wdi['Country Code'] == 'CHN'][wdi['Indicator Code'] == 'NV.AGR.TOTL.KD.ZG'][tt].fillna(0).iloc[0].tolist()

IndustryG_CHN = wdi[wdi['Country Code'] == 'CHN'][wdi['Indicator Code'] == 'NV.IND.TOTL.KD.ZG'][tt].fillna(0).iloc[0].tolist()
cat = ['Agriculture', 'Industry', 'Services']

colors3 = ['#4FD4DB', '#5EA1A2', '#C6C7E3']
fig31, ax31 = plt.subplots(figsize=(14, 6))

fig31.patch.set_facecolor('xkcd:white')



cmap = plt.get_cmap('plasma')

color31 = cmap(np.arange(20, 500, 90))



t03_1 = [t-0.15 for t in t]

t03_2 = [t+0.15 for t in t]

t03_3 = [t+0.3 for t in t]



t03 = [t03_1, t, t03_2]

var_03 = [AgricultureG_CHN, IndustryG_CHN, ServiceG_CHN]

# label31=['Service', 'Agriculture', 'Industry']



for i in range(len(t03)):

    ax31.bar(t03[i], var_03[i], width=0.5, 

             edgecolor=color31[i], facecolor='white', 

             align='center', 

             alpha=0.8, lw=1, 

             label=cat[i]

             )



ax31.set_xticks(t)

ax31.tick_params(axis='x', rotation=90)

ax31.set_ylabel('Growth Rate', fontsize=14, labelpad=10)

ax31.grid(axis='y', ls='--', alpha=0.5)

ax31.legend()



ax31.set_title('Economic Outputs Growth of China (1978 ~ 2010)', fontsize=20)



ax31.spines['top'].set_visible(False)

ax31.spines['right'].set_visible(False)

ax31.spines['left'].set_visible(False)



plt.show()
fig32, ax32 = plt.subplots(figsize=(14, 6))

fig32.patch.set_facecolor('xkcd:white')



height = [AgricultureG_CHN, IndustryG_CHN, ServiceG_CHN]



b0 = [0]*len(AgricultureG_CHN)

b1 = [a if a > 0 else 0 for a in AgricultureG_CHN]

b2 = [sum(b) for b in zip(b1, IndustryG_CHN)]

bottom = [b0, b1, b2]



for h, b, c, i in zip(height, bottom, colors3, cat):

    ax32.bar(x=t, height=h, bottom=b, color=c, lw=1, edgecolor='white', alpha=1, label=i)



ax32.set_xticks(t)

ax32.tick_params(axis='x', rotation=90)

ax32.set_ylabel('Growth Rate', fontsize=14, labelpad=10)

ax32.grid(axis='y', ls='--', alpha=0.5)

ax32.legend()



ax32.set_title('Economic Outputs Growth of China (1978 ~ 2010)', fontsize=20)



ax32.spines['top'].set_visible(False)

ax32.spines['right'].set_visible(False)

ax32.spines['left'].set_visible(False)



plt.show()
Agriculture_CHN = wdi[wdi['Country Code'] == 'CHN'][wdi['Indicator Code'] == 'NV.AGR.TOTL.KD'][tt].fillna(0).iloc[0].tolist()

Industry_CHN = wdi[wdi['Country Code'] == 'CHN'][wdi['Indicator Code'] == 'NV.IND.TOTL.KD'][tt].fillna(0).iloc[0].tolist()

Service_CHN = wdi[wdi['Country Code'] == 'CHN'][wdi['Indicator Code'] == 'NV.SRV.TOTL.KD'][tt].fillna(0).iloc[0].tolist()
SUM = [sum(a) for a in zip(Agriculture_CHN, Industry_CHN, Service_CHN)]
AC = [a/b for (a, b) in zip(Agriculture_CHN, SUM)]

IC = [a/b for (a, b) in zip(Industry_CHN, SUM)]

SC = [a/b for (a, b) in zip(Service_CHN, SUM)]
fig33, ax33 = plt.subplots(figsize=(12, 6))

fig33.patch.set_facecolor('xkcd:white')



colors3_1 = ['white', 'none', '#C6C7E3']

ax33.stackplot(t, AC, IC, SC, 

               baseline='zero', 

               edgecolor='white', lw=1.5, 

               colors=colors3, 

               labels=cat

               )



ax33.set_xticks(t)

ax33.tick_params(axis='x', rotation=90)

ax33.set_ylabel('Economic Component', fontsize=14, labelpad=10)

ax33.legend(loc='upper right', bbox_to_anchor=(1.12, 0.95))



ax33.set_title('Economic Component of China (1978 ~ 2010)', fontsize=20)



ax33.spines['top'].set_visible(False)

ax33.spines['right'].set_visible(False)

ax33.spines['left'].set_visible(False)



plt.show()
