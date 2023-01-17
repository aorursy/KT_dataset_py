# import packages
import re, sklearn, warnings, math, pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt, matplotlib
import plotly.graph_objs as go, plotly.offline as py, plotly.tools as tls
from matplotlib_venn import venn2
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# import settings
%matplotlib inline
matplotlib.rcParams.update({'font.size': 12})
py.init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')
# read in data
df = pd.read_csv("../input/real_estate_db.csv", encoding='ISO-8859-1',index_col='UID') 

# calculate basic data
df['pop_density'] = df['pop'].values/ df['ALand'].values# calculate population density

# calculate average median age
t_male_yrs       = df.male_age_median.values*df.male_pop.values
t_female_yrs     = df.female_age_median.values*df.female_pop.values 
df['age_median'] = (t_male_yrs + t_female_yrs)/(df.male_pop.values + df.female_pop.values)
# obtain top x = 2,000 location with the highest second mortgage rates
term = 'second_mortgage'; scale = 1000; places = []; 
limits = [(0,399),(400,799),(800,1199),(1200,1599),(1600,2000)]
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(255,133,27)","rgb(76, 199, 144)"]
df_pop = df[df['pct_own']> .1][[term,'place','lat','lng','pct_own','pop','age_median']].sort_values(by=term, ascending=False)
df_pop['text'] = df_pop.place + '<br>Second Mortgage ' + round(df_pop[term]*1e2,4).astype(str) + '%'

# create colors for plot
for i in range(len(limits)): 
    nme = '{0} - {1}'.format(limits[i][0],limits[i][1]); df_sub = df_pop[limits[i][0]:limits[i][1]]; places.append(dict(type = 'scattergeo', 
    locationmode = 'USA-states', lon = df_sub.lng, lat = df_sub.lat, text = df_sub.text,marker = dict(color = colors[i],
    size = list(map(lambda x: 15 + x*scale,df_sub[term].values)),line = dict(width=0.5, color='rgb(40,40,40)'), sizemode = 'area'), name = nme))

# set layout data
layout = dict(title = '<br>Top 2,000 Highest Second Mortgage Locations<br>', showlegend = True,geo = dict(
         scope='usa', projection=dict( type='albers usa' ),showland = True,landcolor = 'rgb(217, 217, 217)',
         subunitwidth=1,countrywidth=1, subunitcolor="rgb(255, 255, 255)", countrycolor="rgb(255, 255, 255)"))

# display plot
fig = dict( data=places, layout=layout )
py.iplot( fig, validate=False, filename='d3-mortgage',show_link=False,config={'displayModeBar':False,'showLink': False})
# list of fields 
flds = ['second_mortgage','home_equity','home_equity_second_mortgage'];
df['bad_debt']  = df[flds[0]].values + df[flds[1]].values - df[flds[2]].values
df['good_debt'] = df['debt'] - df['bad_debt'];
df['no_debt']   = np.asarray(list(map(lambda x: 1-x,df['debt'])));
# calculate mean value of dataframe:
mean_vals = df.mean(axis=0);

# Bad Debt Overview Ven Diagram:
fig, ax = plt.subplots(1,2, figsize=(12, 6)); label = ['2$^e$$^d$ mortgage','Home Equity Loan']
flds = ['second_mortgage','home_equity','home_equity_second_mortgage']; term = ['10','01','11'];

# create plots
out  = venn2(subsets=(mean_vals[flds[0]],mean_vals[flds[1]],mean_vals[flds[2]]),set_labels=(label[0],label[1]),ax=ax[0])
out1 = venn2(subsets = (mean_vals['bad_debt'], mean_vals['good_debt'],.00001), set_labels = ('Bad Debt', 'Good Debt'), ax=ax[1])

# non for loop settings
out1.get_label_by_id('10').set_text(str(round(100*mean_vals['bad_debt'],2)) + '%')
out1.get_label_by_id('01').set_text(str(round(100*mean_vals['good_debt'],2)) + '%')
out1.get_label_by_id('11').set_text(' ')

# for loop settings
for i in [0,1,2]: out.get_label_by_id(term[i]).set_text(str(round(100*mean_vals[flds[i]],2)) + '%')
for text in out.set_labels: text.set_fontsize(12); 
for text in out.subset_labels: text.set_fontsize(12)
for i in [0,1]: out1.set_labels[i].set_fontsize(12); out1.subset_labels[i].set_fontsize(12);
    
# title and plot data:
ax[1].title.set_text("Debt Overview"); 
ax[0].title.set_text("Bad Debt Overview"); 
ax[0].title.set_fontsize(18); ax[0].title.set_fontsize(18);
ax[1].title.set_fontsize(18); ax[1].title.set_fontsize(18);
plt.show()


# data we wish to analize:
flds  = ['city','second_mortgage','home_equity','bad_debt','good_debt','pop'];
fldp = ['second_mortgage','home_equity','good_debt','bad_debt']; 
fldt = ['2<sup>ed</sup> Mortgage','Home Equity','Good Debt','Bad Debt']; 

# define titles for plot:
title = {
    fldp[0]:  'Highest Median Hosuehold Income Cities <br><sub>Cities with 50+ records</sub>',
    fldp[1]:  'Highest Median Rent Cities <br><sub>Cities with 50+ records</sub>',
    fldp[2]:  'Highest Median Family Income Cities <br><sub>Cities with 50+ records</sub>',
    fldp[3]:  'Highest Median Family Income Cities <br><sub>Cities with 50+ records</sub>',
}



drop_name = {
    fldp[0]:  fldt[0],
    fldp[1]:  fldt[1],
    fldp[2]:  fldt[2],
    fldp[3]:  fldt[3],
}
 
# fldp[4]:  '#D62728', #9575D2 fldp[3]:  '#D62728',
color_drop = {
     fldp[0]:  '1F77B4',
     fldp[1]:  '#FF7F0E',
     fldp[2]:  '#2CA02C',
     fldp[3]:  '#D62728',
}

# drop down names:
drop_viz = {
    fldp[0]:  [True,  False, False, False],
    fldp[1]:  [False, True,  False, False],
    fldp[2]:  [False, False, True,  False],
    fldp[3]:  [False, False, False, True ],
}

# will be used for new plots
city_count = df.city.value_counts()
bad_cities = city_count[city_count.values < 25].index.tolist();


# group data & filter data
df_city = df[df['pct_own']> .1][flds].groupby(['city']).mean().dropna()
df_city = df_city[~df_city.index.isin(bad_cities)]



# g1 hi income
buttons = []; data = []

for wrd in fldp:
    gg = df_city.sort_values(wrd,ascending=0).index.tolist()[1:15]
    data.append(go.Box(x=df[df.city.isin(gg)]['city'],
                       y=df[df.city.isin(gg)][wrd],
                       marker = dict(color = color_drop[wrd]),
                       visible= drop_viz[wrd][0],
                       name=drop_name[wrd],
                           showlegend=True))
    
    buttons.append(dict(label = drop_name[wrd],
                        method = 'restyle',
             args = ['visible', drop_viz[wrd], 
                     'title', title[wrd]]))


updatemenus = list([dict(
    y=1.12, x= .98,buttons=buttons)])
layout = dict(title='Debt Statistics <br><sub>select field from drop down menu</sub>', 
              width=700,height=700,  yaxis=dict(tickformat='%'),
              margin=go.Margin( l=50, r=50, b=100, t=100, pad=4),
              font=dict(family='Open Sans', size=12),
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='dropdown',config={'displayModeBar':False,'showLink': False,
                      'shape':{'layer':'below','hoverinfo':'none'}})

# set up fields
pop_data  = ['hs_degree','age_median','married','married_snp']
real_data = ['second_mortgage','pct_own','bad_debt'];

# create corrilation plot
dat = df[pop_data+real_data].dropna()

# Standardize features by removing the mean and scaling to unit variance
x = StandardScaler().fit_transform(dat[pop_data+real_data].values)

# perform factor analysis
FA = FactorAnalysis(n_components = 2).fit(x)

# obtain covariance matrix:
loadings = np.matrix(FA.components_); # loading est.
diag_psi = np.matrix(np.diag(FA.noise_variance_)); # diagonal psi
cov = loadings.T * loadings + diag_psi    

# transfomed data and join to our main df:
dat['latent_1'] = FA.transform(dat[pop_data+real_data].values).T[0]
dat['latent_2'] = FA.transform(dat[pop_data+real_data].values).T[1]
df = df.join(dat[['latent_1','latent_2']])
# look up dictionary for display names
flds = {'hs_degree':'Degree','age_median':'Age','second_mortgage':'2<sup>ed</sup> mortgage',
        'home_equity':'home equity','pct_own':'% Own','debt':'Debt','bad_debt':'Bad debt',
        'rem_costs':'Costs','rem_income':'Income','good_debt':'Good Debt',
       'married':'Married','divorced':'Divorced','separated':'Separated',
       'married_snp':'Spouse not present'};




# Plot constants
C1 = 'rgba(44, 62, 80, 1)'; C2 = 'rgba(44, 62, 80, .2)'
MAX = 300; trace = []; shapes = [];


# create original shape
shapes.append({'type': 'circle','layer':'below','xref': 'x','yref': 'y',
'x0': -1,'y0': -1,'x1': 1,'y1': 1,'fillcolor': 'rgba(44, 62, 80, .35)',
'line': {'color': 'rgba(0, 0, 0,0)'}})


for i in range(MAX):
    shapes.append({'type': 'circle','layer':'below','xref': 'x','yref': 'y',
                   'x0': -i**3/MAX**3,'y0': -i**3/MAX**3,'x1': i**3/MAX**3,
                   'y1': i**3/MAX**3,'fillcolor': 'rgba(250,250,250, .1)',
                   'line': {'color': 'rgba(0, 0, 0,0)'}})

for i in range(loadings.shape[1]):
    col_name = flds[list(dat.columns.values)[i]]
    trace.append(go.Scatter(x = [0,loadings[0,i]],
                            y = [0,loadings[1,i]],
                            line={'width':3},
                            marker = dict(size = 8),
                            name =col_name))

layout = go.Layout(shapes = shapes,width=700,height=700,
         margin=go.Margin( l=50, r=50, b=100, t=100, pad=4),
         xaxis=dict(zerolinecolor=C2,gridcolor=C2,range=[-1.25,1.25],
         color=C1,title='<b>Latent Factor<sub>1</sub><b>'),
         yaxis=dict(zerolinecolor=C2,gridcolor=C2,range=[-1.25,1.25],
         color=C1,title='<b>Latent Factor<sub>2</sub><b>'),
         font=dict(family='Open Sans', size=14),
         title='<b>Factor Analysis: LF<sub>1</sub> & LF<sub>2</sub></b>')

fig = go.Figure(data=trace, layout=layout)
py.iplot(fig, filename='basic-line',
              config={'displayModeBar':False,'showLink': False,
                      'shape':{'layer':'below','hoverinfo':'none'}})

# Graphic Top Income Cities with a Population above 50 records

# data we wish to analize:
flds  = ['city','hi_median','family_median','rent_median','pop'];
fldp  = ['hi_median','rent_median','family_median']

# define titles for plot:
title = {
    'hi_median':  'Highest Median Hosuehold Income Cities <br><sub>Cities with 50+ records</sub>',
     'rent_median':'Highest Median Rent Cities <br><sub>Cities with 50+ records</sub>',
    'family_median':  'Highest Median Family Income Cities <br><sub>Cities with 50+ records</sub>',
}

# drop down names:
transferKey = {'hi_median':  'Household Income',
               'rent_median':'family_median',
               'family_median':'rent_median'}

drop_name = {'hi_median':  'Household Income',
             'rent_median':'Rent','family_median':'Family Income'}

color_drop = {'hi_median':  '1F77B4',
    'rent_median':'#FF7F0E', 'family_median':'#2CA02C'}

# drop down names:
drop_viz = {
    'hi_median':  [True, False, False],
    'rent_median':[False, True, False],
    'family_median':[False, False, True],
}

# will be used for new plots
city_count = df.city.value_counts()
bad_cities = city_count[city_count.values < 50].index.tolist();


# group data & filter data
df_city = df[flds].groupby(['city']).mean().dropna()
df_city = df_city[~df_city.index.isin(bad_cities)]



# g1 hi income
buttons = []; data = []

for wrd in fldp:
    gg = df_city.sort_values(wrd,ascending=0).index.tolist()[1:15]
    data.append(go.Box(x=df[df.city.isin(gg)]['city'],
                       y=df[df.city.isin(gg)][wrd],
                       visible= drop_viz[wrd][0],
                       marker = dict(color = color_drop[wrd]),
                       name=drop_name[wrd],
                           showlegend=True))
    
    buttons.append(dict(label = drop_name[wrd],
                        method = 'restyle',
             args = ['visible', drop_viz[wrd], 
                     'title', title[wrd]]))


updatemenus = list([dict(y=1.12, x= .98,buttons=buttons)])
layout = dict(title='Median Income & Rent Data<br><sub>select field from drop down menu</sub>', 
              width=700,height=700,  yaxis=dict(tickformat='$,d'),
              margin=go.Margin( l=50, r=50, b=100, t=100, pad=4),
              font=dict(family='Open Sans', size=12),
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='dropdown',config={'displayModeBar':False,'showLink': False,
                      'shape':{'layer':'below','hoverinfo':'none'}})


# break down the three categories into basic equations
noDebt   = df.no_debt.values*df.hc_mean.values;
goodDebt = df.good_debt.values*df.hc_mortgage_median.values;
badDebt  = df.bad_debt.values*df.hc_mortgage_median.values; # +df['hc_mortgage_stdev'].values);

# hc_feature: housing cost feature
df['hc_feature'] = goodDebt + noDebt + badDebt
# adjust from one month to one year:
home_costs =  np.asarray(list(map(lambda x: 12*x ,df.hc_feature)))
rent_costs =  np.asarray(list(map(lambda x: 12*x ,df.rent_median)))

# calculate rent and home adjusted income                              
rent_adj_inc = df.family_median.values - rent_costs; # adj family income                       
home_adj_inc = df.hi_median.values - home_costs; # adj household income                    
pct_own  = df.pct_own.values; # percent own                          
pct_rent = np.asarray(list(map(lambda x: 1-x,df.pct_own))); # percent rent   

# save remaining income and remaining costs
df['rem_income'] = (pct_own*home_adj_inc) + (pct_rent*rent_adj_inc); # remaining income 
df['rem_costs']  = (pct_own*home_costs) + (pct_rent*rent_costs);     # remaining costs 
# fields to create plots
flds = ['family_median','rem_income','type','hi_median','rent_median'];

# plot expendable_income & discounted_income
f, ax = plt.subplots(figsize=(12,8))
plt_df = df.dropna(subset=['rem_income'])[flds];
sns.distplot(plt_df.rem_income,bins=50)
sns.distplot(plt_df.hi_median,bins=50)
sns.distplot(plt_df.family_median,bins=50)
plt.legend(['Remaining Income','Family Income','Household Income'])


# plot a bubble in the senter
plt.xlabel("US Dollar", fontsize=14,color = '#34495E'); 
plt.ylabel("Probability", fontsize=14,color = '#34495E')
plt.title('Income Distribution Plots', fontsize=18,color = '#34495E')
plt.setp(ax.spines.values(), color='#34495E',alpha = .8)
ax.grid(color = '#2C3E50',alpha = .08)
ax.patch.set_alpha(0)
plt.grid(True)
plt.show()
# Let us just impute the missing values with mean values to compute correlation coefficients #
data_1 = ['rent_gt_10','rent_gt_15','rent_gt_20','rent_gt_25','rent_gt_30','rent_gt_35','rent_gt_50'];
data_2 = ['family_median','hc_mortgage_median','hc_median','rent_median','rem_income','rem_costs']
data_3 = ['pop_density','married','married_snp','latent_1','latent_2'] 


# create df_corr dataframe & drop all nans:
data = data_1 + data_2 + data_3
df_corr = df[data].dropna()
df_corr[data] = df_corr[data].apply(lambda x: (x - np.mean(x))/np.std(x))
corrmat = df_corr.corr(method='spearman')


# Draw the heatmap using seaborn
f = plt.figure(figsize=(10, 10))
sns.set(font_scale= 1.2,rc={"font.size": 2.1})
mask = np.zeros_like(corrmat); mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): ax = sns.heatmap(corrmat, mask=mask, vmax=.3, square=True)
plt.title("variables correlation Overview", fontsize=15); plt.show()
# Let us just impute the missing values with mean values to compute correlation coefficients #
data_1 = ['hs_degree','hs_degree_male','hs_degree_female','female_age_median','male_age_mean'];
data_2 = ['pop_density','married','married_snp','latent_1','latent_2']
data_3 = ['pct_own','second_mortgage','home_equity']



# create df_corr dataframe & drop all nans:
data = data_1 + data_2 + data_3
df_corr = df[data].dropna()
df_corr[data] = df_corr[data].apply(lambda x: (x - np.mean(x))/np.std(x))
corrmat = df_corr.corr(method='spearman')

# Draw the heatmap using seaborn
f = plt.figure(figsize=(10, 10))
sns.set(font_scale= 1.2,rc={"font.size": 2.1})
mask = np.zeros_like(corrmat); mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): ax = sns.heatmap(corrmat, mask=mask, vmax=.3, square=True)
plt.title("variables correlation Overview", fontsize=15); plt.show()