# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_terrorism = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', encoding = "ISO-8859-1")
df_terrorism = df_terrorism[['eventid','iyear','imonth','iday','extended','country','country_txt','region', 'region_txt','success', 'suicide', 'attacktype1', 'attacktype1_txt', 'targtype1', 'targtype1_txt', 'gname', 'weaptype1', 'weaptype1_txt', 'crit1','latitude' ,'longitude']]
df_terrorism.head()
!pip install scikit-posthocs
!pip3 install factor_analyzer
df_freedom_index = pd.read_csv('/kaggle/input/the-human-freedom-index/hfi_cc_2019.csv')

df_freedom_index = df_freedom_index[['year','ISO_code','countries','region','hf_score','hf_rank','pf_ss_women','pf_ss', 'pf_movement_foreign', 'pf_religion', 'pf_association_political_establish','pf_association_political', 'pf_expression_killed', 'pf_expression_jailed','pf_expression_influence','pf_expression_control','pf_expression_newspapers','pf_expression_internet','pf_score','pf_rank','ef_government','ef_legal_judicial','ef_money_growth','ef_money_inflation','ef_money_currency', 'ef_money']]
df_freedom_index = df_freedom_index[df_freedom_index['hf_score']!='-']
df_freedom_index.head()
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
import plotly.express as px
df_freedom_index['ISO_code'] = df_freedom_index['ISO_code'].astype(str)
df_freedom_index['hf_score'] = df_freedom_index['hf_score'].astype(float)
fig = px.scatter_geo(df_freedom_index, 
                     locations='ISO_code',
                     color='region',
                     size="hf_score", hover_name='countries', 
                     projection="natural earth", 
                     animation_frame="year",
                     title='Human Freedom index over years')
iplot(fig)
from tqdm import tqdm
def count_terrorism(df_terrorism):
    counts_ = []
    for i in df_terrorism.groupby(['iyear', 'country_txt'])['eventid'].count():
        for j in range(i):
            counts_.append(i)
    return counts_

df_terrorism = df_terrorism.sort_values(['iyear', 'country_txt'], ascending=[True, True]) 
df_terrorism['attacks'] = count_terrorism(df_terrorism)
df_terrorism.head()

def get_locations(df_terror, df_freedom):
    dict_ = dict(zip(df_freedom['countries'].unique(), df_freedom['ISO_code'].unique()))
    codes = []
    for country in df_terror['country_txt']:
        if country == 'Dominican Republic':
            codes.append(dict_['Dominican Rep.'])
        elif country =='East Germany (GDR)':
            codes.append(dict_['Germany'])
        elif country =='West Germany (FRG)':
            codes.append(dict_['Germany'])
        elif country == 'South Yemen' or country =='Yemen':
            codes.append(dict_['Yemen, Rep.'])
        elif country =='Brunei':
            codes.append(dict_['Brunei Darussalam'])
        elif country == 'Czechoslovakia'or country == 'Czech Republic':
            codes.append(dict_['Czech Rep.'])
        elif country == "People's Republic of the Congo" or country =='Republic of the Congo' or country=='Democratic Republic of the Congo':
            codes.append(dict_['Congo, Rep. Of'])
        elif country == 'South Vietnam':
            codes.append(dict_['Vietnam'])
        elif country =='West Bank and Gaza Strip':
             codes.append(dict_['Israel'])
        elif country =='Zaire':
            codes.append(dict_['Congo, Rep. Of'])
        elif country == 'Afghanistan':
            codes.append('-')
        elif country =='Yugoslavia':
            codes.append('-')
        elif country == 'Andorra':
            codes.append('-')
        elif country == 'North Yemen':
            codes.append(dict_['Yemen, Rep.'])
        elif country == 'South Korea':
            codes.append(dict_['Korea, South'])
        elif country == 'Somalia':
            codes.append('-')
        elif country == 'Djibouti':
            codes.append('-')
        elif country == 'Rhodesia':
            codes.append('-')
        elif country == 'Soviet Union':
            codes.append('-')
        elif country == 'Western Sahara':
            codes.append('-')
        elif country == 'Grenada':
            codes.append('-')
        elif country == 'Guadeloupe':
            codes.append('-')
        elif country == 'New Hebrides':
            codes.append('-')
        elif country == 'Central African Republic':
            codes.append(dict_['Central Afr. Rep.'])
        elif country =='Dominica':
            codes.append('-')
        elif country =='Martinique':
            codes.append('-')
        elif country == 'Vatican City':
            codes.append('-')
        elif country == 'Swaziland':
            codes.append('-')
        elif country == 'Falkland Islands':
            codes.append('-')
        elif country == 'French Guiana':
            codes.append('-')
        elif country == 'New Caledonia':
            codes.append('-')
        elif country == 'Maldives':
            codes.append('-')
        elif country == 'Papua New Guinea' or country =='Turkmenistan':
            codes.append('-')
        elif country == 'Cuba':
            codes.append('-')
        elif country == 'Antigua and Barbuda' or country == 'Solomon Islands':
            codes.append('-')
        elif country == 'Bosnia-Herzegovina':
            codes.append(dict_['Bosnia and Herzegovina'])
        elif country == 'Comoros':
            codes.append('-')
        elif country == 'Equatorial Guinea' or country =='Kyrgyzstan':
            codes.append('-')
        elif country == 'Ivory Coast':
            codes.append(dict_['C?te d\'Ivoire'])
        elif country == 'Uzbekistan':
            codes.append('-')
        elif country =='Gambia'or country =='St. Kitts and Nevis':
            codes.append('-')
        elif country =='Macedonia' or country =='Vanuatu' or country =='St. Lucia':
            codes.append('-')
        elif country =='North Korea' or country=='Kosovo' or country =='International':
            codes.append('-')
        elif country == 'Slovak Republic':
            codes.append(dict_['Slovak Rep.'])
        elif country == 'Wallis and Futuna' or country =='Eritrea' or country =='French Polynesia' or country =='Macau':
            codes.append('-')
        elif country=='East Timor':
            codes.append(dict_['Timor-Leste'])
        elif country == 'Serbia-Montenegro':
            codes.append(dict_['Serbia'])
        elif country =='South Sudan':
            codes.append(dict_['Sudan'])
        else:
            codes.append(dict_[country])
    return codes
df_terrorism['attacks'] = df_terrorism['attacks'].astype(int)
df_terrorism['ISO_code'] = get_locations(df_terrorism, df_freedom_index)
analysis = df_terrorism.drop_duplicates(subset=['country_txt', 'attacks'], keep='first')
analysis = analysis[analysis['ISO_code']!='-']
fig = px.scatter_geo(analysis, 
                     size="attacks", hover_name='country_txt',
                     locations='ISO_code',
                     projection="natural earth", 
                     animation_frame="iyear",
                     color='region_txt',
                     title='Terrorism Events over years')
iplot(fig) 
df_terrorism = df_terrorism[df_terrorism['iyear']>=2008]
df_terrorism['success'] = df_terrorism['success'].map({1: 'success', 0: 'fail'})

df_terrorism.head()
import matplotlib.pyplot as plt

success = df_terrorism[df_terrorism['success']=='success'].shape[0]
fail = df_terrorism[df_terrorism['success']=='fail'].shape[0]

fig = px.pie(values=[success, fail], labels=['success', 'fail'], hole=0.7)
fig.show()
def count_success_attacks(df_terrorism):
    counts_ = []
    for i in df_terrorism.groupby(['country_txt'])['eventid'].count():
        for j in range(i):
            counts_.append(i)
    return counts_

success_attacks = df_terrorism[df_terrorism['success']=='success']
success_attacks = success_attacks.sort_values(['country_txt'], ascending=[True]) 

success_attacks = success_attacks[success_attacks['ISO_code']!='-']
success_attacks['success_attacks'] = count_success_attacks(success_attacks)

success_attacks = success_attacks.drop_duplicates(subset=['country_txt', 'success_attacks'], keep='first')

fig = px.scatter_geo(success_attacks, 
                     size="success_attacks", hover_name='country_txt',
                     locations='ISO_code',
                     projection="natural earth", 
                     color='region_txt',
                     title='Success Attacks')
iplot(fig) 
df_terrorism = df_terrorism.rename(columns={'iyear':'year'})
evalue_freedom_terrorism = pd.merge(df_terrorism, df_freedom_index, how='left', on=['ISO_code', 'year'])
evalue_freedom_terrorism.head()
evalue_freedom_terrorism = evalue_freedom_terrorism[evalue_freedom_terrorism['ISO_code']!='-']
evalue_freedom_terrorism.info()
test = evalue_freedom_terrorism[~evalue_freedom_terrorism['countries'].isna()]
test = test.drop_duplicates(subset=['country_txt', 'hf_score', 'year'], keep='first')
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(x=test['hf_score'], marker_color='salmon'))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Distribution of happiness score',
})
fig.show()

print(test['hf_score'].skew())
print(test['hf_score'].kurtosis())
print(test[test['hf_score']==test['hf_score'].max()]['country_txt'])
print(test[test['hf_score']==test['hf_score'].min()]['country_txt'])
def get_countries_by_regions(test):
    regions = {}
    for region in test['region_txt'].unique():
        tmp = test[test['region_txt']==region]
        regions[region] = len(tmp['country_txt'].unique())
    return regions

regions = get_countries_by_regions(test)
fig = go.Figure()
fig.add_trace(go.Bar(y=list(regions.values()), x=list(regions.keys()), marker_color='salmon'))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Country by region',
})
fig.show()
def get_region_mean_std(df):
    mean = {}
    std = {}
    for year in df['year'].unique():
        tmp = df[df['year']==year]
        mean[year] = tmp['hf_score'].mean()
        std[year] = tmp['hf_score'].std()
    return mean, std
mean, std = get_region_mean_std(test)
fig = go.Figure()
fig.add_trace(go.Bar(
    y = list(mean.values()), x=list(mean.keys()), marker_color='lightcoral'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Score happiness by score',
})
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=list(std.values()),
    x=list(std.keys()),
    mode='markers',
    marker=dict(size= list(map(lambda x: x * 50, list(std.values()))),
                color=['lightcoral','salmon','darksalmon',
                       'lightsalmon','crimson','red','firebrick','darkred',
                       'coral','tomato','orangered','peachpuff','papayawhip'])
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Happiness Score by region',
})
fig.show()
def get_region_mean_std(df):
    mean = {}
    std = {}
    for region in df['region_txt'].unique():
        tmp = df[df['region_txt']==region]
        mean[region] = tmp['hf_score'].mean()
        std[region] = tmp['hf_score'].std()
    return mean, std
mean, std = get_region_mean_std(test)
fig = go.Figure()
fig.add_trace(go.Bar(
    y = list(mean.values()), x=list(mean.keys()), marker_color='salmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Mean of happiness score by region',
})
fig.show()
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=list(std.values()),
    x=list(std.keys()),
    mode='markers',
    marker=dict(size= list(map(lambda x: x * 50, list(std.values()))),
                color=['lightcoral','salmon','darksalmon',
                       'lightsalmon','crimson','red','firebrick','darkred',
                       'coral','tomato','orangered','peachpuff','papayawhip'])
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Happiness Score by region',
})
fig.show()
import scipy
from scipy.stats import f_oneway
from sklearn.preprocessing import StandardScaler

b = test[test['region_txt']=='South America']

def standard(df, column, region):
    values = []
    current_region = df[df['region_txt']==region]
    for year in current_region['year'].unique():
        tmp = current_region[current_region['year']==year]
        to_scale = []
        for value in tmp[column]:
            to_scale.append([value])
        scaler = StandardScaler().fit_transform(to_scale)
        for norm in scaler:
            values.append(norm[0])
    return values

b = b.sort_values(by=['year'], ascending=False)
b['stand'] = standard(b, 'hf_score', 'South America')

from statsmodels.graphics.gofplots import qqplot
def qq_plot(b, column):
    qqplot_data = qqplot(b[column], line='45').gca().lines
    return qqplot_data
qq_plot(test, 'hf_score')
from scipy import stats
regions =[]
for region in test['region_txt'].unique():
    tmp = test[test['region_txt']==region]
    regions.append(tmp['hf_score'])

statistic, p_value = stats.kruskal(regions[0],regions[1],regions[2],regions[3],regions[4],
                                    regions[5],regions[6],regions[7],regions[8],regions[9],
                                    regions[10],regions[11])
critical_value = 0.05
if critical_value < p_value:
    print('All distribution are same')
else:
    print('That are some distribution that is diferent')
import scikit_posthocs
frame = scikit_posthocs.posthoc_nemenyi([regions[0],regions[1],regions[2],regions[3],regions[4],
                                    regions[5],regions[6],regions[7],regions[8],regions[9],
                                    regions[10],regions[11]])
def highlight_max(s):
    return ['background-color: salmon' if v < 0.05 else '' for v in s]

frame = frame.set_index(test['region_txt'].unique())
values = []
for i in range(1, 13):
    values.append(i)
map_ = dict(zip(values, test['region_txt'].unique()))
frame = frame.rename(columns=map_)
frame.style.apply(highlight_max)
hf_frame = test[['hf_score','pf_ss_women','pf_ss', 'pf_movement_foreign', 'pf_religion', 'pf_association_political_establish', 'pf_expression_jailed','pf_expression_influence','pf_expression_control','pf_expression_newspapers','pf_expression_internet','pf_score','ef_government','ef_legal_judicial','ef_money_growth','ef_money_inflation','ef_money_currency', 'ef_money']]
hf_frame = hf_frame.replace('-', -1)
hf_frame = hf_frame.apply(pd.to_numeric)
scaler = StandardScaler().fit_transform(hf_frame)
hf_frame_std = pd.DataFrame(scaler, columns=hf_frame.columns)
qq_plot(hf_frame_std, 'hf_score')
from factor_analyzer.factor_analyzer import  FactorAnalyzer
import matplotlib.pyplot as plt
fa = FactorAnalyzer()
fa.fit(hf_frame_std, 25)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
plt.scatter(range(1, hf_frame_std.shape[1]+1),ev)
plt.plot(range(1, hf_frame_std.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
fa = FactorAnalyzer(n_factors=5)

fa.fit(hf_frame)
factors = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2', 'Factor3', 'Factor4','Factor5'])
factors = factors.set_index(hf_frame.columns)
def highlight_max_(s):
    return ['background-color: tomato' if v > 0.49 or v < -0.49 else '' for v in s]

factors.style.apply(highlight_max_)
df_2009 = test[test['year']==2009]

def plot_scatter(df, column):
    fig = go.Figure()
    colors= ['lightcoral','salmon','darksalmon',
            'lightsalmon','crimson','red','firebrick','darkred',
            'coral','tomato','orangered','peachpuff','papayawhip']

    for region, color in zip(df['region_txt'].unique(), colors):
        tmp = df[df['region_txt']==region]
        fig.add_trace(go.Scatter(
            y=list(tmp['hf_score']),
            x=list(tmp[column]),
            marker=dict(size= list(map(lambda x: x * 3, list(tmp['hf_score']))),
                        color=color),
            mode='markers',
            name=region
        ))
    fig.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'title': 'Distribution of happiness score',
    })
    fig.show()
    
plot_scatter(df_2009, 'ef_money_growth')
plot_scatter(df_2009, 'ef_money_currency')
plot_scatter(df_2009, 'ef_money_inflation')
plot_scatter(df_2009, 'ef_money')
def column_correlation(df_tmp, list_):
    correlation = []
    for column in list_:
        df_tmp[column] = df_tmp[column].replace('-', 0)
        df_tmp[column] = df_tmp[column].astype(float) 
        corr = np.corrcoef(df_tmp[column], df_tmp['hf_score'])[0][1]
        correlation.append(corr)
    return correlation
list_ = ['ef_money_inflation', 'ef_money', 'ef_money_growth', 'ef_money_currency']
correlation = column_correlation(df_2009, list_)
correlation
fig = px.scatter(test, x="pf_score", y="hf_score", size='attacks', hover_data=['country_txt', 'year'], color='region_txt')
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
import plotly.graph_objects as go

df_terrorism_gname = df_terrorism.pivot_table(columns='gname', 
                                              aggfunc='size', fill_value=0)

unknown = df_terrorism[df_terrorism['gname']=='Unknown'].shape[0]
known = df_terrorism[df_terrorism['gname']!='Unknown'].shape[0]

fig = px.pie(values=[unknown, known], labels=['Unknown', 'Known'], hole=0.7)
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
import plotly.graph_objects as go
known = df_terrorism[df_terrorism['gname']!='Unknown']
known = known.pivot_table(columns='gname', aggfunc='size', fill_value=0)
terror_gname = dict(zip(known.index, known[:]))
terror_gname = sorted(terror_gname.items(), key=lambda kv: kv[1], reverse=True)
terror_gname = dict(terror_gname)
terror_gname_100_keys = list(terror_gname.keys())
terror_gname_100_values = list(terror_gname.values())
terror_gname_100_values = terror_gname_100_values[:100]
terror_gname_100_keys =terror_gname_100_keys[0:100]
fig = go.Figure()
fig.add_trace(go.Bar(
    y = terror_gname_100_values, x=terror_gname_100_keys, marker_color='salmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
place_attacks = df_terrorism.pivot_table(columns='targtype1_txt', aggfunc='size', fill_value=0)
place_attacks = dict(zip(place_attacks.index, place_attacks[:]))
place_attacks = sorted(place_attacks.items(), key=lambda kv: kv[1], reverse=True)
place_attacks = dict(place_attacks)
place_attacks_keys = list(place_attacks.keys())
place_attacks_values = list(place_attacks.values())
place_attacks_values = place_attacks_values[:100]
place_attacks_keys =place_attacks_keys[0:100]
fig = go.Figure()
fig.add_trace(go.Bar(
    y = place_attacks_values, x=place_attacks_keys, marker_color='salmon'
))
#fig = go.Figure([go.Bar(x=place_attacks_keys, y=place_attacks_values)])
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
weapons = df_terrorism.pivot_table(columns='weaptype1_txt', aggfunc='size', fill_value=0)
weapons = dict(zip(weapons.index, weapons[:]))
weapons = sorted(weapons.items(), key=lambda kv: kv[1], reverse=True)
weapons_ = dict(weapons)
weapons_keys = list(weapons_.keys())
weapons_values = list(weapons_.values())
weapons_values = weapons_values[:100]
weapons_keys = weapons_keys[0:100]
fig = go.Figure()
fig.add_trace(go.Bar(
    y = weapons_values, x=weapons_keys, marker_color='salmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
attacks = df_terrorism.pivot_table(columns='attacktype1_txt', aggfunc='size', fill_value=0)
attacks = dict(zip(attacks.index, attacks[:]))
attacks = sorted(attacks.items(), key=lambda kv: kv[1], reverse=True)
attacks = dict(attacks)
attacks_keys = list(attacks.keys())
attacks_values = list(attacks.values())
attacks_values = attacks_values[:100]
attacks_keys = attacks_keys[0:100]
fig = go.Figure()

fig.add_trace(go.Bar(
    x = attacks_keys, y=attacks_values, marker_color='salmon'
))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()

import plotly.express as px
fig = px.box(test, x="region_txt", y="hf_score", hover_data={'country_txt'})
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
from plotly.subplots import make_subplots

def attack_forms(df, gnames):
    i = 1
    fig = make_subplots(rows=len(gnames), cols=5, vertical_spacing=0.09)
    for name in gnames:
        tmp = df[df['gname']==name]
        country_attacks = tmp.pivot_table(columns='region_txt', aggfunc='size', fill_value=0)
        country_attacks = dict(zip(country_attacks.index, country_attacks[:]))
        country_attacks = sorted(country_attacks.items(), key=lambda kv: kv[1], reverse=True)
        country_attacks = dict(country_attacks)
        country_attacks_keys = list(country_attacks.keys())
        country_attacks_values = list(country_attacks.values())
        country_attacks_values = country_attacks_values[:100]
        country_attacks_keys =country_attacks_keys[0:100]
        fig.add_trace(go.Bar(x=country_attacks_keys, y=country_attacks_values), i, 1)
        
        country_attacks = tmp.pivot_table(columns='country_txt', aggfunc='size', fill_value=0)
        country_attacks = dict(zip(country_attacks.index, country_attacks[:]))
        country_attacks = sorted(country_attacks.items(), key=lambda kv: kv[1], reverse=True)
        country_attacks = dict(country_attacks)
        country_attacks_keys = list(country_attacks.keys())
        country_attacks_values = list(country_attacks.values())
        country_attacks_values = country_attacks_values[:100]
        country_attacks_keys =country_attacks_keys[0:100]
        fig.add_trace(go.Bar(x=country_attacks_keys, y=country_attacks_values), i, 2)
            
        place_attacks = tmp.pivot_table(columns='targtype1_txt', aggfunc='size', fill_value=0)
        place_attacks = dict(zip(place_attacks.index, place_attacks[:]))
        place_attacks = sorted(place_attacks.items(), key=lambda kv: kv[1], reverse=True)
        place_attacks = dict(place_attacks)
        place_attacks_keys = list(place_attacks.keys())
        place_attacks_values = list(place_attacks.values())
        place_attacks_values = place_attacks_values[:100]
        place_attacks_keys =place_attacks_keys[0:100]
        fig.add_trace(go.Bar(x=place_attacks_keys, y=place_attacks_values), i, 3)
            
        attacks = tmp.pivot_table(columns='attacktype1_txt', aggfunc='size', fill_value=0)
        attacks = dict(zip(attacks.index, attacks[:]))
        attacks = sorted(attacks.items(), key=lambda kv: kv[1], reverse=True)
        attacks = dict(attacks)
        attacks_keys = list(attacks.keys())
        attacks_values = list(attacks.values())
        attacks_values = attacks_values[:100]
        attacks_keys = attacks_keys[0:100]
        fig.add_trace(go.Bar(x=attacks_keys, y=attacks_values), i, 4)
        
        weapons = tmp.pivot_table(columns='weaptype1_txt', aggfunc='size', fill_value=0)
        weapons = dict(zip(weapons.index, weapons[:]))
        weapons = sorted(weapons.items(), key=lambda kv: kv[1], reverse=True)
        weapons_ = dict(weapons)
        weapons_keys = list(weapons_.keys())
        weapons_values = list(weapons_.values())
        weapons_values = weapons_values[:100]
        weapons_keys = weapons_keys[0:100]
        fig.add_trace(go.Bar(x=weapons_keys, y=weapons_values), i, 5)
        i+=1
        fig.update_layout(height=8000, width=1000, showlegend=False, title_text=name + " Attack forms")
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()
regions = attack_forms(df_terrorism, terror_gname_100_keys[:10])