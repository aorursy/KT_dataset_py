import pandas as pd

from math import pi

from pathlib import Path

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from wordcloud import WordCloud, STOPWORDS

from scipy.stats.stats import pearsonr

import seaborn as sns

import numpy as np

import matplotlib as mpl

plt.rcParams["patch.force_edgecolor"] = True

plt.style.use('fivethirtyeight')

mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)

%matplotlib inline

init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None # Warning for chained copies disabled
df = pd.read_csv('../input/en.openfoodfacts.org.products.tsv', low_memory=False, sep='\t')

df.info()
def filling_factor(df):

    missing_df = df.isnull().sum(axis=0).reset_index()

    missing_df.columns = ['column_name', 'missing_count']

    missing_df['filling_factor'] = (df.shape[0]-missing_df['missing_count'])/df.shape[0]*100

    missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)

    return missing_df

#____________________________________

missing_df = filling_factor(df)

missing_df[missing_df['filling_factor'] == 0]
df = df.dropna(axis = 1, how = 'all')

df.shape
#_______________________

# looking at empty raws

composant = []

for s in df.columns:

    if '_100g' in s: composant.append(s)

df_subset_columns = df[composant]

print('empty _100g raws: {}'.format(df_subset_columns.isnull().all(axis=1).sum()))

#___________________

# and deleting them

df_new = df[df_subset_columns.notnull().any(axis=1)]
list_columns = ['categories', 'categories_tags', 'categories_en']

df_new[df_new[list_columns].notnull().any(axis=1)][['product_name']+ list_columns][:20:3]
columns_to_remove = []

for s in df_new.columns:

    if "_en" in s: 

        t = s.replace('_en', '')

        u = s.replace('_en', '_tags')

        print("{:<20} 'no suffix' -> {} ; '_tags' suffix -> {}".format(s,

                                t in df_new.columns, u in df_new.columns))

        if t in df_new.columns: columns_to_remove.append(t)

        if u in df_new.columns: columns_to_remove.append(u)

df_new.drop(columns_to_remove, axis = 1, inplace = True)
def count_words(df, colonne = 'categories_en'):

    list_words = set()

    for word in df[colonne].str.split(','):

        if isinstance(word, float): continue

        list_words = set().union(word, list_words)       

    print("Nb of categories in '{}': {}".format(colonne, len(list_words)))

    return list(list_words)
list_countries = count_words(df, 'countries_en')
country_replacement = {'Tunisie': 'Tunisia', 'Niederlande': 'Netherland', 

    'fr:Bourgogne-aube-nogent-sur-seine':'France', 'fr:Sverige': 'Sweden', 

    'Vereinigtes-konigreich': 'United Kingdom',  'fr:Suiza':'Switzerland',

    'fr:Kamerun':'Cameroon', 'Other-japon':'Japon', 'fr:Marokko':'Morocco', 

    'ar:Tunisie':'Tunisia', 'fr:Marseille-5':'France', 'Australie':'Australia',

    'fr:Marseille-6':'France', 'fr:Scotland':'United Kingdom', 'Soviet Union':'Russia',

    'fr:Vereinigte-staaten-von-amerika':'United States', 'fr:Neukaledonien':'France',

    'fr:Nederland':'Netherland', 'Mayotte':'France', 'Spanje':'Spain', 'Frankrijk':'France',

    'Suisse':'Switzerland', 'fr:Belgie':'Belgium', 'Other-turquie':'Turkey',

    'fr:Spanien':'Spain', 'Pays-bas':'Netherland', 'fr:Saudi-arabien':'Saudi Arabia',

    'Virgin Islands of the United States':'United States', 'fr:England':'England',

    'Allemagne':'Germany', 'fr:Vereinigtes-konigreich':'United Kingdom', 'Belgique':'Belgium',

    'United-states-of-america':'United States', 'Réunion':'France', 'Martinique':'France',

    'Guadeloupe':'France','French Guiana':'France', 'Czech':'Czech Republic', 'Quebec':'Canada',

    'fr:Quebec':'Canada', 'fr:Deutschland':'Germany', 'Saint Pierre and Miquelon':'France'}
for index, countries in df['countries_en'].str.split(',').items():

    if isinstance(countries, float): continue

    country_name = []

    found = False

    for s in countries:

        if s in country_replacement.keys():

            found = True

            country_name.append(country_replacement[s])

        else:

            country_name.append(s)

    if found:

        df.loc[index, 'countries_en'] = ','.join(country_name)    
list_countries = count_words(df, 'countries_en')
country_count = dict()

for country in list(list_countries):

    country_count[country] = df['countries_en'].str.contains(country).sum()
data = dict(type='choropleth',

locations = list(country_count.keys()),

locationmode = 'country names', z = list(country_count.values()),

text = list(country_count.keys()), colorbar = {'title':'Product nb.'},

colorscale=[[0.00, 'rgb(204,255,229)'], [0.01, 'rgb(51,160,44)'],

            [0.02, 'rgb(102,178,255)'], [0.03, 'rgb(166,206,227)'],

            [0.05, 'rgb(31,120,180)'], [0.10, 'rgb(251,154,153)'],

            [0.20, 'rgb(255,255,0)'], [1, 'rgb(227,26,28)']])

layout = dict(title='Availability of products per country',

geo = dict(showframe = True, projection={'type':'Mercator'}))

choromap = go.Figure(data = [data], layout = layout)

iplot(choromap, validate=False)
category_keys = count_words(df_new, 'categories_en')
count_keyword = dict()

for index, col in df_new['categories_en'].iteritems():

    if isinstance(col, float): continue

    for s in col.split(','):

        if s in count_keyword.keys():

            count_keyword[s] += 1

        else:

            count_keyword[s] = 1



keyword_census = []

for k,v in count_keyword.items():

    keyword_census.append([k,v])

keyword_census.sort(key = lambda x:x[1], reverse = True)

    
keyword_census[:5]
#_____________________________________________

# Function that control the color of the words

def random_color_func(word=None, font_size=None, position=None,

                      orientation=None, font_path=None, random_state=None):

    h = int(360.0 * tone / 255.0)

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)

#_____________________________________________

# UPPER PANEL: WORDCLOUD

fig = plt.figure(1, figsize=(11,9))

ax1 = fig.add_subplot(1,1,1)

#_______________________________________________________

# I define the dictionary used to produce the wordcloud

words = dict()

trunc_occurences = keyword_census[0:100]

for s in trunc_occurences:

    words[s[0]] = s[1]

tone = 170.0 # define the color of the words

#________________________________________________________

wordcloud = WordCloud(width=900,height=500, background_color='lightgray', 

                      max_words=1628,relative_scaling=0.6,

                      color_func = random_color_func,

                      normalize_plurals=False)

wordcloud.generate_from_frequencies(words)

ax1.imshow(wordcloud, interpolation="bilinear")

ax1.axis('off')

plt.show()
pnns_group1_keys = count_words(df_new, 'pnns_groups_1')  

pnns_group2_keys = count_words(df_new, 'pnns_groups_2')  
pnns_group1_keys
corresp = dict()

corresp['cereals-and-potatoes']  = 'Cereals and potatoes'

corresp['fruits-and-vegetables'] = 'Fruits and vegetables'

corresp['sugary-snacks'] = 'Sugary snacks'

corresp['salty-snacks'] = 'Salty snacks'
df_new['pnns_groups_1'] = df_new['pnns_groups_1'].replace(corresp)

pnns_group1_keys = count_words(df_new, 'pnns_groups_1')

pnns_group1_keys
col_filling = filling_factor(df_new)
y_axis  = list(col_filling['filling_factor'])

x_axis  = [i for i in range(len(col_filling))]

x_label = list(col_filling['column_name'])

fig = plt.figure(figsize=(10, 22))

plt.yticks(x_axis, x_label)

plt.title('Filling factor (%)', fontsize = 15)

plt.barh(x_axis, y_axis)

plt.show()
col_filling_2 = col_filling.loc[col_filling['filling_factor'] < 0.2]

y_axis  = list(col_filling_2['filling_factor'])

x_axis  = [i for i in range(len(col_filling_2))]

x_label = list(col_filling_2['column_name'])

fig = plt.figure(figsize=(11, 8))

plt.xticks(rotation=90)

plt.xticks(x_axis, x_label)

plt.ylabel('Filling factor (%)', fontsize = 15)

plt.bar(x_axis, y_axis)

plt.axhline(y=0.02, linewidth=2, color = 'r')

plt.text(5, 0.025, 'threshold for deletion', fontsize = 16, color = 'r')

plt.tight_layout()

plt.show()
columns_to_remove = list(col_filling[df_new.shape[0] - 

                                     col_filling['missing_count'] < 70]['column_name'])

columns_to_remove
df_new.drop(columns_to_remove, axis = 1, inplace = True)
df_new.rename(columns={'biotin_100g':'vitamin-b7_100g'}, inplace=True)

df_new.rename(columns={'pantothenic-acid_100g':'vitamin-b5_100g'}, inplace=True)

df_new.rename(columns={'vitamin-pp_100g':'vitamin-b3_100g'}, inplace=True)
quantite = ['energy_100g', 'vitamin-a_100g', 'vitamin-c_100g', 'vitamin-b3_100g',

            'vitamin-b6_100g','vitamin-b9_100g','vitamin-b5_100g', 'vitamin-b7_100g',

            'vitamin-b12_100g', 'vitamin-e_100g', 'zinc_100g','copper_100g']
sigma = [0 for _ in range(12)]

mediane = [0 for _ in range(12)]

for i in range(len(quantite)):

    colonne = quantite[i]

    mediane[i] = df_new[pd.notnull(df_new[colonne])][colonne].median()

    test = df_new[pd.notnull(df_new[colonne])][colonne]

    test = test.sort_values()    

    if i != 4: sigma[i] = np.std(test[:-15])

    else :     sigma[i] = np.std(test[:-25])
#plt.style.use('ggplot')

tPlot, axes = plt.subplots(nrows=4, ncols=3, sharex=False, sharey=False, figsize=(11,11))

axes = np.array(axes)



i=0

for ax in axes.reshape(-1):

    colonne = quantite[i]

    test = df_new[pd.notnull(df_new[colonne])][colonne]

    ax.tick_params(labelcolor='black',top='off',bottom='on',left='on',right='off',labelsize=8)

    ax.set_ylabel(colonne.rstrip("_100g"), fontsize = 12)

    ax.set_yscale("log")

    ax.plot(list(test), 'b.', markeredgewidth = 0.3, markeredgecolor='w')

    for tick in ax.get_xticklabels():

        tick.set_rotation(30)

    ax.axhline(y=mediane[i] + 12*sigma[i], color='r', linestyle='-')

    ax.text(0., 0.02, 'median:{:.3} \n sigma:{:.3}'.format(mediane[i], sigma[i]),

            style='italic', transform=ax.transAxes, fontsize = 12,

            bbox={'facecolor':'green', 'alpha':0.5, 'pad':0})

    i += 1



tPlot.text(0.5, 1.01, "Outliers ?", ha='center', fontsize = 14)

plt.tight_layout()
df_new[df_new['energy_100g'] > 100000][['product_name', 'energy_100g']]
for i in range(len(quantite)):

    colonne = quantite[i]

    print('{:<30}: deletion if > {}'.format(colonne, round(mediane[i] + 12*sigma[i],3)))

    mask1 = df_new[colonne] > (mediane[i] + 12*sigma[i])

    df_new = df_new.drop(df_new[mask1].index)
tPlot, axes = plt.subplots(nrows=4, ncols=3, sharex=False, sharey=False, figsize=(11,9))

axes = np.array(axes)



i=0

for ax in axes.reshape(-1):

    colonne = quantite[i]

    mediane_2 = df_new[pd.notnull(df_new[colonne])][colonne].median()

    test = df_new[pd.notnull(df_new[colonne])][colonne]  

    ax.set_ylabel(colonne.rstrip("_100g"), fontsize = 16)

    ax.tick_params(labelcolor='black', top='off', bottom='on', left='on',

                   right='off', labelsize = 11)

    for tick in ax.get_xticklabels():

        tick.set_rotation(30)

    

    if   i == 1: ax.hist(test, bins=np.linspace(0,0.0025,50))

    elif i == 0: ax.hist(test, bins=range(0,6000,125))

    elif i == 2: ax.hist(test, bins=np.linspace(0,0.3,50))

    elif i == 3: ax.hist(test, bins=np.linspace(0,0.06,50))

    elif i == 4: ax.hist(test, bins=np.linspace(0,0.005,50))

    elif i == 5: ax.hist(test, bins=np.linspace(0,0.001,50))

    elif i == 6: ax.hist(test, bins=np.linspace(0,0.01,50))

    elif i == 7: ax.hist(test, bins=np.linspace(0,0.0001,50))

    elif i == 8: ax.hist(test, bins=np.linspace(0,0.00001,50))

    elif i == 9: ax.hist(test, bins=np.linspace(0,0.1,50))

    elif i == 10: ax.hist(test, bins=np.linspace(0,0.03,50))

    elif i == 11: ax.hist(test, bins=np.linspace(0,0.004,50))

    

    ax.text(0.6, 0.92, 'median:{:.3}'.format(mediane_2), style='italic', fontsize = 12,

            transform=ax.transAxes, bbox={'facecolor':'green','alpha':0.5,'pad':0})

    i += 1

    

tPlot.text(0.5, 1.01, "Distribution of products", ha='center', fontsize = 14)

plt.tight_layout()
sns.set(context="paper", font_scale = 1.2)

corrmat = df_new.corr()

f, ax = plt.subplots(figsize=(12, 12))

f.text(0.45, 0.93, "Pearson's correlation coefficients", ha='center', fontsize = 18)

sns.heatmap(corrmat, square=True, linewidths=0.01, cmap="coolwarm")

plt.tight_layout()
sns.set(context="paper", font_scale = 1.2)

f, ax = plt.subplots(figsize=(11, 11))

cols = corrmat.nlargest(25, 'carbohydrates_100g')['carbohydrates_100g'].index

cm = corrmat.loc[cols, cols] 

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 9}, linewidth = 0.1, cmap = 'coolwarm',

                 yticklabels=cols.values, xticklabels=cols.values)

f.text(0.5, 0.93, "Correlation with carbohydrates", ha='center', fontsize = 18)

plt.show()
sns.set(context="paper", font_scale = 1.2)

f, ax = plt.subplots(figsize=(11, 11))

cols = corrmat.nlargest(25, 'nutrition-score-uk_100g')['nutrition-score-uk_100g'].index

cm = corrmat.loc[cols, cols] 

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 9}, linewidth = 0.1, cmap = 'coolwarm',

                 yticklabels=cols.values, xticklabels=cols.values)

f.text(0.5, 0.93, "Correlation with nutrition score", ha='center', fontsize = 18)

plt.show()
df_new['pnns_groups_1'].unique()
categ_prod = [] ; label_prod = []

for i,s in enumerate(df_new['pnns_groups_1'].unique()):

    if isinstance(s, float): continue

    if s == 'unknown': continue    

    produit = str(s)

    df1 = df_new[df_new['pnns_groups_1'] == produit]

    table_1 = pd.Series(df1[pd.notnull(df1['nutrition-score-uk_100g'])]['nutrition-score-uk_100g'])

    categ_prod.append(table_1)

    label_prod.append(s)
tPlot, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False, figsize=(11,7))

axes = np.array(axes)



i=0

for ax in axes.reshape(-1):

    t = categ_prod[i].value_counts(sort=True)

    t.sort_index(inplace=True)

    hist = list(t)

    bins = list(t.index)

    

    ax.tick_params(labelcolor='black', top='off', bottom='on', left='on', right='off')

    ax.set_ylabel(label_prod[i], fontsize = 12)

    ax.hist(categ_prod[i], bins=range(-20,36,2), edgecolor = 'k')



    mediane = int(categ_prod[i].median())

    color = 'green' if mediane <= 4 else 'red'

    if mediane < 11:

        ax.text(0.68, 0.9, 'mediane:{}'.format(int(categ_prod[i].median())), style='italic',

                transform=ax.transAxes, fontsize = 12,

                bbox={'facecolor':color, 'alpha':0.8, 'pad':5})

    else:

        ax.text(0.02, 0.9, 'mediane:{}'.format(int(categ_prod[i].median())), style='italic',

                transform=ax.transAxes, fontsize = 12,

                bbox={'facecolor':'red', 'alpha':0.8, 'pad':5})

    i += 1



tPlot.text(0.5, 1.01, 'nutrition sore', ha='center', fontsize = 18)

plt.tight_layout()

liste_columns = ['fat_100g', 'saturated-fat_100g', 'sugars_100g', 'carbohydrates_100g',

                 'proteins_100g', 'salt_100g']
df_new[liste_columns][:5]
df_chart = df_new[pd.notnull(df_new[liste_columns]).all(axis = 1)]

mean_values = list(df_chart[liste_columns].mean().values)
def spider(values, cat, ax):

    N = len(cat)

    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # for circularity

    values += values[:1]

    x_as += x_as[:1]

    # Set color of axes

    plt.rc('axes', linewidth=0.5, edgecolor="#888888")

    # Set clockwise rotation. That is:

    ax.set_theta_offset(pi / 2)

    ax.set_theta_direction(-1)

    # Set position of y-labels

    ax.set_rlabel_position(0)

    # Set color and linestyle of grid

    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    # Set ticks values and labels    

    ax.set_xticks(x_as[:-1])

    ax.set_xticklabels([])

    ax.set_yticks([0.1, 0.5, 1, 2, 10])

    ax.set_yticklabels(["0.1", "0.5", "1", "2", "10"])

    # Plot data

    ax.plot(x_as, values, linewidth=0, linestyle='solid', zorder=3)

    # Fill area

    ax.fill(x_as, values, 'b', alpha=0.3)

    # Set axes limits

    ax.set_ylim(0, 3)

    # Draw ytick labels to make sure they fit properly

    for i in range(N):

        angle_rad = i / float(N) * 2 * pi

        if angle_rad == 0:

            ha, distance_ax = "center", 3

        elif 0 < angle_rad < pi:

            ha, distance_ax = "left", 3

        elif angle_rad == pi:

            ha, distance_ax = "center", 3

        else:

            ha, distance_ax = "right", 3



        ax.text(angle_rad, 0.2+distance_ax, cat[i], size=10,

                horizontalalignment=ha, verticalalignment="center")
fig, axes = plt.subplots(nrows=3, ncols=3, subplot_kw=dict(projection='polar'), figsize=(11,11))

axes = np.array(axes)



list_nutriments = [s.strip('_100g') for s in liste_columns]



ind = 0

for ax in axes.reshape(-1):

    ind += 1

    ind2 = 4*ind

    absolute_values = list(df_chart.iloc[ind2][liste_columns].T.values)

    values  = [ val/mean_values[i] for i, val in enumerate(absolute_values)]

    spider(values, list_nutriments, ax)

    ax.set_title(df_chart.iloc[ind2]['product_name'], fontsize = 15)

    

fig.subplots_adjust(hspace=0.5)

plt.show()