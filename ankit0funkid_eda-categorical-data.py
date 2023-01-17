# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib.patches import Patch

from matplotlib.lines import Line2D
data_df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
# Getting age values as int

age_val = []

for age in data_df['Age']:

#     print(age)

    if pd.isnull(age):

        age_val.append(np.NaN)

    elif age == 'all':

        age_val.append(0)

    else:

        age_val.append(int(age.strip('+')))



# Getting Rotten Tomatoes values as int

RT_val = []

for RT_percentage in data_df['Rotten Tomatoes']:

#     print(age)

    if pd.isnull(RT_percentage):

        RT_val.append(np.NaN)

#     elif age == 'all':

#         RT_val.append(0)

    else:

        RT_val.append(float(RT_percentage.strip('%')))

        

data_df['Age'] = age_val

data_df['Rotten Tomatoes'] = RT_val
data_df.info()
# Seperating the data based on categorical values

# Genre Dummies

# Language Dummies

# Country

# Directors
def getCatDummies(data,columns):

    data_temp = data.copy()

    unq_values = {}

    for column in columns:

        col_dummies = data_temp[column].str.get_dummies(',')

        unq_values[column] = pd.unique(data_temp[column].str.split(',',expand = True).values.ravel())

        data_temp = pd.concat([data_temp, col_dummies], axis = 1, sort = False)

    return data_temp,unq_values
data_df_dum,unq_values = getCatDummies(data_df,['Genres','Language','Country'])

data_df_dum.head()
# #Splitting data based on the platform

# Netflix               

# Hulu                  

# Prime Video           

# Disney+ 
netflix_df = data_df_dum[data_df_dum['Netflix'] == 1].copy()

hulu_df = data_df_dum[data_df_dum['Hulu'] == 1].copy()

prime_df = data_df_dum[data_df_dum['Prime Video'] == 1].copy()

disney_df = data_df_dum[data_df_dum['Disney+'] == 1].copy()



netflix_df.drop(['Hulu', 'Prime Video', 'Disney+', 'Type', 'Unnamed: 0','Genres','Language','Country','Rotten Tomatoes'],axis = 1,inplace = True)

hulu_df.drop(['Netflix', 'Prime Video', 'Disney+', 'Type', 'Unnamed: 0','Genres','Language','Country','Rotten Tomatoes'],axis = 1,inplace = True)

prime_df.drop(['Hulu', 'Netflix', 'Disney+', 'Type', 'Unnamed: 0','Genres','Language','Country','Rotten Tomatoes'],axis = 1,inplace = True)

disney_df.drop(['Hulu', 'Prime Video', 'Netflix', 'Type', 'Unnamed: 0','Genres','Language','Country','Rotten Tomatoes'],axis = 1,inplace = True)
# Distributing data into bins based on ratings



def rating_bin(data,src_column,bins = [0,2,4,6,8,10]

               ,names = ['Low [0 - 2]'

                         ,'Moderaly Low [2 - 4]'

                         ,'Medium [4 - 6]'

                         ,'Moderately High [6  - 8]'

                         ,'High [8 - 10]']):

    d = dict(enumerate(names, 1))

    return np.vectorize(d.get)(np.digitize(data[src_column], bins)),d
netflix_df['Rating'],rating_dict = rating_bin(netflix_df,'IMDb')

hulu_df['Rating'],rating_dict = rating_bin(hulu_df,'IMDb')

prime_df['Rating'],rating_dict = rating_bin(prime_df,'IMDb')

disney_df['Rating'],rating_dict = rating_bin(disney_df,'IMDb')
# Let's work only on Netflix data for now

# Will move ahead with rest on same trail
netflix_df.drop(['Age'],axis = 1,inplace = True)
def get_categories(category):

    categ = unq_values[category]

    categ = [x for x in categ if x == x and x is not None]

    return categ
def catData_list(data_df_,category,rating_dict):

    categVals = get_categories(category)

    catData_list = []

    for cat in categVals:

        temp = []

        data_df_x = data_df_[netflix_df[cat] == 1]

        temp.append(cat)

        grp_data = data_df_x.groupby(['Rating']).sum()[cat].sort_index()

        if rating_dict[1] in grp_data.index:

            low = grp_data[rating_dict[1]]

        else:

            low = 0

            

        if rating_dict[2] in grp_data.index:

            modlow = grp_data[rating_dict[2]]

        else:

            modlow = 0

            

        if rating_dict[3] in grp_data.index:

            med = grp_data[rating_dict[3]]

        else:

            med = 0

            

        if rating_dict[4] in grp_data.index:

            modHigh = grp_data[rating_dict[4]]

        else:

            modHigh = 0

            

        if rating_dict[5] in grp_data.index:

            high = grp_data[rating_dict[5]]

        else:

            high = 0

        if 'None' in grp_data.index:

            none = grp_data['None']

        else:

            none = 0

               

        temp.extend([low,modlow,med,modHigh,high,none,sum(data_df_x.groupby(['Rating']).sum()[cat])])

        catData_list.append(temp)

    return catData_list
def cat_count_df(data_df_,category,rating_dict):

    catData_lists = catData_list(data_df_,category,rating_dict)

    cat_counts_df = pd.DataFrame(data = catData_lists,columns = ['Genre','Low','Mod Low','Med','Mod High','High','None','Total'])

    cat_counts_df = cat_counts_df.sort_values(['Total'],ascending = False)

    return cat_counts_df
def cat_per_df(cat_counts_df):

    data_list = []

    rows = len(cat_counts_df)

    for i in range(rows):

        temp_ls = []

        data_sr = cat_counts_df.iloc[i]

        total = int(data_sr['Total'])

        genre = data_sr['Genre']

        low = int(data_sr['Low'])

        mod_low = int(data_sr['Mod Low'])

        med = int(data_sr['Med'])

        mod_high = int(data_sr['Mod High'])

        high = int(data_sr['High'])

        none = int(data_sr['None'])

        temp_ls = np.array([total,low,mod_low,med,mod_high,high,none])/total

        temp_ls *= 100

        temp_ls = list(temp_ls)

        temp_ls.insert(0,genre)

        data_list.append(list(temp_ls))



    cat_per_df = pd.DataFrame(data = data_list,columns = ['Genre','Total','Low','Mod_Low','Med','Mod_High','High','None'])

    cat_per_df = cat_per_df.round({'Low':2,'Mod_Low':2,'Med':2,'Mod_High':2,'High':2,'None':2})

    return cat_per_df
# Plotting



def count_plot_cat(cat_counts_df,x = 15,y = 10,z = -0.15):

    plt.figure(figsize = (x, y))

#     sns.set_color_codes('pastel')

    sns.barplot(y = cat_counts_df['Genre']

                ,x = cat_counts_df['Total']

                ,dodge = False

               ,color = 'b')

    sns.barplot(y = cat_counts_df['Genre']

                ,x = cat_counts_df['Mod High']

                ,dodge = False

               ,color = '#C0EBE9')

    sns.barplot(y = cat_counts_df['Genre']

                ,x = cat_counts_df['Med']

                ,dodge = False

               ,color = '#F3F3AF')

    sns.barplot(y = cat_counts_df['Genre']

                ,x = cat_counts_df['High']

                ,dodge = False

               ,color = '#98DAA7')

    sns.barplot(y = cat_counts_df['Genre']

                ,x = cat_counts_df['Mod Low']

                ,dodge = False

               ,color = '#F3ABA8')

    # sns.lineplot(y = cat_counts_df['Genre']

    #             ,x = cat_counts_df['Total'])

    legend_elements = [Patch(facecolor='#C0EBE9',

                             label='Moderately High'),

                       Patch(facecolor='#F3F3AF',

                             label='Medium'),

                       Patch(facecolor='#98DAA7',

                             label='High'),

                      Patch(facecolor='#F3ABA8',

                             label='Moderately Low')]

    plt.legend(handles=legend_elements,loc='lower center', bbox_to_anchor=(0.5, z),

              ncol=5, fancybox=True)

    plt.show()



def perc_plot_cat(cat_per_df,cat_counts_df,x = 15,y = 10,z = -0.15):

    plt.figure(figsize = (x, y))

#     sns.set_color_codes('pastel')

    sns.barplot(y = cat_per_df['Genre']

                ,x = cat_per_df['Total']

                ,dodge = False

               ,color = 'b')

    sns.barplot(y = cat_per_df['Genre']

                ,x = cat_per_df['Mod_High']

                ,dodge = False

               ,color = '#C0EBE9')

    sns.barplot(y = cat_per_df['Genre']

                ,x = cat_per_df['Med']

                ,dodge = False

               ,color = '#F3F3AF')

    sns.barplot(y = cat_per_df['Genre']

                ,x = cat_per_df['High']

                ,dodge = False

               ,color = '#98DAA7')

    sns.barplot(y = cat_per_df['Genre']

                ,x = cat_per_df['Mod_Low']

                ,dodge = False

               ,color = '#F3ABA8')

    # sns.lineplot(y = cat_per_df['Genre']

    #             ,x = cat_per_df['Total'])

    legend_elements = [Patch(facecolor='#C0EBE9',

                             label='Moderately High'),

                       Patch(facecolor='#F3F3AF',

                             label='Medium'),

                       Patch(facecolor='#98DAA7',

                             label='High'),

                      Patch(facecolor='#F3ABA8',

                             label='Moderately Low')]

    for i,cat in enumerate(list(cat_per_df['Genre'])):

        plt.text(100,i,

                 cat_counts_df[cat_counts_df['Genre'] == cat]['Total'].values[0],

                 ha = 'left')

    plt.legend(handles=legend_elements,loc='lower center', bbox_to_anchor=(0.5, z),

              ncol=5, fancybox=True)

    plt.show()
# Generating the categorized dataset for Genre
genre_count_df = cat_count_df(netflix_df,'Genres',rating_dict)

genre_per_df = cat_per_df(genre_count_df)
# Plotting

sns.set_style("whitegrid", {'axes.grid' : False})
count_plot_cat(genre_count_df,25,10)

perc_plot_cat(genre_per_df,genre_count_df,25,10)
lang_count_df = cat_count_df(netflix_df,'Language',rating_dict)

lang_count_df = lang_count_df.drop(lang_count_df[lang_count_df['Total'] == 0].index)
lang_per_df = cat_per_df(lang_count_df)
# count_plot_cat(lang_count_df,20,20)

perc_plot_cat(lang_per_df,lang_count_df,25,30,-0.05)