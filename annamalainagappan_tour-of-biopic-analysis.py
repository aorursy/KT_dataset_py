import zipfile



Dataset = "biopics"



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:

    z.extractall(".")
from subprocess import check_output

print(check_output(["ls", "biopics"]).decode("utf8"))
import pandas as pd

# There's only one file above...we'll select it.

df = pd.read_csv(Dataset+"/biopics.csv", encoding='latin-1')

print(df.head())
print(len(df))
# Safely dropping rows with movie title and year release

df = df.drop_duplicates(subset=['title', 'year_release'])

print(len(df))



# Replace numpy nan in missing box office records

import numpy as np

df['box_office'] = df['box_office'].replace("-", np.NaN)
bins = [1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

group_names = ['1910', '1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']

df['decade'] = pd.cut(df['year_release'], bins, labels=group_names)
print (df[['title', 'year_release', 'decade']].head())
def data_agg(data, grp_col, agg_col, sorting=0):

    df_agg = data.groupby(grp_col, as_index=False).agg({agg_col: len})

    df_agg = df_agg.rename(columns = {agg_col: 'count_m'})

    if sorting:

        df_agg = df_agg.sort_values('count_m', ascending=True)

    return df_agg.reset_index(drop=True)
df_grp_decades = data_agg(df, 'decade', 'site')

print(df_grp_decades)
%matplotlib inline

from pylab import *

plot(list(df_grp_decades['decade']), list(df_grp_decades['count_m']))

xlabel('Year in Decades (s)')

ylabel('No. of Movies')

title('No. of Movies in Decades')

grid(True)

show()
df_grp_cntry = data_agg(df, 'country', 'site', 1)

print(df_grp_cntry)
import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib

import matplotlib.pyplot as plt



SMALL_SIZE = 8

matplotlib.rc('font', size=SMALL_SIZE)

matplotlib.rc('axes', titlesize=SMALL_SIZE)
y_pos = np.arange(len(df_grp_cntry['country']))

plt.bar(y_pos, df_grp_cntry['count_m'], align='center', alpha=0.5, width=0.4)

plt.xticks(y_pos, df_grp_cntry['country'])

plt.ylabel('No of Biopics')

plt.title('No of Biopics released Country Wise Split');
df_grp_sub = data_agg(df, 'type_of_subject', 'site', 1)

print(df_grp_sub.head(5))
y_pos = np.arange(len(df_grp_sub['type_of_subject']))

plt.barh(y_pos, df_grp_sub['count_m'], align='center', alpha=0.5)

plt.yticks(y_pos, df_grp_sub['type_of_subject'])

plt.ylabel('Biopics Subject')

plt.title('No of Biopics released Subject Wise Split');
df_grp_gender = data_agg(df, 'subject_sex', 'site', 1)

print(df_grp_gender)
y_pos = np.arange(len(df_grp_gender['subject_sex']))

plt.bar(y_pos, df_grp_gender['count_m'], align='center', alpha=0.5, width=0.4)

plt.xticks(y_pos, df_grp_gender['subject_sex'])

plt.ylabel('No of Biopics')

plt.title('No of Biopics released Gender Wise Split');
df_grp_subject_no = data_agg(df, 'number_of_subjects', 'site', 0)

print(df_grp_subject_no)
labels = df_grp_subject_no['number_of_subjects']

sizes = df_grp_subject_no['count_m']

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']



patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
df_grp_actor = data_agg(df, 'lead_actor_actress', 'site', 0)

df_grp_actor = df_grp_actor.sort_values('count_m', ascending=False).reset_index(drop=True)

print(df_grp_actor.head(10))
df_actor_movies = df[df['lead_actor_actress'] == 'Leonardo DiCaprio']

print(df_actor_movies[['title', 'year_release', 'box_office']].sort_values('year_release', ascending=True))
df_grp_director = data_agg(df, 'director', 'site', 0)

df_grp_director = df_grp_director.sort_values('count_m', ascending=False).reset_index(drop=True)

print(df_grp_director.head(10))
df_actor_director = df[df['director'] == 'Michael Curtiz']

print(df_actor_director[['title', 'year_release', 'box_office']].sort_values('year_release', ascending=True))

# Box Office records are not present
print(df[['title', 'year_release', 'box_office']].sort_values('box_office', ascending=False).head(20))