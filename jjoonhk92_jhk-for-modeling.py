# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import ast

import json

from collections import Counter

import itertools

from itertools import zip_longest



import pickle



%matplotlib inline

%precision 3

pd.set_option('precision', 3)



import warnings

warnings.filterwarnings('ignore')
#データを読み取る

train = pd.read_csv('../input/tmdb-box-office-prediction/train.csv')

test = pd.read_csv('../input/tmdb-box-office-prediction/test.csv')
df = pd.concat([train, test])
print(train.shape,test.shape)

train.columns
#columnsを確認し、除外する変数をdrop

df.drop(columns=['overview','status','imdb_id','poster_path','original_title'], inplace = True)

print(df.columns)
df.columns
#変数の内容を全て確認

#for i, e in enumerate(df['cast'][:1]):

#    print(i, e)
#columnsの辞書化のために変数を定義

dict_columns = ['belongs_to_collection', 'genres', 'production_companies','spoken_languages',

                'production_countries', 'Keywords', 'cast', 'crew']
#JSONの形になっている変数を辞書化させ、対応できるような定義を作る

def text_to_dict(df):

    for column in dict_columns:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df



df = text_to_dict(df)
#映画の中にどれだけの人がキャストされたか表示

print('Number of casted persons in films')

df['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head()
#castの中にある俳優の名前をリスト化させる

list_of_cast_names = list(df['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df['num_cast'] = df['cast'].apply(lambda x: len(x) if x != {} else 0)

df['all_cast'] = df['cast'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')





top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(30)]

for g in top_cast_names:

    df['cast_name_' + g] = df['all_cast'].apply(lambda x: 1 if g in x else 0)
#Counter([i for j in list_of_cast_names for i in j]).most_common(30)
list_of_cast_genders = list(df['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)



df['genders_0_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_cast'] = df['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))    



#df = df.drop(['cast'], axis=1)

df['cast_gen0_ratio'] = df['genders_0_cast'].sum()/df['num_cast'].sum()

df['cast_gen1_ratio'] = df['genders_1_cast'].sum()/df['num_cast'].sum()

df['cast_gen2_ratio'] = df['genders_2_cast'].sum()/df['num_cast'].sum()
#for i, e in enumerate(df['crew'][:1]):

#    print(i, e)
#print('Number of casted persons in films')

#df['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(20)
#crewのname

list_of_crew_names = list(df['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

df['num_crew'] = df['crew'].apply(lambda x: len(x) if x != {} else 0)

df['all_crew'] = df['crew'].apply(lambda x: ','.join(sorted([i['name'] for i in x])) if x != {} else '')

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]

for g in top_crew_names:

    df['crew_name_' + g] = df['all_crew'].apply(lambda x: 1 if g in x else 0)

top_crew_names
Counter([i for j in list_of_crew_names for i in j]).most_common(15)
list_of_crew_department = list(df['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)

df['all_department'] = df['crew'].apply(lambda x: '　'.join(sorted([i['department']for i in x])) if x != {} else '')

top_crew_department = [m[0] for m in Counter(i for j in list_of_crew_department for i in j).most_common(12)]

for g in top_crew_department:

    df['crew_department_' + g] = df['crew'].apply(lambda x: sum([1 for i in x if i['department'] == g]))
list_of_crew_job = list(df['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)

top_crew_job = [m[0] for m in Counter(i for j in list_of_crew_job for i in j).most_common(10)]

for g in top_crew_job:

    df['crew_job_' + g] = df['crew'].apply(lambda x: sum([1 for i in x if i['job'] == g]))
top_crew_job
df['genders_0_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))

df['genders_1_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))

df['genders_2_crew'] = df['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
df['crew_gen0_ratio'] = df['genders_0_crew'].sum()/df['num_crew'].sum()

df['crew_gen1_ratio'] = df['genders_1_crew'].sum()/df['num_crew'].sum()

df['crew_gen2_ratio'] = df['genders_2_crew'].sum()/df['num_crew'].sum()
all_crew_job = [m[0] for m in Counter([i for j in list_of_crew_job for i in j]).most_common()]

#all_crew_job
all_crew_department = [m[0] for m in Counter([i for j in list_of_crew_department for i in j]).most_common()]

all_crew_department
def select_department(list_dict, department):

    return [ dic['name'] for dic in list_dict if dic['department']==department]
for z in all_crew_department:

    df['{}_list'.format(z)] = df["crew"].apply(select_department, department=z)

    globals()[z] = [m[0] for m in Counter([i for j in df['{}_list'.format(z)] for i in j]).most_common(15)]

    for i in globals()[z]:

        df['crew_{}_{}'.format(z,i)] = df['{}_list'.format(z)].apply(lambda x: sum([1 for i in x]))
def select_job(list_dict, job):

    return [ dic["name"] for dic in list_dict if dic["job"]==job]
for z in top_crew_job:

    df['{}_list'.format(z)] = df["crew"].apply(select_job, job=z)

    globals()[z] = [m[0] for m in Counter([i for j in df['{}_list'.format(z)] for i in j]).most_common(15)]

    for i in globals()[z]:

        df['crew_{}_{}'.format(z,i)] = df['{}_list'.format(z)].apply(lambda x: sum([1 for i in x]))
df.columns
df2 = df
df2.drop(columns = ['Keywords', 'belongs_to_collection', 'budget', 'cast', 'crew', 'genres',

       'homepage', 'id', 'original_language', 'popularity','production_companies', 

    'production_countries', 'release_date','revenue', 'runtime', 

                    'spoken_languages', 'tagline', 'title'], inplace = True)

df2.columns
for del_name in all_crew_department:

    df2.drop(columns = del_name+'_list', inplace = True)

    

for del_name in top_crew_job:

    df2.drop(columns = del_name+'_list', inplace = True)
df_use_jhk = df2
with open('df_use_jhk.pkl', 'wb') as f:

      pickle.dump(df_use_jhk , f)
#for i, e in enumerate(df.columns[:]):

#    print(i, e)
'num_cast','all_cast','cast_name_Samuel L. Jackson','cast_name_Robert De Niro','cast_name_Bruce Willis',

'cast_name_Morgan Freeman','cast_name_Liam Neeson','cast_name_Willem Dafoe','cast_name_Steve Buscemi',

'cast_name_Sylvester Stallone','cast_name_Nicolas Cage','cast_name_Matt Damon','cast_name_J.K. Simmons',

'cast_name_John Goodman','cast_name_Julianne Moore','cast_name_Christopher Walken','cast_name_Robin Williams',

'cast_name_Johnny Depp','cast_name_Stanley Tucci','cast_name_Harrison Ford','cast_name_Richard Jenkins',

'cast_name_Ben Stiller','cast_name_Susan Sarandon','cast_name_Brad Pitt','cast_name_Tom Hanks',

'cast_name_Keith David','cast_name_John Leguizamo','cast_name_Woody Harrelson','cast_name_Bill Murray','cast_name_Dennis Quaid','cast_name_James Franco','cast_name_Dustin Hoffman','genders_0_cast','genders_1_cast',

'genders_2_cast','cast_gen0_ratio','cast_gen1_ratio','cast_gen2_ratio','num_crew','all_crew','crew_name_Avy Kaufman','crew_name_Steven Spielberg',

'crew_name_Robert Rodriguez','crew_name_Mary Vernieu','crew_name_Deborah Aquila','crew_name_Bob Weinstein','crew_name_Harvey Weinstein','crew_name_Hans Zimmer','crew_name_Tricia Wood','crew_name_James Newton Howard',

'crew_name_James Horner','crew_name_Luc Besson','crew_name_Francine Maisler','crew_name_Kerry Barden','crew_name_Jerry Goldsmith','all_department','crew_department_Production','crew_department_Sound',

'crew_department_Art','crew_department_Crew','crew_department_Writing','crew_department_Costume & Make-Up','crew_department_Camera','crew_department_Directing','crew_department_Editing','crew_department_Visual Effects','crew_department_Lighting','crew_department_Actors','crew_job_Producer','crew_job_Executive Producer','crew_job_Director','crew_job_Screenplay','crew_job_Editor','crew_job_Casting','crew_job_Director of Photography','crew_job_Original Music Composer','crew_job_Art Direction','crew_job_Production Design',

'genders_0_crew','genders_1_crew','genders_2_crew','crew_gen0_ratio','crew_gen1_ratio','crew_gen2_ratio',

'crew_Production_Avy Kaufman','crew_Production_Mary Vernieu','crew_Production_Deborah Aquila','crew_Production_Bob Weinstein','crew_Production_Harvey Weinstein','crew_Production_Tricia Wood','crew_Production_Francine Maisler','crew_Production_Kerry Barden','crew_Production_Billy Hopkins','crew_Production_Steven Spielberg','crew_Production_Suzanne Smith',

'crew_Production_Arnon Milchan','crew_Production_Scott Rudin','crew_Production_John Papsidera','crew_Production_Tim Bevan','crew_Sound_James Newton Howard','crew_Sound_Hans Zimmer','crew_Sound_James Horner','crew_Sound_Jerry Goldsmith','crew_Sound_John Williams',

'crew_Sound_Alan Silvestri','crew_Sound_Danny Elfman',"crew_Sound_Dan O'Connell",'crew_Sound_Mark Isham','crew_Sound_John Debney','crew_Sound_Marco Beltrami',

'crew_Sound_Kevin Kaska','crew_Sound_Christophe Beck','crew_Sound_Graeme Revell','crew_Sound_Carter Burwell','crew_Art_Helen Jarvis','crew_Art_Ray Fisher','crew_Art_Rosemary Brandenburg',

'crew_Art_Cedric Gibbons','crew_Art_Walter M. Scott','crew_Art_Nancy Haigh','crew_Art_Robert Gould','crew_Art_J. Michael Riva','crew_Art_Maher Ahmad','crew_Art_Henry Bumstead','crew_Art_Leslie A. Pope',

'crew_Art_Gene Serdena','crew_Art_Jann Engel','crew_Art_David F. Klassen','crew_Art_Cindy Carr','crew_Crew_J.J. Makaro','crew_Crew_Brian N. Bentley',

'crew_Crew_Brian Avery','crew_Crew_James Bamford','crew_Crew_Mark Edward Wright','crew_Crew_Karin Silvestri','crew_Crew_Gregory Nicotero','crew_Crew_G.A. Aguilar',

'crew_Crew_Doug Coleman','crew_Crew_Sean Button',"crew_Crew_Chris O'Connell",'crew_Crew_Tim Monich','crew_Crew_Denny Caira',

'crew_Crew_Susan Hegarty','crew_Crew_Michael Queen','crew_Writing_Luc Besson','crew_Writing_Stephen King','crew_Writing_Woodyallen','crew_Writing_John Hughes',

'crew_Writing_Ian Fleming','crew_Writing_Robert Mark Kamen','crew_Writing_Sylvester Stallone','crew_Writing_David Koepp','crew_Writing_Terry Rossio',

'crew_Writing_George Lucas','crew_Writing_Stan Lee','crew_Writing_Akiva Goldsman','crew_Writing_Brian Helgeland','crew_Writing_Ted Elliott','crew_Writing_William Goldman','crew_Costume & Make-Up_Ve Neill',

'crew_Costume & Make-Up_Bill Corso','crew_Costume & Make-Up_Colleen Atwood','crew_Costume & Make-Up_Camille Friend','crew_Costume & Make-Up_Edith Head','crew_Costume & Make-Up_Louise Frogley','crew_Costume & Make-Up_Ellen Mirojnick',

'crew_Costume & Make-Up_Mary Zophres','crew_Costume & Make-Up_Edouard F. Henriques','crew_Costume & Make-Up_Jean Ann Black','crew_Costume & Make-Up_Marlene Stewart','crew_Costume & Make-Up_Ann Roth','crew_Costume & Make-Up_Deborah La Mia Denaver',

'crew_Costume & Make-Up_Alex Rouse','crew_Costume & Make-Up_Shay Cunliffe','crew_Camera_Hans Bjerno','crew_Camera_Roger Deakins','crew_Camera_Dean Semler',

'crew_Camera_David B. Nowell','crew_Camera_Mark Irwin','crew_Camera_John Marzano','crew_Camera_Matthew F. Leonetti','crew_Camera_Dean Cundey',

'crew_Camera_Frank Masi','crew_Camera_Oliver Wood','crew_Camera_Robert Elswit','crew_Camera_Pete Romano','crew_Camera_Merrick Morton',

'crew_Camera_Robert Richardson','crew_Camera_Philippe Rousselot','crew_Directing_Steven Spielberg','crew_Directing_Clint Eastwood','crew_Directing_Woodyallen',

'crew_Directing_Ridley Scott','crew_Directing_Karen Golden','crew_Directing_Alfred Hitchcock','crew_Directing_Kerry Lyn McKissick','crew_Directing_Ron Howard','crew_Directing_Dianne Dreyer','crew_Directing_Wilma Garscadden-Gahret',

'crew_Directing_Martin Scorsese','crew_Directing_Brian De Palma','crew_Directing_Ana Maria Quintana','crew_Directing_Dug Rotstein',

'crew_Directing_Tim Burton','crew_Editing_Michael Kahn','crew_Editing_Chris Lebenzon','crew_Editing_Jim Passon',

'crew_Editing_Gary Burritt','crew_Editing_Dale E. Grahn','crew_Editing_Joel Cox','crew_Editing_Mark Goldblatt',

'crew_Editing_Conrad Buff IV','crew_Editing_John C. Stuver','crew_Editing_Pietro Scalia','crew_Editing_Paul Hirsch',

'crew_Editing_Don Zimmerman','crew_Editing_Robert Troy','crew_Editing_Steven Rosenblum','crew_Editing_Dennis McNeill',

'crew_Visual Effects_Dottie Starling','crew_Visual Effects_Phil Tippett','crew_Visual Effects_James Baker','crew_Visual Effects_Hugo Dominguez',

'crew_Visual Effects_Larry White','crew_Visual Effects_Ray McIntyre Jr.','crew_Visual Effects_James Baxter','crew_Visual Effects_Aaron Williams',"crew_Visual Effects_Julie D'Antoni",'crew_Visual Effects_Frank Thomas','crew_Visual Effects_Milt Kahl','crew_Visual Effects_Peter Chiang','crew_Visual Effects_Chuck Duke','crew_Visual Effects_Dave Kupczyk','crew_Visual Effects_Craig Barron','crew_Lighting_Justin Hammond','crew_Lighting_Howard R. Campbell',

'crew_Lighting_Arun Ram-Mohan','crew_Lighting_Chuck Finch','crew_Lighting_Russell Engels','crew_Lighting_Frank Dorowsky',

'crew_Lighting_Bob E. Krattiger','crew_Lighting_Ian Kincaid','crew_Lighting_Thomas Neivelt','crew_Lighting_Dietmar Haupt','crew_Lighting_James J. Gilson',

'crew_Lighting_Dan Cornwall','crew_Lighting_Andy Ryan','crew_Lighting_Lee Walters','crew_Lighting_Jay Kemp','crew_Actors_Francois Grobbelaar',

"crew_Actors_Mick 'Stuntie' Milligan",'crew_Actors_Sol Gorss','crew_Actors_Mark De Alessandro','crew_Actors_Leigh Walsh',

'crew_Producer_Joel Silver','crew_Producer_Brian Grazer','crew_Producer_Scott Rudin','crew_Producer_Neal H. Moritz',

'crew_Producer_Tim Bevan','crew_Producer_Eric Fellner','crew_Producer_Jerry Bruckheimer','crew_Producer_Arnon Milchan',

'crew_Producer_Gary Lucchesi','crew_Producer_John Davis','crew_Producer_Jason Blum','crew_Producer_Tom Rosenberg','crew_Producer_Kathleen Kennedy',

'crew_Producer_Luc Besson','crew_Producer_Steven Spielberg','crew_Executive Producer_Bob Weinstein','crew_Executive Producer_Harvey Weinstein','crew_Executive Producer_Bruce Berman',

'crew_Executive Producer_Steven Spielberg','crew_Executive Producer_Toby Emmerich','crew_Executive Producer_Stan Lee','crew_Executive Producer_Ryan Kavanaugh','crew_Executive Producer_Ben Waisbren','crew_Executive Producer_Michael Paseornek','crew_Executive Producer_Thomas Tull','crew_Executive Producer_Arnon Milchan','crew_Executive Producer_Nathan Kahane','crew_Executive Producer_John Lasseter','crew_Executive Producer_Tessa Ross',

'crew_Executive Producer_Gary Barber','crew_Director_Steven Spielberg','crew_Director_Clint Eastwood','crew_Director_Woodyallen','crew_Director_Ridley Scott',

'crew_Director_Alfred Hitchcock','crew_Director_Ron Howard','crew_Director_Brian De Palma','crew_Director_Martin Scorsese','crew_Director_Tim Burton',

'crew_Director_Blake Edwards','crew_Director_Joel Schumacher','crew_Director_Oliver Stone','crew_Director_Robert Zemeckis','crew_Director_Steven Soderbergh',

'crew_Director_Wes Craven','crew_Screenplay_Sylvester Stallone','crew_Screenplay_Luc Besson','crew_Screenplay_John Hughes','crew_Screenplay_Akiva Goldsman','crew_Screenplay_David Koepp','crew_Screenplay_William Goldman','crew_Screenplay_Robert Mark Kamen','crew_Screenplay_Oliver Stone',

'crew_Screenplay_Woodyallen','crew_Screenplay_Richard Maibaum','crew_Screenplay_John Logan','crew_Screenplay_Terry Rossio','crew_Screenplay_Harold Ramis',

'crew_Screenplay_Brian Helgeland','crew_Screenplay_Ted Elliott','crew_Editor_Michael Kahn','crew_Editor_Chris Lebenzon','crew_Editor_Joel Cox',

'crew_Editor_Mark Goldblatt','crew_Editor_Conrad Buff IV','crew_Editor_Pietro Scalia','crew_Editor_Paul Hirsch','crew_Editor_Don Zimmerman',

'crew_Editor_Christian Wagner','crew_Editor_Anne V. Coates','crew_Editor_William Goldenberg','crew_Editor_Michael Tronick','crew_Editor_Daniel P. Hanley',

'crew_Editor_Paul Rubell','crew_Editor_Stephen Mirrione','crew_Casting_Avy Kaufman','crew_Casting_Mary Vernieu','crew_Casting_Deborah Aquila',

'crew_Casting_Tricia Wood','crew_Casting_Kerry Barden','crew_Casting_Francine Maisler','crew_Casting_Billy Hopkins','crew_Casting_Suzanne Smith',

'crew_Casting_John Papsidera','crew_Casting_Denise Chamian','crew_Casting_Jane Jenkins','crew_Casting_Janet Hirshenson','crew_Casting_Mike Fenton',

'crew_Casting_Mindy Marin','crew_Casting_Sarah Finn','crew_Director of Photography_Dean Semler','crew_Director of Photography_Roger Deakins','crew_Director of Photography_Mark Irwin',

'crew_Director of Photography_Matthew F. Leonetti','crew_Director of Photography_Dean Cundey','crew_Director of Photography_Oliver Wood','crew_Director of Photography_Robert Elswit','crew_Director of Photography_Robert Richardson',

'crew_Director of Photography_Philippe Rousselot','crew_Director of Photography_Dante Spinotti','crew_Director of Photography_Julio Macat','crew_Director of Photography_Dariusz Wolski','crew_Director of Photography_Don Burgess',

'crew_Director of Photography_Janusz Kami≈Ñski','crew_Director of Photography_Peter Deming','crew_Original Music Composer_James Newton Howard','crew_Original Music Composer_James Horner','crew_Original Music Composer_Hans Zimmer',

'crew_Original Music Composer_Jerry Goldsmith','crew_Original Music Composer_John Williams','crew_Original Music Composer_Danny Elfman','crew_Original Music Composer_Christophe Beck','crew_Original Music Composer_Alan Silvestri',

'crew_Original Music Composer_John Powell','crew_Original Music Composer_Marco Beltrami','crew_Original Music Composer_Howard Shore','crew_Original Music Composer_Graeme Revell','crew_Original Music Composer_John Debney',

'crew_Original Music Composer_Carter Burwell','crew_Original Music Composer_Mark Isham','crew_Art Direction_Cedric Gibbons','crew_Art Direction_Hal Pereira','crew_Art Direction_Helen Jarvis',

'crew_Art Direction_Lyle R. Wheeler','crew_Art Direction_David Lazan','crew_Art Direction_Andrew Max Cahn','crew_Art Direction_Jack Martin Smith','crew_Art Direction_Robert Cowper',

'crew_Art Direction_Stuart Rose','crew_Art Direction_David F. Klassen','crew_Art Direction_Dan Webster','crew_Art Direction_Steven Lawrence','crew_Art Direction_Jesse Rosenthal',

'crew_Art Direction_Richard L. Johnson','crew_Art Direction_Kevin Constant','crew_Production Design_J. Michael Riva','crew_Production Design_Jon Hutman','crew_Production Design_Carol Spier',

'crew_Production Design_Ida Random','crew_Production Design_Dennis Gassner','crew_Production Design_Perry Andelin Blake','crew_Production Design_David Gropman',

'crew_Production Design_Mark Friedberg','crew_Production Design_Rick Carter','crew_Production Design_Stuart Craig','crew_Production Design_Jim Clay',

'crew_Production Design_Kristi Zea','crew_Production Design_David Wasco','crew_Production Design_Wynn Thomas','crew_Production Design_Dante Ferretti'