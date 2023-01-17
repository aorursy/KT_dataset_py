# working with the paths
import os
import glob
import sys
# data manipulation
import pandas as pd
import numpy as np
import zipfile

# stats
from scipy.stats import kurtosis, skew
# plotting / visualization
import plotly.offline as pyoff
import plotly.graph_objs as go
from plotly.offline import iplot , plot, init_notebook_mode
init_notebook_mode()
import matplotlib.pyplot as plt
%matplotlib inline

for i in glob.glob("*.csv"):
    print (i)
powers = pd.read_csv("../input/super_hero_powers.csv",na_values=["-"])
hero_info = pd.read_csv("../input/heroes_information.csv",na_values=["-"])
powers.info()
def print_num_cols(df, name):
    print ("The {} dataset has {} rows and {} columns".format(name,df.shape[0],df.shape[1]))
print_num_cols(powers,"powers")
print_num_cols(hero_info,"hero_info")
# print ("The column names of the train dataset are \n {}".format(powers.columns.values))
def data_types(df):
    print (df.dtypes.value_counts())
data_types(powers)
data_types(hero_info)
def extract_type_cols(df):
    """This functions extracts numeric, categorical , datetime and boolean column types.
    Returns 4 lists with respective column types"""
    num_cols_list = [i for i in df.columns if df[i].dtype in ['int64','float64']]
    cat_cols_list = [i for i in df.columns if df[i].dtype in ['object']]
    date_cols_list = [i for i in df.columns if df[i].dtype in ['datetime64[ns]']]
    bool_cols_list = [i for i in df.columns if df[i].dtype in ['bool']]
    print ("Numeric Columns:", len(num_cols_list))
    print ("Categorical/Character Columns:", len(cat_cols_list))
    print ("Date Columns:",len(date_cols_list))
    print ("Boolean Columns:",len(bool_cols_list))
    return(num_cols_list,cat_cols_list,date_cols_list,bool_cols_list)
powers_num_cols_list,powers_cat_cols_list,powers_date_cols_list,powers_bool_cols_list = extract_type_cols(powers)
extract_type_cols(hero_info)
num_cols_list,cat_cols_list,date_cols_list,bool_cols_list = extract_type_cols(hero_info)
def plot_bar(x,y,title,color):
    trace =go.Bar(
            x=x,
            y=y,text = y,textposition = 'auto',
            marker=dict(
                color=color,
                line=dict(
                    color='black',
                    width=1.5),
            ),
            opacity=0.9
    )
    data = [trace]
    layout = go.Layout(title =title)
    fig = go.Figure(data= data,layout = layout)
    iplot(fig)
colors = ['rgb(237,29,36)',
         'rgb(170,20,40)',
         'rgb(170,5,5)',
         'rgb(185,125,16)',
         'rgb(103,199,235)',
         'rgb(251,202,3)',
         'rgb(3,173,233)',
         'rgb(254,88,22)']
for i in enumerate(cat_cols_list):
    if i[1] != 'name':
        x = hero_info[i[1]].value_counts().index
        y = hero_info[i[1]].value_counts()
        plot_bar(x,y,title = "Distribution of {} column".format(i[1]),color = colors[i[0]])
for i in hero_info[hero_info.Race=='Kryptonian'].name:
    print (i)
powers_counts_df = powers.iloc[:,1:].apply(lambda x : np.sum(x),axis = 0).sort_values(ascending=False).to_frame('Counts')
powers_counts_df['Percentage'] = powers_counts_df.Counts/len(hero_info.name.unique())*100
plot_bar(x = powers_counts_df.head(5).index,
         y = powers_counts_df.head(5).Counts,
         title='Top 5 most common powers',
        color = 'rgb(250, 223, 127)')
rare_half_percent = powers_counts_df[powers_counts_df.Percentage<=0.5]
rare_half_percent = rare_half_percent.sort_values('Percentage',ascending=True)
plot_bar(x = rare_half_percent.index,
         y = rare_half_percent.Counts,
         title='Rare powers <br> only half a percent of the super heros have these powers',
        color = 'rgb(237,29,36)')
def plot_bar_alignment_interactive():
    """This funtion asks the use for the publisher name and plots the alignemnt of super heros."""
    publisher_name = input("Enter the name of the publisher for which you want to see the Alignment Distribution: ")
    if publisher_name not in hero_info.Publisher.tolist():
        print ("The publisher name you mentioned is not available in the hero_info tablem, please check the spelling or the name and try again.")
        plot_bar_alignment_interactive()
    else:
        alignment_df = hero_info.Alignment[hero_info.Publisher == publisher_name].value_counts().to_frame()
        cols_dict = {'good' : 'rgb (23, 185, 120)',
        'bad' : 'rgb (181, 0, 12)',
        'neutral' : 'rgb(169, 169, 169)'}
        color = [cols_dict[a] for a in alignment_df.index]
        plot_bar(x = alignment_df.index,
                y= alignment_df.Alignment,title = "{} Super Heroes Alignment".format(publisher_name),
                color=color)
# plot_bar_alignment_interactive()
def plot_bar_alignment(publisher_name):
    """This funtion take the name of the publisher name and plots the alignemnt of super heros."""
    if publisher_name not in hero_info.Publisher.tolist():
        print ("The publisher name you mentioned is not available in the hero_info tablem, please check the spelling or the name and try again.")
    else:
        alignment_df = hero_info.Alignment[hero_info.Publisher == str(publisher_name)].value_counts().to_frame()
        cols_dict = {'good' : 'rgb (23, 185, 120)',
        'bad' : 'rgb (181, 0, 12)',
        'neutral' : 'rgb(169, 169, 169)'}
        color = [cols_dict[a] for a in alignment_df.index]
        plot_bar(x = alignment_df.index,
                y= alignment_df.Alignment,title = "{} Super Heroes Alignment".format(publisher_name),
                color=color)
plot_bar_alignment("DC Comics")
plot_bar_alignment("Marvel Comics")
merged_data = hero_info.merge(powers,left_on = 'name',right_on='hero_names')
merged_data['Powers_Count'] = merged_data.loc[:,powers_bool_cols_list].apply(lambda x : np.sum(x.dropna()),axis = 1)
merged_data['Powers_Percent'] = (merged_data['Powers_Count']/len(powers_bool_cols_list))*100
villain = merged_data[merged_data.Alignment=='bad']
heros = merged_data[merged_data.Alignment=='good']
t = villain[(villain.Powers_Percent==np.max(villain.Powers_Percent))]
print("The most powerful villain is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
t = villain[(villain.Weight==np.max(villain.Weight))]
print("The heaviest villain is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
t = villain[(villain.Height==np.max(villain.Height))]
print("The tallest villain is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
t = heros[(heros.Powers_Percent==np.max(heros.Powers_Percent))]
print("The most powerful hero is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
human_heros = heros[heros.Race =='Human']
female_human_heros = human_heros[human_heros.Gender == 'Female']
male_human_heros = human_heros[human_heros.Gender == 'Male']
t = human_heros[(human_heros.Powers_Percent==np.max(human_heros.Powers_Percent))]
print("The most powerful human hero is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
female_heros = merged_data[merged_data.Gender =='Female']
t = female_heros[(female_heros.Powers_Percent==np.max(female_heros.Powers_Percent))]
print("The most powerful female hero is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
gods = merged_data[merged_data.Race == 'God / Eternal']
t = gods[(gods.Powers_Percent==np.max(gods.Powers_Percent))]
print("The most powerful god is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
god_villian  = gods [gods.Alignment =='bad']
t = god_villian[(god_villian.Powers_Percent==np.max(god_villian.Powers_Percent))]
print("The most powerful villain and a god is {} and is from {}.".format(t.name.values[0],t.Publisher.values[0]))
human_mutant_df = merged_data[merged_data.Race.isin(['Human','Mutant'])]

human_mutant_df_alignment =human_mutant_df.groupby(['Race','Alignment']).size().reset_index()
human_mutant_df_alignment.columns = ['Race','Alignment','Counts']

good = go.Bar(x= human_mutant_df_alignment.Race[human_mutant_df_alignment.Alignment=='good'],
              y = human_mutant_df_alignment.Counts[human_mutant_df_alignment.Alignment=='good'],
              name = 'Heros')
bad = go.Bar(x= human_mutant_df_alignment.Race[human_mutant_df_alignment.Alignment=='bad'],
              y = human_mutant_df_alignment.Counts[human_mutant_df_alignment.Alignment=='bad'],name= 'Villains')
layout = go.Layout(title = "Heros and Villains distribution in Humans and Mutants")
data = [good,bad]
fig = go.Figure(data= data,layout = layout)
iplot(fig)