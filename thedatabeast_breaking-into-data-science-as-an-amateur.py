!pip install pywaffle

!pip install squarify

!pip install dash
# Loading necessary libraries

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from pywaffle import Waffle

import squarify



# user-defined utility scripts

import surveycleaner
# loading data

df = pd.read_csv("../input/cleaned-mcr-kaggle-survey-2019/clean_multiple_choice_responses.csv")



# first column is an extra; remove it

df = df.drop(["Unnamed: 0"], axis=1)

df.head()
""" Utility Functions """



def topN(feature, dataframe, colnames=["Index", "Value"], omit_cols = [], all_values=False, N=10):

    """

    Returns Top N values in a feature as a dictionary

    """

    dataframe = dataframe[~dataframe[feature].isin(omit_cols)]

    # Prepare data subset

    if(all_values==False):

        subset = dataframe[feature].value_counts().sort_values(ascending=False).head(N)

    else :

        subset = dataframe[feature].value_counts().sort_values(ascending=False)

    index = list(subset.index)

    vals = list(subset.values)

    topN = pd.DataFrame({colnames[0]:index, colnames[1]:vals})

    

    return topN



def sum_vals(dictionary):

    """

    Returns sum of all values in a dictionary

    """

    temp = 0

    for val in dictionary.values():

        temp = temp + val

    return temp



def norm_dict_vals(dictionary):

    """

    Returns a dictionary with normalized values (out of 100)

    """

    norm_keys = list(dictionary.keys())

    norm_vals = []

    sum_dict_vals = sum_vals(dictionary)

    for val in dictionary.values():

        norm_vals.append(round((val/sum_dict_vals * 100)))

    

    return (

        dict(

            zip(

                norm_keys,norm_vals

            )

        )

    )



def norm_dict_vals_2(dictionary):

    """

    Returns a dictionary with normalized values (out of 100)

    """

    norm_keys = list(dictionary.keys())

    norm_vals = []

    sum_dict_vals = sum_vals(dictionary)

    for val in dictionary.values():

        norm_vals.append(round((val/sum_dict_vals * 100 * 3)))

    

    return (

        dict(

            zip(

                norm_keys,norm_vals

            )

        )

    )

    

def bar_viz(dataframe, xvar, yvar, another_order=None, title="No Title Specified", bar_cmap="Set3", edge_color="black", title_color="black", figure_size=(6,6)):

    """

    Displays a simple bar plot

    """

    fig,ax =  plt.subplots(figsize=figure_size)

    color_list = matplotlib.cm.get_cmap(bar_cmap, (len(dataframe[xvar])))

    

    if(another_order==None):

        order = yvar

    else:

        order = another_order

    

    ax.barh(

        y=xvar,

        data=dataframe.sort_values(by=order, ascending=True),

        width=yvar,

        color=color_list(np.linspace(0,1,(len(dataframe[xvar])))),

        edgecolor=edge_color

    )

    ax.set_title(title, fontsize=15, color=title_color)

    plt.show()

    

def aggregate_vals(dataframe, feature):

    aggr_list = []

    for i in range(ds.shape[0]):

        aggr_list = aggr_list + ds[feature].iloc[i].replace("\',",";").replace('[','').replace(']','').replace('\'','').split(';')

    

    return aggr_list



def counting_dict(our_list):

    cnt_dict = {}

    for ele in our_list:

        ele=ele.strip() # removing stray spaces

        ele = main_cat(ele)

        if(not(ele in list(cnt_dict.keys()))):

            cnt_dict[ele] = 1

        else:

            cnt_dict[ele] += 1

            

    return cnt_dict



def main_cat(option):

    """

    returns the main category i.e

    before braces

    """

    return (option.split('(')[0].strip())
# What kind of data roles do the correspondents work in?



data_roles = topN(

    "Select the title most similar to your current role (or most recent title if retired): - Selected Choice",

    df,

    colnames = ["Data Role", "Respondents"],

    omit_cols=["Other"],

    all_values = True,

)



data_roles_dict = norm_dict_vals(

    dict(

        zip(

            list(data_roles["Data Role"]),

            list(data_roles["Respondents"])

        )

    )

)



fig = plt.figure(

    FigureClass=Waffle, 

    columns=20,

    values=data_roles_dict, 

    cmap_name="tab20",

    interval_ratio_x=0.2,

    interval_ratio_y=0.2,

    title={

        'label': 'Percentage of Respondents as per their current Data Role\n',

        'loc': 'left',

        'fontsize': 20

          },

    labels=["{0} ({1}%)".format(k, v) for k, v in data_roles_dict.items()],

    #icons='user-tie',

    #icon_legend=True,

    legend={

        'loc': 'upper left',

        'bbox_to_anchor': (1, 1),

        'ncol': 2,

        'fontsize': 'x-large',

        'framealpha': 0},

    starting_location='NW',

    figsize=(20,6),

)

plt.show()



# Subsetting "Data Scientists" and all the code to create the visualizations related to Data Scientists



ds = df[df['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=="Data Scientist"]



# Top 10 countries in terms of the number of data scientists responding to the survey

ds_by_country = topN(

    "In which country do you currently reside?",

    ds,

    colnames = ["Country", "DS_Respondents"],

    omit_cols=["Other"],

    all_values = False,

)

ds_by_country = ds_by_country.replace("United Kingdom of Great Britain and Northern Ireland", "UK and Northern Island") # a slight cleanup



# Size of Companies where data scientists work

ds_comp_size = topN(

    'What is the size of the company where you are employed?',

    ds,

    colnames = ["Company_Size", "DS_Respondents"],

    omit_cols=["Other"],

    all_values=True,

)



# Size of Data Science Teams

ds_team_size = topN(

    'Approximately how many individuals are responsible for data science workloads at your place of business?',

    ds,

    colnames = ["Team_Size", "DS_Respondents"],

    omit_cols=["Other"],

    all_values=True,

)



# Salary for Data Scientists

ds_salary = topN(

    'What is your current yearly compensation (approximate $USD)?',

    ds,

    colnames=["Salary", "DS_Respondents"],

    omit_cols=["Other"],

    all_values=True

)

# ordering it up a bit

def add_feature(x):

    return int(x.replace(',','').replace('$','').replace('>','').strip().split('-')[0])

ds_salary["to_order"] = ds_salary["Salary"].apply(add_feature)

ds_salary = ds_salary.sort_values(by="to_order", ascending=True)



# Programming Languages for an anspiring data scientist

ds_prog_lang = topN(

    "What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice",

    ds,

    colnames=["Programming_Languages", "DS_Respondents"],

    omit_cols=["None"],

    all_values=False,

    N=5

)



# Length of writing code

ds_length_code_write = topN(

    'How long have you been writing code to analyze data (at work or at school)?',

    ds,

    colnames=["Years_Writing_Code", "DS_Respondents"],

    all_values=True

)

# ordering it up a bit

def add_feature(x):

    return float(x.replace('years','').replace('< 1','0.5').replace('+','').replace('I have never written code','0').strip().split('-')[0])

ds_length_code_write["to_order"] = ds_length_code_write["Years_Writing_Code"].apply(add_feature)

ds_length_code_write = ds_length_code_write.sort_values(by="to_order", ascending=True)



# Length of using ML methods

ds_ml_use = topN(

    'For how many years have you used machine learning methods?',

    ds,

    colnames=["Years_Using_ML", "DS_Respondents"],

    all_values=True

)

# ordering it up a bit

def add_feature(x):

    return float(x.replace('years','').replace('< 1','0.5').replace('+','').replace('I have never written code','0').strip().split('-')[0])

ds_ml_use["to_order"] = ds_ml_use["Years_Using_ML"].apply(add_feature)

ds_ml_use = ds_ml_use.sort_values(by="to_order", ascending=True)
# Where are most of the responders employed as data scientists from?

bar_viz(ds_by_country, "Country", "DS_Respondents", title="Top 10 Countries acc. to Data Scientist responses\n", figure_size=(6,4))
# Sizes of companies where Data Scientsts work

bar_viz(ds_comp_size, "Company_Size", "DS_Respondents", title="Company Sizes where Data Scientists work\n", bar_cmap="Set2")
# Sizes of teams responsible for data science workloads

bar_viz(ds_team_size, "Team_Size", "DS_Respondents", title="Team Sizes\n", bar_cmap="Pastel2")
# Salary for Data Scientists

bar_viz(ds_salary, "Salary", "DS_Respondents", another_order="to_order", title="Salary Distribution for Data Scientists\n",

        bar_cmap="viridis", figure_size=(10,8))
# Programming Languages for Aspiring Data Scientists

bar_viz(ds_prog_lang, "Programming_Languages", "DS_Respondents", title="Top 5 Programming Languages for Aspiring Data Scientists\n",

        bar_cmap="Blues", figure_size=(10,4))
# How long have the data scientists been writing code

bar_viz(ds_length_code_write, "Years_Writing_Code", "DS_Respondents", title="How long have the data scientists in the survey been writing code\n",

        another_order="to_order", bar_cmap="RdBu_r", figure_size=(10,4))
# How long have the data scientists been using ML methods

bar_viz(ds_ml_use, "Years_Using_ML", "DS_Respondents", title="How long have the data scientists used ML methods\n",

        another_order="to_order", bar_cmap="RdBu_r", figure_size=(10,4))
# reset the index

ds = ds.reset_index(drop=True)
# Data Scientist's Activities at Work



feature = 'Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice'

            

# dictionary of vals

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None of these activities are an important part of my role at work']



fig = plt.figure(

    FigureClass=Waffle, 

    columns=20,

    values=norm_dict_vals(cnt_dict), 

    cmap_name="Dark2",

    interval_ratio_x=0.2,

    interval_ratio_y=0.2,

    title={

        'label': 'Percentage of Main Activities for Data Scientists at Work (Highest to Lowest) \n',

        'loc': 'left',

        'fontsize': 20

          },

    labels=["{0} ({1}%)".format(k, v) for k, v in norm_dict_vals(cnt_dict).items()],

    legend={

        'loc': 'upper left',

        'bbox_to_anchor': (0, -0.4),

        'ncol': 2,

        'fontsize': 'x-large',

        'framealpha': 0},

    starting_location='NW',

    figsize=(20,6),

)

plt.show()
# Most favoured media sources reporting about data science



feature = 'Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice'

# dictionary of vals

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



media_src_df = pd.DataFrame({"Media_Source":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})
# Where do most data scientists gain their data science related topics?

bar_viz(media_src_df, "Media_Source", "DS_Respondents", title="Most favoured data science reporting Media Sources\n", figure_size=(6,4))
# Primary Tool used by Data scientists



feature = 'What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

good_options = ['Local development environments', 'Cloud-based data software & APIs', 'Basic statistical software', 'Advanced statistical software',

               'Business intelligence software']

cnt_dict = {new_key: cnt_dict[new_key] for new_key in good_options}



fig = plt.figure(

    FigureClass=Waffle, 

    columns=20,

    values=norm_dict_vals(cnt_dict), 

    cmap_name="Set2",

    interval_ratio_x=0.2,

    interval_ratio_y=0.2,

    title={

        'label': 'Primary Tools used by Data Scientists (Highest to Lowest)\n',

        'loc': 'left',

        'fontsize': 20

          },

    labels=["{0} ({1}%)".format(k, v) for k, v in norm_dict_vals(cnt_dict).items()],

    legend={

        'loc': 'upper left',

        'bbox_to_anchor': (1, 1),

        'ncol': 1,

        'fontsize': 'x-large',

        'framealpha': 0},

    starting_location='NW',

    figsize=(20,6),

)

plt.show()



tools_df = pd.DataFrame({"Primary_Tool":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})
# IDEs



feature = 'Which of the following integrated development environments (IDE\'s) do you use on a regular basis?  (Select all that apply) - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



ide_df = pd.DataFrame({"IDE":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})
# What are the main IDE tools used by data scientists?

bar_viz(ide_df, "IDE", "DS_Respondents", title="Most Common IDEs used by Data Scientists\n", figure_size=(6,4))
# Hosted Notebook Products



feature = 'Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



host_nb_df = pd.DataFrame({"Hosted_Notebooks":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})



bar_viz(host_nb_df, "Hosted_Notebooks", "DS_Respondents", title="Most Common Hosted Notebooks\n", figure_size=(6,4))
# Most used prog languages



feature = 'What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



host_nb_df = pd.DataFrame({"Prog_Lang":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})



bar_viz(host_nb_df, "Prog_Lang", "DS_Respondents", title="Most Used Programming Languages\n", figure_size=(6,4))
# Most used data viz tools



feature = 'What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



host_nb_df = pd.DataFrame({"Dat_Viz":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})



bar_viz(host_nb_df, "Dat_Viz", "DS_Respondents", title="Most Used Data Viz Tools\n", figure_size=(6,4))
# Most commonly used ML Algos



feature = 'Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



host_nb_df = pd.DataFrame({"ML_algo":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})



bar_viz(host_nb_df, "ML_algo", "DS_Respondents", title="Most Commonly used ML Algos\n", figure_size=(6,4), bar_cmap="Set3")
# Most used data viz tools



feature = 'Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice'

cnt_dict = dict(sorted((counting_dict(aggregate_vals(ds, feature))).items(), key=lambda x: x[1], reverse=True))

# slight clean_up

del cnt_dict['']

del cnt_dict['Other']

del cnt_dict['None']



host_nb_df = pd.DataFrame({"Rdbms":list(cnt_dict.keys()), "DS_Respondents":list(cnt_dict.values())})



bar_viz(host_nb_df, "Rdbms", "DS_Respondents", title="Most Used Relational Database Products\n", figure_size=(6,4), bar_cmap="Set3")