

# imports

import matplotlib.pyplot as plt

import datetime as dt

import pandas as pd 

%matplotlib inline



# fetch datasets

competitions = pd.read_csv('../input/meta-kaggle/Competitions.csv')

datasets = pd.read_csv('../input/meta-kaggle/Datasets.csv')

dataset_version = pd.read_csv('../input/meta-kaggle/DatasetVersions.csv')

users = pd.read_csv('../input/meta-kaggle/Users.csv')

kernels = pd.read_csv("../input/meta-kaggle/Kernels.csv")

kernel_versions = pd.read_csv('../input/meta-kaggle/KernelVersions.csv')

forum_topics = pd.read_csv("../input/meta-kaggle/ForumTopics.csv")



import warnings

warnings.filterwarnings("ignore")



# to extact 2019 data

def year_extract_2019(df,date_column,format="%m/%d/%Y %H:%M:%S", year_start =dt.datetime(2018,12,31,23,59,59),

                      year_end =dt.datetime(2019,12,31,23,59,59)):

    df[date_column] = pd.to_datetime(df[date_column], format= format)

    mask = (df[date_column] > year_start) & (df[date_column] <= year_end)

    return df.loc[mask]



# for clickable url

def make_clickable_both(val): 

    title, competition_url = val.split('#')

    return f'<a href="{competition_url}">{title}</a>'

competitions_2019 = year_extract_2019(competitions, 'EnabledDate')

print(f'Total Number of Competitions Hosted in Kaggle in 2019 :{competitions_2019.shape[0]}')
ax = competitions_2019['HostSegmentTitle'].value_counts().plot(kind='barh', figsize=(10,7),

                                                 color="slateblue", fontsize=13);

ax.set_alpha(0.8)

ax.set_title(" Competitions in 2019", fontsize=18)

ax.set_xlabel("Number of Competitions", fontsize=18);

ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800,900, 1000])

for i in ax.patches:

    ax.text(i.get_width()+.1, i.get_y()+.31, \

            str(round((i.get_width()), 2)), fontsize=15, color='dimgrey')



# invert for largest on top 

ax.invert_yaxis()
df = competitions_2019.sort_values(by ='TotalSubmissions' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]

df['competition_url'] ='https://www.kaggle.com/c/'+df['Slug']

df['Competition'] = df['Title'] + '#' + df['competition_url']

df= df.filter(['Competition','TotalSubmissions', 'HostSegmentTitle'], axis=1)

df.rename(columns={'TotalSubmissions':'Total Submissions','HostSegmentTitle':'Competition Type'}, inplace=True)

df.reset_index(drop=True).style.format({'Competition': make_clickable_both}).hide_index()
df = competitions_2019.sort_values(by ='TotalTeams' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]

df['competition_url'] ='https://www.kaggle.com/c/'+df['Slug']

df['Competition'] = df['Title'] + '#' + df['competition_url']

df= df.filter(['Competition','TotalTeams', 'HostSegmentTitle'], axis=1)

df.rename(columns={'TotalTeams':'Total Teams','HostSegmentTitle':'Competition Type'}, inplace=True)

df.reset_index(drop=True).style.format({'Competition': make_clickable_both}).hide_index()
df = competitions_2019.sort_values(by ='NumPrizes' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]

df['competition_url'] ='https://www.kaggle.com/c/'+df['Slug']

df['Competition'] = df['Title'] + '#' + df['competition_url']

df= df.filter(['Competition','NumPrizes', 'HostSegmentTitle'], axis=1)

df.rename(columns={'HostSegmentTitle':'CompetitionType'}, inplace=True)

df.reset_index(drop=True).style.format({'Competition': make_clickable_both}).hide_index()
datasets_2019 = year_extract_2019(datasets,'CreationDate')

print (f'Total Number of Datasets Uploaded on Kaggle in 2019  :{ datasets_2019.shape[0]}')
df = datasets_2019.sort_values(by ='TotalDownloads' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]

df = df[['Id','CreatorUserId','TotalDownloads']]

df2 = dataset_version[['DatasetId','Title','Slug']].drop_duplicates(subset=['Title','Slug'])

df = pd.merge(df, df2, how='left', left_on=['Id'], right_on=['DatasetId'])

df = pd.merge(df, users, how='left', left_on=['CreatorUserId'], right_on=['Id'])

# special case

df['UserName'].replace(

    to_replace=['promptcloud'],

    value='PromptCloudHQ',

    inplace=True

)

df['dataset_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['Slug']

df['Dataset'] = df['Title'] + '#' + df['dataset_url']

df= df.filter(['Dataset','TotalDownloads'], axis=1)

df.reset_index(drop=True).style.format({'Dataset': make_clickable_both}).hide_index()

df = datasets_2019.sort_values(by ='TotalViews' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]



df = df[['Id','CreatorUserId','TotalViews']]

df2 = dataset_version[['DatasetId','Title','Slug']]

df = pd.merge(df, df2, how='left', left_on=['Id'], right_on=['DatasetId'])

df= df.drop_duplicates(subset=['Title','Slug'])

df = pd.merge(df, users, how='left', left_on=['CreatorUserId'], right_on=['Id'])



df['UserName'].replace(

    to_replace=['promptcloud'],

    value='PromptCloudHQ',

    inplace=True

)



df['dataset_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['Slug']

df['Dataset'] = df['Title'] + '#' + df['dataset_url']

df= df.filter(['Dataset','TotalViews'], axis=1)

df.reset_index(drop=True).style.format({'Dataset': make_clickable_both}).hide_index()
df = datasets_2019.sort_values(by ='TotalVotes' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]

df = df[['Id','CreatorUserId','TotalVotes']]

df2 = dataset_version[['DatasetId','Title','Slug']]



df = pd.merge(df, df2, how='left', left_on=['Id'], right_on=['DatasetId'])

df= df.drop_duplicates(subset=['Title','Slug'])

df = pd.merge(df, users, how='left', left_on=['CreatorUserId'], right_on=['Id'])

df= df.drop_duplicates(subset=['CreatorUserId','DatasetId'])





df['UserName'].replace(

    to_replace=['promptcloud'],

    value='PromptCloudHQ',

    inplace=True

)



df['dataset_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['Slug']

df['Dataset'] = df['Title'] + '#' + df['dataset_url']

df= df.filter(['Dataset','TotalVotes'], axis=1)

df.reset_index(drop=True).style.format({'Dataset': make_clickable_both}).hide_index()

users_2019 = year_extract_2019(users, 'RegisterDate',"%m/%d/%Y")

performancetier = {0 : "Novice",1: "Contributor", 2: "Expert", 3: "Master", 4: "Grandmaster", 5 :"Kaggle Team Member" }

users_2019['PerformanceTier'] = users_2019['PerformanceTier'].map(performancetier)



print (f'Total Number of Users Registered in Kaggle in 2019  :{ users_2019.shape[0]}')
ax = users_2019['PerformanceTier'].value_counts().plot(kind='barh', figsize=(10,7),

                                                 color="slateblue", fontsize=13);

ax.set_alpha(0.8)

ax.set_title(" Status of Users Registered in 2019", fontsize=18)

ax.set_xlabel("Number of Users ", fontsize=18);

ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000 ])





# set individual bar lables using above list

for i in ax.patches:

     

    ax.text(i.get_width()+.1, i.get_y()+.31, \

            str(round((i.get_width()), 2)), fontsize=15, color='dimgrey')
df = users_2019.loc[users_2019['PerformanceTier'].isin(['Master','Grandmaster'])].sort_values(by=['PerformanceTier'])

df['user_url'] ='https://www.kaggle.com/'+df['UserName']

df['User'] = df['DisplayName'] + '#' + df['user_url']

df= df.filter(['User','PerformanceTier'], axis=1)

df.rename(columns={'PerformanceTier':'Status'}, inplace=True)

df.reset_index(drop=True).style.format({'User': make_clickable_both}).hide_index()
kernels_2019 = year_extract_2019(kernels, 'CreationDate')

print (f'Total Number of Kernels Created in Kaggle in 2019  :{ kernels_2019.shape[0]}')
df = kernels_2019.sort_values(by ='TotalVotes' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]



df = pd.merge(df, kernel_versions, how='left', left_on=['CurrentKernelVersionId'],

               right_on=['Id'])

 

df= df[['AuthorUserId_x','CurrentUrlSlug','Title', 'TotalVotes_x']]



df = pd.merge(df, users, how='left', left_on=['AuthorUserId_x'], right_on=['Id'])

df['keranel_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['CurrentUrlSlug']

df['Kernel'] = df['Title'] + '#' + df['keranel_url']

df= df.filter(['Kernel','TotalVotes_x'], axis=1)

df.rename(columns={'TotalVotes_x':'TotalVotes'}, inplace=True)

df.reset_index(drop=True).style.format({'Kernel': make_clickable_both}).hide_index()
df = kernels_2019.sort_values(by ='TotalViews' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]



df = pd.merge(df, kernel_versions, how='left', left_on=['CurrentKernelVersionId'],

               right_on=['Id'])

 

df= df[['AuthorUserId_x','CurrentUrlSlug','Title', 'TotalViews']]



df = pd.merge(df, users, how='left', left_on=['AuthorUserId_x'], right_on=['Id'])

df['keranel_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['CurrentUrlSlug']

df['Kernel'] = df['Title'] + '#' + df['keranel_url']

df= df.filter(['Kernel','TotalViews'], axis=1)

#df.rename(columns={'TotalVotes_x':'TotalVotes'}, inplace=True)

df.reset_index(drop=True).style.format({'Kernel': make_clickable_both}).hide_index()
df = kernels_2019.sort_values(by ='TotalComments' , 

                                     axis=0, ascending=False, inplace=False, 

                                     kind='quicksort', na_position='last')[:10]



df = pd.merge(df, kernel_versions, how='left', left_on=['CurrentKernelVersionId'],

               right_on=['Id'])

 

df= df[['AuthorUserId_x','CurrentUrlSlug','Title', 'TotalComments']]



df = pd.merge(df, users, how='left', left_on=['AuthorUserId_x'], right_on=['Id'])

df['keranel_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['CurrentUrlSlug']

df['Kernel'] = df['Title'] + '#' + df['keranel_url']

df= df.filter(['Kernel','TotalComments'], axis=1)

 

df.reset_index(drop=True).style.format({'Kernel': make_clickable_both}).hide_index()