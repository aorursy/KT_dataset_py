# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.metrics.pairwise import cosine_similarity

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
donors = pd.read_csv("../input/Donors.csv")
donations = pd.read_csv("../input/Donations.csv")
donations.head()
temp = donations["Donor ID"].value_counts()
single_time_donors = [x for x in temp.index if temp[x] == 1]
donations_temp = donations[donations["Donor ID"].isin(single_time_donors)]
def donations_filter(months=None, year=None):
    if year is None:
        raise ValueError("year argument, year cannot be none or empty list")
    
    donations_temp_interval = donations_temp.copy()
    donations_temp_interval["Donation Received Date"] = donations_temp_interval["Donation Received Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    donations_temp_interval["year"] = donations_temp_interval["Donation Received Date"].apply(lambda x: x.year)
    donations_temp_interval["Month"] = donations_temp_interval["Donation Received Date"].apply(lambda x: x.month)
    if months is not None:
        donations_temp_interval_new = donations_temp_interval.loc[(donations_temp_interval["year"].isin(year)) & (donations_temp_interval["Month"].isin(months))].copy()
        return donations_temp_interval_new
    if year is not None:
        donations_temp_interval_new = donations_temp_interval.loc[donations_temp_interval["year"].isin(year)].copy()
        return donations_temp_interval_new
# Getting the list of donations from the year 2017
donations_temp_interval_new = donations_filter(months=[1,2,3,4,5,6,7,8,9,10,11,12], year=[2017])
# Verify that donations are from 2017 year
donations_temp_interval_new.head()
# Getting list of donors who daontaed once
single_item_donors_interval = donations_temp_interval_new["Donor ID"].values
projects = pd.read_csv("../input/Projects.csv")
projects_dummies = pd.get_dummies(projects["Project Subject Category Tree"])
project_subject_values = projects_dummies.columns.values
# Appending Project Subject category Tree dummies columns to projects dataset (dataframe)
projects = pd.concat([projects, projects_dummies], axis=1)
projects_dummies = pd.get_dummies(projects["Project Resource Category"])
# Appending  Project Resource category dummies columns to projects dataset (dataframe)
project_recource_category_values = projects_dummies.columns.values
#project_recource_category_values
projects = pd.concat([projects, projects_dummies], axis=1)
def cost_range(amount):
    if amount < 500:
        return "Project_Cost_Category_1"
    elif 500 < amount < 1000:
        return "Project_Cost_Category_2"
    elif 1000 < amount < 2000:
        return "Project_Cost_Category_3"
    elif 2000 < amount < 5000:
        return "Project_Cost_Category_4"
    else:
        return "Project_Cost_Category_5"
# Applying the cost_range to make the project cost range to 5 different categories 
projects["Project_Cost_Encode"] = projects["Project Cost"].apply(cost_range)
projects_dummies = pd.get_dummies(projects["Project_Cost_Encode"])
project_cost_values = projects_dummies.columns.values
projects = pd.concat([projects, projects_dummies], axis=1)
# Returns the projects Dataframe list with only project current status as Live 
def get_live_projects(projects_temp=None):
    projects_temp_live = projects_temp.loc[projects_temp["Project Current Status"]=="Live"].copy()
    projects_temp_live["Project Expiration Date"] = projects_temp_live["Project Expiration Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
    return projects_temp_live
projects_live = get_live_projects(projects)
# Calculates the cosine similarity between given project and all the projects 

# Get the project attributes to find similarity
projects_features = project_subject_values.tolist()
# Creating the project features list to calculate cosine similarity
projects_features = projects_features + project_recource_category_values.tolist() +  project_cost_values.tolist()
#projects_features.append("Project_Cost_Encode")

def get_cosine_similarity(df1, projects_temp=None):
    # Removes the project if its present in Live projects list 
    if df1["Project Current Status"].values[0] == "Live":
        t = df1["Project ID"].values[0]
        index = projects_temp[projects_temp["Project ID"] == t].index[0]
        projects_temp.drop(projects_temp.index[index])
        
    # Calculates the cosine similarity of the project and remainig projects
    cosine_values = cosine_similarity(df1[projects_features], 
                                      projects_temp[projects_features])
    
    projects_temp["cosine values"] = cosine_values[0]
    return projects_temp
def recommend(donor_id):
    # Identify and store the project donated by a donor
    donor_project_temp = None
    donor_project_temp = donations[donations["Donor ID"]==donor_id]
    project = donor_project_temp["Project ID"].values
    project = projects[projects["Project ID"]==project[0]]
    if not project.empty:
        print("Donor:{0} donated to Project with ID:{1} only once".format(donor_id, project["Project ID"].values[0]))
        return get_cosine_similarity(project, projects_live).sort_values(by="cosine values",ascending = False)
    return None
def get_recommened_project(no_of_donors):
    for donor_id in single_item_donors_interval[0:no_of_donors]:
        recommended_projects = None
        recommended_projects = recommend(donor_id)
        print("Top 10 recommended projects for Donor with ID:{}".format(donor_id))
        if not recommended_projects is None:
            #recommended_projects = recommended_projects.sort_values(by="Project Expiration Date",ascending=True)
            print(recommended_projects["Project ID"].head(10).values) 
        else:
            print("Could not found record for Project")
        print("\n")

get_recommened_project(4)
project_attributes = ["Project ID","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost"]
projects[projects["Project ID"]=="000109232e37607c2de30ac1e103fa22"][project_attributes]
top_recommended_projects =['8fd5b4df6e1750062b55aa5cbdfe0351','3aa3950688d9dbede42119b4bdefbc7f',
 '25d8504da49ba9116960edef4ee0e938', '3d47a32f4510fa517fc57e38ee91fa81',
 'ac462cb7d16b2e8ba93a53044082acec', '42706a57dcdd71cf50b21e359966ad65',
 '2a73094b0553648156acf80f729b8211', 'b1b0a430b4de793575770a502243b22e',
 'd6b9fcdf13cdb22a2e4aa83a3b00c0ac', '6631b2bfe584d276be23e869511f7a76']

donor_id= "cf41d1bde03d1633b1338a92f1824577"
projects[projects["Project ID"].isin(top_recommended_projects)][project_attributes]
schools = pd.read_csv("../input/Schools.csv")
def mixed_bag_projects(donor_id=None, projects_temp=None):
    if donor_id is not None and len(recommended_projects):
        donor = donors[donors["Donor ID"] == donor_id]
        school = schools[schools["School ID"].isin(projects_temp["School ID"].values.tolist())]
        projects_temp = pd.merge(projects_temp, school,on="School ID",how="left")
        projects_expire_soon = projects_temp.sort_values(by = "Project Expiration Date", ascending = True)
        
        projects_local = projects_temp[projects_temp["School State"].isin([donor["Donor State"].values[0]])]
        projects_non_local = projects_temp[projects_temp["School State"] != donor["Donor State"].values[0]]
        projects_low_income = projects_non_local[projects_non_local["Project Short Description"].str.contains("low income")]
        return projects_local, projects_non_local, projects_low_income
recommended_projects = recommend(donor_id)
recommended_projects = recommended_projects.sort_values(by = "cosine values", ascending = False).head(10)
projects_local, projects_non_local, projects_low_income = mixed_bag_projects(donor_id="07a164db3073f2b0974d58ac64e3e9fe", projects_temp=recommended_projects)
project_attributes = ["Project ID","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost", "School State", "Project Short Description"]
projects_local_new = None
if not projects_local.empty:
    print("Projects which will expire soon and belongs to the state where donor is")
    projects_local_new = projects_local.sort_values(by = "Project Expiration Date", ascending = True)[project_attributes]
projects_local_new
projects_low_income_new = None
if not projects_low_income.empty:
    print("Projects which will expire soon and have low income")
    projects_low_income_new = projects_low_income[project_attributes]
projects_low_income_new
projects_non_local_remaining = None
if not projects_low_income.empty:        
    print("Most recommended projects")
    projects_non_local_remaining = projects_non_local[~projects_non_local["Project ID"].isin(projects_low_income["Project ID"].values)][project_attributes]
projects_non_local_remaining
projects_non_local_new = None
if projects_low_income.empty:
    print("Most recommended projects")
    projects_non_local_new = projects_non_local[project_attributes]
projects_non_local_new