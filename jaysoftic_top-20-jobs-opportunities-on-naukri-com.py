# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
%matplotlib inline
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv")
data.sample(5)
# replace space with underscore in column name for quick access
data.columns = data.columns.str.replace(" ", "_")
# filter and find unique() cities from data set

data.Location = data.Location.str.upper()
new_location =data.Location.str.strip().str.split(",", expand = True)[0].str.split(" ", expand = True)[0].value_counts().reset_index()
new_location.columns = ["Location", "Job_Opportunities"]
top_20_new_location = new_location[:20]
top_20_new_location.style.background_gradient(cmap = "Reds")
plt.figure(figsize = (12, 8))
plt.bar(top_20_new_location.Location, top_20_new_location.Job_Opportunities, color = "r")
plt.xlabel("Locations")
plt.ylabel("Job Opportunities")
plt.xticks(top_20_new_location.Location, rotation = "60")
plt.title("Job Opportunities in Top 20 Locations", fontdict={"fontsize" :20})
plt.show()
# filter and find unique Industry from data set

new_Industry = pd.DataFrame(data.Industry.str.split(",", expand = True).values.ravel("f"),columns = ["Industry"])
new_Industry = new_Industry.dropna()
new_Industry.Industry = new_Industry.Industry.str.upper()
new_Industry.Industry = new_Industry.Industry.apply(lambda x: re.sub("[^A-Za-z0-9 -]+", ",", x))
new_Industry = pd.DataFrame(new_Industry.Industry.str.split(",", expand = True).values.ravel("f"), columns = ["Industry"])
new_Industry = new_Industry.dropna()
new_Industry.Industry = new_Industry.Industry.str.lstrip().str.rstrip()
# pure 20 Industry
new_Industry = new_Industry.Industry.value_counts().reset_index()
new_Industry.columns =["Industry", "Job_Opportunities"]
top_20_new_Industry = new_Industry[:20]
top_20_new_Industry.style.background_gradient(cmap = "Blues")
plt.figure(figsize = (12, 8))
plt.bar(top_20_new_Industry.Industry, top_20_new_Industry.Job_Opportunities, color = "b")
plt.xlabel("Industries")
plt.ylabel("Job Opportunities")
plt.xticks(top_20_new_Industry.Industry, rotation = "vertical")
plt.title("Job Opportunities in Top 20 Industries", fontdict={"fontsize" :20})
plt.show()
new_Role_Category = data.Role_Category.str.lstrip().str.rstrip().value_counts().reset_index()
new_Role_Category.columns = ["Role_Category", "Job_Opportunities"]
top_20_new_Role_Category = new_Role_Category[:20]
top_20_new_Role_Category.style.background_gradient(cmap = "Greens")
plt.figure(figsize = (12, 8))
plt.bar(top_20_new_Role_Category.Role_Category, top_20_new_Role_Category.Job_Opportunities, color = "g")
plt.xlabel("Job Catergories")
plt.ylabel("Job Opportunities")
plt.xticks(top_20_new_Role_Category.Role_Category, rotation = "vertical")
plt.title("Top 20 Job Categories", fontdict={"fontsize" :20})
plt.show()
new_Job_Role = data.Role.str.lstrip().str.rstrip().value_counts().reset_index()
new_Job_Role.columns = ["Role", "Job_Opportunities"]
top_20_new_Job_Role = new_Job_Role[:20]
top_20_new_Job_Role.style.background_gradient(cmap = "Purples")
plt.figure(figsize = (12, 8))
plt.bar(top_20_new_Job_Role.Role, top_20_new_Job_Role.Job_Opportunities, color = "purple")
plt.xlabel("Job Role")
plt.ylabel("Job Opportunities")
plt.xticks(top_20_new_Job_Role.Role, rotation = "90")
plt.title("Top 20 Job Role", fontdict={"fontsize" :20})
plt.show()
city_industries = pd.DataFrame({"Location": data.Location.str.split(",", expand = True)[0],
                                "Industry": data.Industry.str.split(",", expand = True)[0]})
city_industries.dropna(inplace = True)
new_CT_IND = pd.crosstab(city_industries.Location, city_industries.Industry).reset_index()
new_CT_IND = new_CT_IND.melt(id_vars = "Location", value_name = "Job_Opportunities")
new_CT_IND = new_CT_IND.sort_values(by = ["Industry", "Job_Opportunities"], ascending = False).reset_index(drop = True)
new_CT_IND = new_CT_IND.dropna()
new_CT_IND = new_CT_IND.sort_values(by = "Job_Opportunities", ascending = False).reset_index(drop = True)

# in classy data frame is only for getting top 20 Industry Name
classy = new_CT_IND.groupby("Industry").sum().reset_index().sort_values(by = "Job_Opportunities", ascending = False)[:20]
# saved industry name in list for fetching only 5 same name industry or only five Location of particular indusry
Industry_20_lst = classy.Industry.reset_index(drop = True)
limit = 0
pure_loc_Ind = pd.DataFrame(columns=["Location", "Industry", "Job_Opportunities"])
for j in range(20):
    limit = 0
    for i in range (new_CT_IND.size):
        if limit == 5:
            break
        if Industry_20_lst[j] == new_CT_IND.loc[i, "Industry"]:
            pure_loc_Ind = pure_loc_Ind.append(new_CT_IND.loc[i], ignore_index = True)
            limit += 1
# yeeeah i did it :)
# plot pie chart of top 5 location of top 20 Industry
for i in Industry_20_lst:
    classy = pure_loc_Ind.loc[pure_loc_Ind.Industry == i, ["Location", "Job_Opportunities"]]
    fig = px.pie(classy, names = "Location", values = "Job_Opportunities", color = "Location",
                 title = "Top 5 Location for " + i + " Industry")
    fig.show()
new_job_skills = data[["Role_Category", "Key_Skills"]].copy()
new_job_skills.Key_Skills = new_job_skills.Key_Skills.str.upper()
classy = new_job_skills.Key_Skills.str.split("|", expand = True) # split keys
classy = pd.concat([new_job_skills, classy], axis = 1) # concat splited keys columns
classy = classy.melt(id_vars = ["Role_Category", "Key_Skills"], var_name = "Job_Opportunities", value_name = "Skills")
# transform Key skills using melt
classy = classy.dropna() 
classy.Job_Opportunities = 1 # assign job opportunites value 1 for counting purpose
classy = classy.groupby(["Role_Category", "Skills"]).Job_Opportunities.sum().reset_index()
# group by role and skills and find sum of job opportunities
classy = classy.sort_values(by = "Job_Opportunities", ascending = False).reset_index(drop = True)
pure_skills_job_cat = pd.DataFrame(columns = ["Role_Category", "Skills", "Job_Opportunities"])
role_category_lst = top_20_new_Role_Category.Role_Category
limit = 0
for i in range(20):
    limit = 0
    for j in range(classy.size):
        if limit == 10:
            break
        if role_category_lst[i] == classy.loc[j, "Role_Category"]:
            pure_skills_job_cat = pure_skills_job_cat.append(classy.loc[j])
            limit += 1
px.sunburst(data_frame = pure_skills_job_cat, values = "Job_Opportunities", path = ["Role_Category", "Skills"],
            color = "Role_Category", title = "Skills Required for various Job Categories",height = 600)
new_salary_loc = data[["Job_Salary", "Location"]].copy()
new_salary_loc["New_Location"] = new_salary_loc.Location.str.strip().str.split(",", expand = True)[0]

classy = new_salary_loc.Job_Salary.apply(lambda x: re.sub("[^0-9 -]", "", str(x)))
new_salary_loc["Min_Salary"] = classy.str.strip().str.split("-", expand = True)[0]
new_salary_loc["Max_Salary"] = classy.str.strip().str.split("-", expand = True)[1]
new_salary_loc.dropna(inplace = True)

# remove space between digits
new_salary_loc.Min_Salary = new_salary_loc.Min_Salary.str.replace(" ", "")
new_salary_loc.Max_Salary = new_salary_loc.Max_Salary.str.replace(" ", "")

# put 0 where value is totally empty
new_salary_loc.loc[new_salary_loc.Max_Salary == "", "Max_Salary"] = "0"
new_salary_loc.loc[new_salary_loc.Min_Salary == "", "Min_Salary"] = "0"

# convert it into 
new_salary_loc.Min_Salary = new_salary_loc.Min_Salary.astype("int64")
new_salary_loc.Max_Salary = new_salary_loc.Max_Salary.astype("int64")
new_salary_loc.drop(columns = ["Location", "Job_Salary"], inplace = True) # drop old columns
classy = new_salary_loc.groupby("New_Location").median().reset_index() # find average of salary

top_20_location_salary = pd.DataFrame()
loc_list = top_20_new_location.Location
for i in range(20):
    for j in range(classy.New_Location.size):
        if loc_list[i] == classy.loc[j, "New_Location"]:
            top_20_location_salary = top_20_location_salary.append(classy.loc[j])
            break

top_20_location_salary = top_20_location_salary.reset_index(drop = True)
top_10_location_salary = top_20_location_salary[:10].copy() # find top 10 
top_10_location_salary.rename(columns = {"New_Location": "Location"}, inplace = True)
top_10_location_salary.Max_Salary = top_10_location_salary.Max_Salary.astype("int64")
top_10_location_salary.Min_Salary = top_10_location_salary.Min_Salary.astype("int64")
top_10_location_salary = top_10_location_salary[top_10_location_salary.columns[[2, 1, 0]]] # change column order
top_10_location_salary.style.background_gradient(cmap = "Reds", subset = "Min_Salary").\
                            background_gradient(cmap = "Greens", subset = "Max_Salary")
plt.figure(figsize = (12, 8))
ax = plt.subplot(111)
ax.bar(top_10_location_salary.index - 0.2, top_10_location_salary.Min_Salary, color = "r", width = 0.4, label = "Min Salary")
ax.bar(top_10_location_salary.index +0.2, top_10_location_salary.Max_Salary, color = "g", width = 0.4, label = "Max Salary")
plt.xlabel("Location")
plt.ylabel("Average Salary")
plt.xticks([i for i in top_10_location_salary.index] ,top_10_location_salary.Location)
plt.legend()
plt.show()
new_avg_sal_ind = data[["Location", "Industry", "Job_Salary"]].copy() # preparing data
new_avg_sal_ind.Location = new_avg_sal_ind.Location.str.split(",", expand = True)[0] # preparing location
new_avg_sal_ind.Industry = new_avg_sal_ind.Industry.str.split(",", expand = True)[0] # preparing industry
new_avg_sal_ind.Job_Salary = new_avg_sal_ind.Job_Salary.apply(lambda x: re.sub("[^0-9 -]", "", str(x))) # salary
new_avg_sal_ind["Min_Salary"] = new_avg_sal_ind.Job_Salary.str.split("-", expand = True)[0] # min salary
new_avg_sal_ind["Max_Salary"] = new_avg_sal_ind.Job_Salary.str.split("-", expand = True)[1] # max salary
new_avg_sal_ind.Min_Salary = new_avg_sal_ind.Min_Salary.str.replace(" ", "")
new_avg_sal_ind.Max_Salary = new_avg_sal_ind.Max_Salary.str.replace(" ", "")
new_avg_sal_ind.loc[new_avg_sal_ind.Min_Salary == "", "Min_Salary"] = "0"
new_avg_sal_ind.loc[new_avg_sal_ind.Max_Salary == "", "Max_Salary"] = "0"
new_avg_sal_ind.dropna(inplace = True)
new_avg_sal_ind.Min_Salary = new_avg_sal_ind.Min_Salary.astype("int64")
new_avg_sal_ind.Max_Salary = new_avg_sal_ind.Max_Salary.astype("int64")
classy = new_avg_sal_ind.groupby(["Industry", "Location"]).median().reset_index()
# group industry and location and find average of salary
classy.Industry = classy.Industry.str.upper() # convert upper case of industry
lst_Industry = new_Industry.Industry[:25] # get 25 top industry
lst_Location = top_20_new_location.Location # get 20 top industry
limit = 0
top_10_avg_sal_industry_location = pd.DataFrame(columns = ["Industry", "Location", "Max_Salary", "Min_Salary"])
# creating empty data frame for storing filtering data
for i in range(25):
    limit = 0
    dummy_df = classy.loc[classy.Industry == lst_Industry[i]].reset_index(drop = True)
    # creating dummy df for top 10 industry
    for j in range(20):
        if limit == 5:
            break
        else:
            for k in range(dummy_df.Location.size):
                if lst_Location[j] == dummy_df.loc[k, "Location"]:
                    # append top 5 country and it's average salary of top 10 industry
                    top_10_avg_sal_industry_location = top_10_avg_sal_industry_location.append(dummy_df.loc[k])
                    limit += 1
                    break
classy = top_10_avg_sal_industry_location.melt(id_vars = ["Industry", "Location"], var_name = "Salaries")
px.sunburst(data_frame = classy, path = ["Industry", "Location", "Salaries"],
            values = "value", title = "Average salary of top 10 Industry in top 5 Location", height = 600)