import pandas as pd  #Library for dataprocessing
import numpy as np   #Library for linear algebra

#Import Library for geographical plot
import plotly.graph_objs as go

from IPython.display import HTML
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

#Import Data Visualisation libaries
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#plt.style.use('dark_background')
%matplotlib inline
sns.set()
#reading in the dataset
multipleChoice = pd.read_csv("../input/multipleChoiceResponses.csv")
#Checking the content of the first five rows in the dataset
multipleChoice.head()
# Checking the information about the dataset
multipleChoice.info()
#Creating a copy of the dataset
mC = multipleChoice.copy()
#Checking for null values in question 1
mC["Q1"].isnull().sum()
# Drop the first column because it is not needed as part of the analysis. It represent each question description.
mC = mC.drop(0, axis =0)
mC.head()
# Checking each sex declared with respect to their numbers
mCSex1 = mC["Q1"].value_counts()
mCSex = pd.DataFrame(mCSex1)
#Resetting the Index
mCSex.reset_index(inplace=True)
#Create an empty column to represent the percentage sex declared
mCSex["sexCount%"] = 0

#Renaming the colums
mCSex.columns = ["Sex", "SexCount", "sexCount%"]
mCSex
# find the percentages of each sex count and save in the sexCount column
sexlist = []
mCSexTotal = mCSex["SexCount"].sum()
for i in range(0, len(mCSex)):
    b = mCSex["SexCount"].iloc[i]
    c = (b/mCSexTotal)*100
    d = round(c, 2)  # round to two decimal place
    sexlist.append(d)
mCSex["sexCount%"] = sexlist
print("Table 1: Dataframe showing the number of repondents with respect their Sex ")
mCSex
plt.figure(figsize=(9,7))
plt.rcParams["xtick.labelsize"] = 16
ax=plt.gca()
ax.set_facecolor("xkcd:pink")
ax= sns.barplot(x=mCSex["Sex"], y=mCSex["sexCount%"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("Figure 1: Percentage of Kaggle Survey Respondent", fontsize= 18)
plt.xlabel("Sex", fontsize =20)
plt.ylabel("Respondent Count(%)", fontsize =20)
plt.show()
# Slicing out the male Kaggle respondent and Data Chicks
maleKag = mC[mC["Q1"] == "Male"]
dataChicks = mC[mC["Q1"] == "Female"]

#Slicing out the country of Data Chicks and counting unique values
dataChicksCtry = dataChicks["Q3"].value_counts()
#Reset index of the new dataframe
dataChicksCtryIndx = dataChicksCtry.reset_index()
#Rename Columns of DataFrame
dataChicksCtryIndx.columns= ["Country", "Number of DataChicks"]
#Create a new column called Code in the Data Chick Country table
dataChicksCtryIndx["Codes"] = 0
#Sort the Country column in ascending order
dataChicksCtryIndx = dataChicksCtryIndx.sort_values("Country")
#Assign country codes to each country in the Data Frame
countryCodes = ["ARG","AUS", "AUT", "BGD","BLR", "BEL", "BRA","CAN", "CHL","CHN", "COL","CZE", "DNK", "EGY",
                 "FIN","FRA", "DEU","GRC","HKG","HUN","NOTDISCLOSE", "IND","IDN","IRN", "IRL", "ISR","ITA", 
                "JPN","KEN","MYS","MEX","MAR","NLD","NZL","NGA","NOR","OTHER", "PAK","PER", "PHL", "POL", "PRT",
                "PRK", "ROU","RUS","SGP","ZAF","KOR", "ESP", "SWE", "CHE", "THA", "TUN", "TUR", "UKR","GBR", "USA", 
                    "VNM"]
dataChicksCtryIndx["Codes"] = countryCodes
#Reset the Index of the Data Frame
dataChicksCtryIndx.reset_index(inplace=True)
#Drop the Index Column
dataChicksCtryIndx.drop(("index"), axis=1, inplace=True)
#Drop the "Other" and "I do not wish to disclose my location from the dataset"
dataChicksCtryIndx.drop([20,36], axis=0, inplace=True)
#Create a new copy of the dataset
dataChickCtryNew= dataChicksCtryIndx.copy()
# Check  the first five row of the Data Chicks country dataset
dataChickCtryNew.head()
# First 5 Countries with high number of data chicks respondent out of a total of 4010
dataChicksCtry.head()
# Last 5 Countries with low number of data chicks respondent out of a total of 4010
dataChicksCtry.tail()
# Setting the text description of the Choropleth map
dataChickCtryNew["text"] = 'DataChicks: ' +dataChickCtryNew["Number of DataChicks"].astype(str) + " Respondents" +'<br>'+\
'Country: ' +dataChickCtryNew["Country"]

# Defining the data inputs for the geographical map
data = [dict(type="choropleth", autocolorscale=False, locations = dataChickCtryNew["Codes"],
             z= dataChickCtryNew["Number of DataChicks"],
            text= dataChickCtryNew["text"], 
            colorscale= 'Rainbow',
            marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        reversescale = False, colorbar = dict(title = "Female Respondents"))]

# Setting parameters for the layout of the map
layout = dict(title = "Figure 2: Data Chicks Kaggle Survey Respondent and their Geographical Locations",
              width= 850,
              height= 700,
              geo= dict(showframe= False,
                       showcoastlines = False,
                       countrycolor = "rgb(211,211,211)",
                       showcountries = True,
                       projection = dict(
                           type= "mercator"
                           ),
                       showlakes = True,
                       lakecolor = "rgb(66, 165, 245)",
                       )
             )
# Visualising the map
choromap = go.Figure(data = data, layout = layout)
iplot(choromap, validate=False)
# Checking the number of countries that Data Chicks come from
dataChicksCtry.value_counts().sum()
#Checking the age Range of Data Chicks
dCAgeRange = pd.DataFrame(dataChicks["Q2"].value_counts())
dCAgeRange.reset_index(inplace=True)
dCAgeRange.columns = ["AgeRange", "DataChicks_Count"]
dCAgeRange
# plot for the age range of Data Chicks Repondent
plt.figure(figsize=(9,7))
plt.rcParams["xtick.labelsize"] = 12
ax=plt.gca()
ax.set_facecolor("xkcd:purple")
ax= sns.countplot(dataChicks["Q2"], order= ["18-21", "22-24", "25-29","30-34","35-39","40-44",
                                            "45-49","50-54","55-59","60-69","70-79","80+"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("Figure 3: Age Range of Data Chicks Respondent", fontsize= 18)
plt.xlabel("Age Range", fontsize =15)
plt.ylabel("Respondent Count", fontsize =15)
plt.show()
# count the unique items in the education columns for data chicks respondent
eduData=dataChicks["Q4"].value_counts()

#Save in a dataframe
eduData2 = pd.DataFrame(eduData)

#Resetting the Index
eduData2.reset_index(inplace=True)
#Create an empty column to represent the percentage degree count
eduData2["FemaleDegreeCount(%)"] = 0

#Renaming the columns
eduData2.columns = ["Degree", "FemaleDegreeCount", "FemaleDegreeCount(%)"]

# find the percentages of each degree count and save in the DegreeCount column
deglist = []
femaleEduTotal = eduData2["FemaleDegreeCount"].sum()
for i in range(0, len(eduData2)):
    eIndex = eduData2["FemaleDegreeCount"].iloc[i]
    f = (eIndex /femaleEduTotal)*100
    g = round(f, 2)  # round to two decimal place
    deglist.append(g)
eduData2["FemaleDegreeCount(%)"] = deglist
#print("Table 2: Showing the level of education of Data Chicks Respondents")
eduData2
# count the unique items in the education columns for male kaggle respondent
maleEduData=maleKag["Q4"].value_counts()

#Save in a dataframe
maleEduData2 = pd.DataFrame(maleEduData)

#Resetting the Index
maleEduData2.reset_index(inplace=True)
#Create an empty column to represent the percentage degree count
maleEduData2["MaleDegreeCount(%)"] = 0

#Renaming the columns
maleEduData2.columns = ["Degrees", "MaleDegreeCount", "MaleDegreeCount(%)"]

# find the percentages of each degree count and save in the DegreeCount column
mdeglist = []
maleEduTotal = maleEduData2["MaleDegreeCount"].sum()
for i in range(0, len(maleEduData2)):
    maleIndex = maleEduData2["MaleDegreeCount"].iloc[i]
    m = (maleIndex /maleEduTotal)*100
    degC = round(m, 2)  # round to two decimal place
    mdeglist.append(degC)
maleEduData2["MaleDegreeCount(%)"] = mdeglist
#print("Table 3: Showing the level of education of Male Kaggle Respondents")
maleEduData2
# Concatenating the two tables
maleFemaleEdu = pd.concat([eduData2,maleEduData2], axis = 1)
#Drop duplicate columns
maleFemaleEdu.drop(columns="Degrees", inplace= True)
print("Table 2: Showing the level of education of Data Chicks and Male Kaggle Respondents")
maleFemaleEdu
plt.figure(figsize=(9,9))
plt.rcParams["xtick.labelsize"] = 12
ax=plt.gca()
ax.set_facecolor("xkcd:magenta")
ax= sns.countplot(dataChicks["Q4"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("Figure 4: Level of Education of Data Chicks", fontsize= 18)
plt.xlabel("Education", fontsize =15)
plt.ylabel("Respondent Count", fontsize =15)
plt.show()
# Selecting Data Chicks Discipline
courseData2=dataChicks["Q5"].value_counts()
courseData = pd.DataFrame(courseData2)
#Resetting the Index
courseData.reset_index(inplace=True)
#Create an empty column to represent the percentage sex declared
courseData["DataChickCount(%)"] = 0

#Renaming the columns
courseData.columns = ["Discipline", "DataChickCount", "DataChickCount(%)"]

# find the percentages of each discipline count and save in the DataChickCount(%) column
deglist = []
totalCourseData  = courseData["DataChickCount"].sum()
for i in range(0, len(courseData)):
    cIndex = courseData["DataChickCount"].iloc[i]
    h = (cIndex /totalCourseData)*100
    j = round(h, 2)  # round to two decimal place
    deglist.append(j)
courseData["DataChickCount(%)"] = deglist
print("Table 3: Showing the course of study of Data Chicks Respondents")
courseData
# Percentage of Data Chicks studying Data Science Related Course
dataScience = courseData["DataChickCount(%)"].iloc[[0,1,2,5,8]]

#Data Science List
names = ["Computer Science", "Maths/Stats", "Engineering",
                                                  "IT", "Physics"]
#Create a figure and set different background
fig= plt.figure()
fig.patch.set_facecolor('pink')

#Change color of the text
plt.rcParams["text.color"] = "black"

#Create a circle for the center of the plot
my_circle = plt.Circle((0,0), 0.7, color="pink")

#pie plot and circle on it
plt.pie(dataScience, labels= names, colors= ['blue', 'yellow', 'olive', 'red', 'tan'],
        wedgeprops ={'linewidth':7, "edgecolor": 'white'})
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title("Figure 5: Donut plot of Data Chicks in Data Science related Discipline", fontsize= 18)
plt.show()

# Course of Study of Data Chicks with respect to their ages
plt.figure(figsize=(20,15))
plt.rcParams["xtick.labelsize"] = 16
ax=plt.gca()
ax.set_facecolor("xkcd:beige")
ax= sns.countplot(dataChicks["Q5"], hue=dataChicks["Q2"], hue_order=["18-21", "22-24", "25-29","30-34","35-39","40-44",
                                                                "45-49","50-54","55-59","60-69","70-79","80+"] )
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("Figure 6: Course of Study of Data Chicks with respect to their ages", fontsize= 22)
plt.xlabel("Discipline", fontsize =20)
plt.ylabel("Respondent Count", fontsize =20)
plt.show()
# Counting the number of Data Chicks with unique job roles
jobDataChick = dataChicks["Q6"].value_counts()

#Save in a dataframe
jobDataChick2 = pd.DataFrame(jobDataChick)

#Resetting the Index
jobDataChick2.reset_index(inplace=True)
#Create an empty column to represent the percentage job count
jobDataChick2["FemaleJobCount(%)"] = 0

#Renaming the columns
jobDataChick2.columns = ["JobRoles", "FemaleJobCount", "FemaleJobCount(%)"]

# find the percentages of each job role count and save in the FemaleJobCount(%) column
fjoblist = []
femaleJobTotal = jobDataChick2["FemaleJobCount"].sum()
for i in range(0, len(jobDataChick2)):
    fjIndex = jobDataChick2["FemaleJobCount"].iloc[i]
    fj = (fjIndex/femaleJobTotal)*100
    y = round(fj, 2)  # round to two decimal place
    fjoblist.append(y)
jobDataChick2["FemaleJobCount(%)"] = fjoblist
print("Table 4: Showing the Job Count of Data Chicks Respondents")
jobDataChick2
# Counting the number of Male Kagglers with unique job roles
jobMaleKag = maleKag["Q6"].value_counts()

#Save in a dataframe
jobMaleKag2 = pd.DataFrame(jobMaleKag)

#Resetting the Index
jobMaleKag2.reset_index(inplace=True)
#Create an empty column to represent the percentage job count
jobMaleKag2["MaleJobCount(%)"] = 0

#Renaming the columns
jobMaleKag2.columns = ["JobRoles", "MaleJobCount", "MaleJobCount(%)"]

# find the percentages of each job role count and save in the MaleJobCount(%) column
mjoblist = []
maleJobTotal = jobMaleKag2["MaleJobCount"].sum()
for i in range(0, len(jobMaleKag2)):
    mjIndex = jobMaleKag2["MaleJobCount"].iloc[i]
    mj = (mjIndex/maleJobTotal)*100
    z = round(mj, 2)  # round to two decimal place
    mjoblist.append(z)
jobMaleKag2["MaleJobCount(%)"] = mjoblist
print("Table 5: Showing the Job Count of Male Kaggle Respondents")
jobMaleKag2
plt.figure(figsize=(16,14))
plt.rcParams["xtick.labelsize"] = 16
ax=plt.gca()
ax.set_facecolor("xkcd:mustard")
ax= sns.countplot(dataChicks["Q5"], hue=dataChicks["Q6"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("Figure 7: Course of Study of Data Chicks with respect to their Job Roles", fontsize= 22)
plt.xlabel("Course of Study", fontsize =20)
plt.ylabel("Respondent Count", fontsize =20)
plt.show()
plt.figure(figsize=(13,13))
plt.rcParams["xtick.labelsize"] = 16
ax=plt.gca()
ax.set_facecolor("xkcd:cyan")
ax= sns.countplot(dataChicks["Q6"], hue=dataChicks["Q4"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("Figure 8: Data Chicks Job roles with respect to their degrees", fontsize= 22)
plt.xlabel("Job Roles", fontsize =20)
plt.ylabel("Respondent Count", fontsize =20)
plt.show()
plt.figure(figsize=(18,15))
plt.rcParams["xtick.labelsize"] = 16
ax=plt.gca()
ax.set_facecolor("xkcd:black")
ax= sns.countplot(dataChicks["Q7"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("FIGURE 9: Data Chicks with respect to their Industry Presence", fontsize= 22)
plt.xlabel("Industry", fontsize =20)
plt.ylabel("Respondent Count", fontsize =20)
plt.show()
plt.figure(figsize=(11,10))
plt.rcParams["xtick.labelsize"] = 16
ax=plt.gca()
ax.set_facecolor("xkcd:magenta")
lowIncome = dataChicks[dataChicks["Q9"] == "0-10,000"]
ax= sns.countplot(lowIncome["Q9"], hue=dataChicks["Q8"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("FIGURE 10: Years of Experience of Data Chicks with respect to their Annual Takehome", fontsize= 22)
plt.xlabel("Annual Takehome (USD)", fontsize =20)
plt.ylabel("Respondent Count", fontsize =20)
plt.show()
dataChicks["Q12_MULTIPLE_CHOICE"].value_counts()
# Plot showing the work enviroments of data chicks
f, ax = plt.subplots(figsize=(12, 8))
ax=plt.gca()
ax.set_facecolor("xkcd:tan")
sns.countplot(y="Q12_MULTIPLE_CHOICE",data=dataChicks);
plt.title("FIGURE 11: Data Chicks with respect to the tools used at work/school", fontsize= 22)
plt.xlabel("Respondent Count", fontsize =20)
plt.ylabel("Tools Used", fontsize =20)
plt.show()
# Plot showing the preferred of data chicks
plt.figure(figsize=(12,8))
plt.rcParams["xtick.labelsize"] = 13
ax=plt.gca()
ax.set_facecolor("xkcd:wine")
sns.countplot(y="Q17",data=dataChicks)
plt.title("FIGURE 12: Data Chicks and Most preferred Programing Languages", fontsize= 22)
plt.xlabel("Respondent Count", fontsize =20)
plt.ylabel("Programming Language", fontsize =20)
plt.show()
codingTime = pd.DataFrame(dataChicks["Q23"].value_counts())
print("Table 6: Showing the Number of Data Chicks and the amount of time spent actively coding ")
codingTime.columns= ["Number of Data Chicks"]
codingTime
plt.figure(figsize=(9,7))
plt.rcParams["xtick.labelsize"] = 12
ax=plt.gca()
ax.set_facecolor("xkcd:purple")
ax= sns.countplot(dataChicks["Q23"], order =["0% of my time", "1% to 25% of my time", "25% to 49% of my time",
                                              "50% to 74% of my time", "75% to 99% of my time", "100% of my time"])
ax.set_xticklabels(ax.get_xticklabels(), rotation= 40, ha="right")
plt.tight_layout()
plt.title("FIGURE 13: Percentage of Time Data Chicks Spends actively coding", fontsize= 18)
plt.xlabel("Time", fontsize =15)
plt.ylabel("Respondent Count", fontsize =15)
plt.show()
#for p, label in zip(ax.patches, dataChicks["Q23"].value_counts()):
#    ax.annotate(label, (p.get_x()+0.375, p.get_height()+0.15))