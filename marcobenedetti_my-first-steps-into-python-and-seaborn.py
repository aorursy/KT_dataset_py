#Import libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
print('Working dir: ', os.getcwd())
#plot in the browser for jupyter
%matplotlib inline 
plt.rcParams['figure.figsize'] = 8,4
import warnings
warnings.filterwarnings('ignore') #ignore /0 errors
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#Extract the data
ms = pd.read_csv("../input/Mass Shootings Dataset Ver 5.csv", encoding = "ISO-8859-1", parse_dates=["Date"])
#Print number of columns and rows to have the cardinality
print("Rows, Columns: ", ms.shape)
print("Max date: ", max(ms.Date))
#Identify the columns
ms.columns
#Clean columns Names removing spaces and special character, this makes it easier to use them later, with Python the syntax df.ColumnName is pretty handy rather than using df['Column Name'] 
ms.columns = ['S#', 'Title', 'Location', 'Date', 'IncidentArea', 'OpenCloseLocation', 'Target',
       'Cause', 'Summary', 'Fatalities', 'Injured', 'TotalVictims',
       'PolicemanKilled', 'Age', 'Employeed', 'EmployedAt',
       'MentalHealthIssues', 'Race', 'Gender', 'Latitude', 'Longitude']
ms.info()
#Show to rows
ms.head(5)
ms.tail(3)
#Split location column
ms['City'] = ms.Location.str.split(', ').str.get(0)
ms['State'] = ms.Location.str.split(', ').str.get(1)
ms = ms.drop(['Location'], axis=1)

#Add Year columns
ms['Year'] = ms['Date'].dt.year

#Convert the categorical columns form object to category type
ms.MentalHealthIssues = ms.MentalHealthIssues.astype('category')
ms.City = ms.City.astype('category')
ms.State = ms.State.astype('category')
ms.Race = ms.Race.astype('category')
ms.Gender = ms.Gender.astype('category')
ms.OpenCloseLocation = ms.OpenCloseLocation.astype('category')
ms.Target = ms.Target.astype('category')
ms.Cause = ms.Cause.astype('category')
ms.Employeed = ms.Employeed.astype('category')
ms.Year = ms.Year.astype('category')

#Check the content of each category
print(ms.MentalHealthIssues.cat.categories)
print(ms.Gender.cat.categories)
print(ms.Race.cat.categories)
print(ms.OpenCloseLocation.cat.categories)
print(ms.Target.cat.categories)
print(ms.Cause.cat.categories)
print(ms.Employeed.cat.categories)
print(ms.Year.cat.categories)
#Add column for Counter
ms['Counter'] = '1'
#Clean category contents rreplacing similar values (e.g. Gender M and Male to Male)
#MentalHealthIssues
conditions = [
    (ms['MentalHealthIssues'] == 'Yes') ,
    (ms['MentalHealthIssues'] == 'No') ]
choices = ['Yes', 'No']
ms['MentalHealthIssues'] = np.select(conditions, choices, default='Unknown')

#Gender
conditions = [
    ((ms['Gender'] == 'M') | (ms['Gender'] == 'Male')) ,
    ((ms['Gender'] == 'F') | (ms['Gender'] == 'Female')) ,
    ((ms['Gender'] == 'M/F') | (ms['Gender'] == 'Male/Female')) ]
choices = ['Male', 'Female', 'Mixed']
ms['Gender'] = np.select(conditions, choices, default='Unknown')

#Race
conditions = [
    ((ms['Race'] == 'Latino') ) ,
    ((ms['Race'] == 'Black') | (ms['Race'] == 'Black American or African American/Unknown') | (ms['Race'] == 'Black American or African American') | (ms['Race'] == 'black') )  ,
    ((ms['Race'] == 'White') | (ms['Race'] == 'White American or European American') | (ms['Race'] == 'White American or European American/Some other Race') | (ms['Race'] == 'White ') | (ms['Race'] == 'white')  ) ,
    ((ms['Race'] == 'Asian') | (ms['Race'] == 'Asian American') | (ms['Race'] == 'Asian American/Some other race')   ),
    ((ms['Race'] == 'Latino') ) , 
    ((ms['Race'] == 'Native American') | (ms['Race'] == 'Native American or Alaska Native')) ]
     
choices = ['Latino', 'Black', 'White', 'Asian', 'Latino', 'Native American' ]
ms['Race'] = np.select(conditions, choices, default='Unknown')

ms.MentalHealthIssues = ms.MentalHealthIssues.astype('category')
ms.Race = ms.Race.astype('category')
ms.Gender = ms.Gender.astype('category')

print(ms.MentalHealthIssues.cat.categories)
print(ms.Gender.cat.categories)
print(ms.Race.cat.categories)

ms.info()
# Let's check some statistical information on the numeric fields
ms.describe().transpose()
ml = sns.distplot(ms.Fatalities, bins=15)
plt.title("Distibutions of Fatalities", fontsize = 15)
plt.show()
ml = sns.distplot(ms.Injured, bins=15)
plt.title("Distibutions of Injured", fontsize = 15)
plt.show()
ml = sns.distplot(ms.TotalVictims, bins=15)
plt.title("Distibutions of Total Victims", fontsize = 15)
plt.show()
#Exclude non relevant columns
cols = [col for col in ms.columns if col not in ['S#']]

#Correlation matix
f, ax = plt.subplots(figsize=(5, 4))
corr = ms[cols].corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()
#Jointplot - exclude outliers
j = sns.jointplot(data = ms, x ='Fatalities', y='Injured', ylim=(0,80),xlim=(0,80), )

#Barchart 
g = sns.factorplot(data=ms, x="Year", kind="count",color = '#1e488f', size=6, aspect=1.5)
g.set_xticklabels(rotation = 90)
plt.title("Number of shootings by year", fontsize = 20)
plt.ylabel("Number of shootings", fontsize = 13)
plt.xlabel("Year", fontsize = 13)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.show()
#Let's build a function!
def Fzfactorplot (field):
    g= sns.factorplot(data=ms, x=field , kind="count", color="#1e488f",  order = ms[field].value_counts().index)
    plt.title("Number of shootings by " + field, fontsize = 13)
    plt.ylabel("Number of shootings", fontsize = 10)
    g.set_xticklabels(rotation = 90)
    plt.show()
Fzfactorplot("MentalHealthIssues")
Fzfactorplot("Gender")
Fzfactorplot("Race")
Fzfactorplot("OpenCloseLocation")
Fzfactorplot("Cause")
#Efficacy
ms_EfficacyByCause = ms.groupby(["Cause"])['Counter','TotalVictims','Fatalities','Injured'].aggregate(np.sum).reset_index().sort_values('TotalVictims')
ms_EfficacyByCause['Efficacy'] = ms_EfficacyByCause.Fatalities / ms_EfficacyByCause.TotalVictims * 100
ms_EfficacyByCauseFiltered = ms_EfficacyByCause[ms_EfficacyByCause.Cause.isin(['psycho','terrorism','anger','frustration','domestic dispute','unemployement','revenge', 'racism'])]

#Causes by total victims
ms_EfficacyByCauseFiltered.sort_values('TotalVictims', ascending=False)
#Efficacy
g= sns.factorplot(data=ms_EfficacyByCauseFiltered, x='Cause' , y = 'Efficacy', color="orange", kind='bar', order = ms_EfficacyByCauseFiltered.sort_values('Efficacy', ascending=False)['Cause'])
plt.title("Shootings efficacy (Fatalities / TotalVictims)", fontsize = 13)
plt.ylabel("Efficacy", fontsize = 10)
g.set_xticklabels(rotation = 90)
plt.show()
#median number of victims by cause
ms_EfficacyByCause = ms.groupby(["Cause"])['TotalVictims','Fatalities','Injured'].aggregate(np.median).reset_index().sort_values('TotalVictims')
ms_EfficacyByCause[::-1]
result = ms[["Title","Year","Cause","Race","IncidentArea", "MentalHealthIssues","Injured","Fatalities","TotalVictims"]].sort_values(["TotalVictims"],ascending =0)
result.head(10)