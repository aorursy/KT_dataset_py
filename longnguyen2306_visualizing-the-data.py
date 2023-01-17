import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import datetime as dt

import seaborn as sns

# import gmplot



from pylab import rcParams

%matplotlib inline

rcParams["figure.figsize"] = 12, 8
crime = pd.read_csv("../input/Crime_Data_2010_2017.csv")
print("The shape is {}".format(crime.shape))
crime.tail(3)
# Formatting to datetime object

try:

    date_reported = [dt.datetime.strptime(d, "%m/%d/%Y").date() for d in crime["Date Reported"]]

except:

    print("Already converted Date Reported")

    

try:

    date_occurred = [dt.datetime.strptime(d, "%m/%d/%Y").date() for d in crime["Date Occurred"]]

except:

    print("Already converted Date Occurred")

    

# Reassign the date reported and occurred columns

crime["Date Reported"] = np.array(date_reported)

crime["Date Occurred"] = np.array(date_occurred)
# Making lists of days, months, and years for reported from datetime objects

day_reported = [d.isoweekday() for d in crime["Date Reported"]]

mon_reported = [d.month for d in crime["Date Reported"]]

year_reported = [d.year for d in crime["Date Reported"]]

# Making new columns for each

crime["Day Reported"] = np.array(day_reported)

crime["Month Reported"] = np.array(mon_reported)

crime["Year Reported"] = np.array(year_reported)
# Making lists of days, months, and years for occurred from datetime objects

day_occurred = [d.isoweekday() for d in crime["Date Occurred"]]

mon_occurred = [d.month for d in crime["Date Occurred"]]

year_occurred = [d.year for d in crime["Date Occurred"]]

# Making new columns for each

crime["Day Occurred"] = np.array(day_occurred)

crime["Month Occurred"] = np.array(mon_occurred)

crime["Year Occurred"] = np.array(year_occurred)
fig, ax = plt.subplots()

# Plotting crimes reported by day

sns.barplot(x=crime["Day Reported"].value_counts().index, y=crime["Day Reported"].value_counts())

# Axes

ax.set_title("Total Crimes Reported by Day of the Week")

ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fontsize=9)

ax.set_xlabel("Day of the Week")

ax.set_ylabel("Total Crimes Reported")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine()
fig, ax = plt.subplots()

# Plotting crimes occurred by day

sns.barplot(x = crime["Day Occurred"].value_counts().index, y = crime["Day Occurred"].value_counts())

# Axes

ax.set_title("Total Crimes Occurred by Day of the Week")

ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fontsize=9)

ax.set_xlabel("Day of the Week")

ax.set_ylabel("Total Crimes Occurred")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine()
# Making a new dataframe

df1 = pd.DataFrame({

    "Day" : list(crime["Day Reported"].value_counts().index),

    "Crime Occurred" : list(crime["Day Occurred"].value_counts()),

    "Crime Reported" : list(crime["Day Reported"].value_counts())

})

dayrepocc = df1.set_index("Day").stack().reset_index().rename(columns={"level_1" : "Variable", 0 : "Crime"})
fig, ax = plt.subplots()

# Plotting side by side crime rep and occ by day

sns.barplot(x = "Day", y = "Crime", hue = "Variable", data=dayrepocc, ax=ax)

# Axes

ax.set_title("Crime Reported and Occured by Day")

ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fontsize=9)

ax.set_ylabel("Crime")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine(fig)
fig, ax = plt.subplots()

# Plotting crimes reported by month

sns.barplot(x = crime["Month Reported"].value_counts().index, y = crime["Month Reported"].value_counts())

# Axes

ax.set_title("Total Crimes Reported by Month")

ax.set_xticklabels(["January", "February", "March", "April", "May", "June",

                    "July", "August", "September", "October", "November", "December"], fontsize=9)

ax.set_xlabel("Month of the Week")

ax.set_ylabel("Total Crimes Reported")

# Adding Values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine()
fig, ax = plt.subplots()

# Plotting crimes occurred by month

sns.barplot(x=crime["Month Occurred"].value_counts().index, y=crime["Month Occurred"].value_counts())

# Axes

ax.set_title("Total Crimes Occurred by Month")

ax.set_xticklabels(["January", "February", "March", "April", "May", "June",

                    "July", "August", "September", "October", "November", "December"], fontsize=9)

ax.set_xlabel("Month of the Week")

ax.set_ylabel("Total Crimes Occurred")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine()
# Making a new dataframe

df2 = pd.DataFrame({

    "Month" : list(crime["Month Reported"].value_counts().index),

    "Crime Reported" : list(crime["Month Reported"].value_counts()),

    "Crime Occurred" : list(crime["Month Occurred"].value_counts())

})

monrepocc = df2.set_index("Month").stack().reset_index().rename(columns={"level_1" : "Variable", 0 : "Crime"})
fig, ax = plt.subplots()

# Plotting side by side crime rep and occ by day

sns.barplot(x = "Month", y = "Crime", hue = "Variable", data=monrepocc, ax=ax)

# Axes

ax.set_title("Crime Reported and Occured by Month")

ax.set_xticklabels(["January", "February", "March", "April", "May", "June",

                    "July", "August", "September", "October", "November", "December"], fontsize=9)

ax.set_ylabel("Crime")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=8, color='black', ha='center', va='bottom')

sns.despine(fig)
fig, ax = plt.subplots()

# Plotting crimes reported by year

plt.plot(crime["Year Reported"].value_counts().sort_index().index, crime["Year Reported"].value_counts().sort_index())

# Axes

ax.set_title("Total Crimes Reported by Year")

ax.set_xlabel("Year")

ax.set_ylabel("Total Crimes Reported")

sns.despine()
fig, ax = plt.subplots()

# Plotting crimes occured by year

plt.plot(crime["Year Occurred"].value_counts().sort_index().index, crime["Year Occurred"].value_counts().sort_index())

# AXes

ax.set_title("Total Crimes Occurred by Year")

ax.set_xlabel("Year")

ax.set_ylabel("Total Crimes Occurred")

sns.despine()
# Making a new dataframe

df3 = pd.DataFrame({

    "Year" : list(crime["Year Reported"].value_counts().index),

    "Crime Reported" : list(crime["Year Reported"].value_counts()),

    "Crime Occurred" : list(crime["Year Occurred"].value_counts())

})

yearrepocc = df3.set_index("Year").stack().reset_index().rename(columns={"level_1" : "Variable", 0 : "Crime"})
fig, ax = plt.subplots()

# Plotting side by side crime rep and occ by day

sns.barplot(x = "Year", y = "Crime", hue = "Variable", data=yearrepocc, ax=ax)

# Axes

ax.set_title("Crime Reported and Occured by Year")

ax.set_ylabel("Crime")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine(fig)
fig, ax = plt.subplots()

# Plot crimes reported over time

plt.plot(crime["Date Reported"].value_counts().sort_index().index, crime["Date Reported"].value_counts().sort_index())

# Axes

ax.set_title("Total Crimes Reported Since 2010")

ax.set_xlabel("Time")

ax.set_ylabel("Total Crimes Reported")

sns.despine()
fig, ax = plt.subplots()

# Plot crimes occurred over time

plt.plot(crime["Date Occurred"].value_counts().sort_index().index, crime["Date Occurred"].value_counts().sort_index())

# Axes

ax.set_title("Total Crimes Occurred Since 2010")

ax.set_xlabel("Time")

ax.set_ylabel("Total Crimes Occurred")

sns.despine()
# Strip the month and the year as string

month_year_rep = [str(m)+"/"+str(y) for m,y in zip(crime["Month Reported"], crime["Year Reported"])]

# Make them date time objects as a list

month_year_rep_formatted = [dt.datetime.strptime(d, "%m/%Y") for d in month_year_rep]

# Turn the list of datetime month and year into a new column

crime["Month Year Rep"] = np.array(month_year_rep_formatted)
fig, ax = plt.subplots()

# Plot crimes reported over months and years

plt.plot(crime["Month Year Rep"].value_counts().sort_index().index, crime["Month Year Rep"].value_counts().sort_index())

# Axes

ax.set_title("Total Crimes Reported Grouped by Month Over the Years")

ax.set_ylim(12500, 22500)

ax.set_xlabel("Time")

ax.set_ylabel("Total Crimes Reported")

sns.despine()
# Strip the month and the year as string

month_year_occ = [str(m)+"/"+str(y) for m,y in zip(crime["Month Occurred"], crime["Year Occurred"])]

# Make them date time objects as a list

month_year_occ_formatted = [dt.datetime.strptime(d, "%m/%Y") for d in month_year_occ]

# Turn the list of datetime month and year into a new column

crime["Month Year Occ"] = np.array(month_year_occ_formatted)
fig, ax = plt.subplots()

# Plot crimes occurred over months and years

plt.plot(crime["Month Year Occ"].value_counts().sort_index().index, crime["Month Year Occ"].value_counts().sort_index())

# Axes

ax.set_title("Total Crimes Occurred Grouped by Month Over the Years")

ax.set_ylim(12500, 22500)

ax.set_xlabel("Time")

ax.set_ylabel("Total Crimes Occurred")

sns.despine()
def makemil(time):

    ntime = ""

    if len(str(time)) == 1:

        ntime = "000" + str(time)

    if len(str(time)) == 2:

        ntime = "00" + str(time)

    if len(str(time)) == 3:

        ntime = "0" + str(time)

    if len(str(time)) == 4:

        ntime = str(time)

    return ntime



def returnhour(miltime):

    return miltime[:2]
# Formatting to 4 char string

crime["Time Occurred"] = crime["Time Occurred"].apply(makemil)
# Formatting to int so it can be sorted

crime["Time Occurred Int"] = crime["Time Occurred"].apply(int)
fig, ax = plt.subplots()

# Plot crime throughout a single day hours

sns.distplot(crime["Time Occurred Int"])

# Axes

ax.set_title("Crime Throughout the Day")

ax.set_xlabel("Time")

ax.set_ylabel("Total Crimes Occurring")

sns.despine()
# Extracting the hour out from time

crime["Hour Occurred"] = crime["Time Occurred"].apply(returnhour)
fig, ax = plt.subplots()

# Crime through the hours

plt.plot(crime["Hour Occurred"].value_counts().sort_index().index, crime["Hour Occurred"].value_counts().sort_index())

# Axes

ax.set_title("Crime Throughout the Day by the Hour")

plt.xticks(range(24))

plt.xlim(0,23)

plt.xlabel("Time (24 hour format)")

plt.ylabel("Total Crimes Occurring")

sns.despine()
fig, ax = plt.subplots()

# Plotting crimes by neighborhood area

sns.barplot(crime["Area Name"].value_counts().index, crime["Area Name"].value_counts(), color="gray", ax=ax)

# Axes

ax.set_title("Crimes by Area")

ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)

ax.set_xlabel("Area Name")

ax.set_ylabel("Total Crimes Occurring")

# Adding Values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine()
# Tally total of top 20 crimes

crime["Crime Code Description"].value_counts().head(20)
# Since the number of unique crimes are more than 100, plot top 20

fig, ax = plt.subplots()

# Plotting crimes by type

sns.barplot(y=crime["Crime Code Description"].value_counts().index[0:20], 

                 x=crime["Crime Code Description"].value_counts().head(20), ax=ax)

# Axes

ax.set_title("Types of Crime Committed")

ax.set_xlabel("Crime Count")

ax.set_ylabel("Crime Committed")

sns.despine()
# Splitting the MO codes per whitespace

MO_list = []

for item in crime["MO Codes"].dropna():

    MO_list.append(str(item).split())
# Making a new DataFrame for MO Codes

tempo_MO_split = []

for i in MO_list:

    for j in i:

        tempo_MO_split.append("MO "+j)

        

tempo_MO_split = np.array(tempo_MO_split)



pre_MO_df = [["","MO Codes"]]

for i in range(len(tempo_MO_split)):

    pre_MO_df.append([i, tempo_MO_split[i]])

    

pre_MO_data = np.array(pre_MO_df)



post_MO_df = pd.DataFrame(data=pre_MO_data[1:,1:],

                  index=pre_MO_data[1:,0],

                  columns=pre_MO_data[0,1:])
fig, ax = plt.subplots()

# Looking into crime by MO

sns.barplot(post_MO_df["MO Codes"].value_counts().index[:20], post_MO_df["MO Codes"].value_counts().head(20), ax=ax)

# Axes

ax.set_title("Crime by MO")

ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)

ax.set_xlabel("MO Code")

ax.set_ylabel("Total Crimes Occurring")

sns.despine()
# Basic stats about Victim Age

crime["Victim Age"].describe()[1:]
fig, ax = plt.subplots()

# Plotting victim's age distribution

sns.distplot(crime["Victim Age"].dropna(), bins=90, ax=ax)

# Axes

ax.set_title("Victim Age")

ax.set_xlabel("Age")

ax.set_ylabel("Probability of Being a Victim")

sns.despine()
# Victim Sex Demographics

crime["Victim Sex"].value_counts()                                
fig, ax = plt.subplots(figsize=(6,4))

# Plotting piechart of victim sex

ax.pie(crime["Victim Sex"].value_counts()[:3],labels=crime["Victim Sex"].value_counts()[:3].index, startangle=90, 

       explode=(0.15,0.15,.15), autopct="%0.1f%%", colors=("blue","mediumvioletred","gray"))

fig.tight_layout()
# Changing the abbreviations to the whole description

Victims_bg = {

    "A": "Other Asian",

    "B": "Black",

    "C": "Chinese",

    "D": "Cambodian",

    "F": "Filipino",

    "G": "Guamanian",

    "H": "Hispanic/Latin/Mexican",

    "I": "American Indian/Alaskan Native",

    "J": "Japanese",

    "K": "Korean",

    "L": "Laotian",

    "O": "Other",

    "P": "Pacific Islander",

    "S": "Samoan",

    "U": "Hawaiian",

    "V": "Vietnamese",

    "W": "White",

    "X": "Unknown",

    "Z": "Asian Indian"

}

crime["Victim Descent"] = crime["Victim Descent"].map(Victims_bg)
fig, ax = plt.subplots()

# Plotting by victim gescent generally

sns.barplot(y=crime["Victim Descent"].value_counts().index, x=crime["Victim Descent"].value_counts(), ax=ax)

# Axes

ax.set_title("Victim Descent")

ax.set_xlabel("Total Crime Victims")

ax.set_ylabel("Victim Descent")

sns.despine()
# Previewing the total tally

crime["Premise Description"].value_counts()
# We will only be looking at the top 20 premises

fig, ax = plt.subplots()

# Plotting top 20 premises

sns.barplot(y=crime["Premise Description"].value_counts().head(20).index,

                 x=crime["Premise Description"].value_counts().head(20), ax=ax)

# Axes

ax.set_title("Crime Premises")

ax.set_xlabel("Total Crimes Occurred")

ax.set_ylabel("Premise")

sns.despine()
# Number of Na values

missvals = crime["Weapon Description"].isnull().sum()

print("There are {} missing values".format(missvals))
crime["Weapon Description"].value_counts().head(10)
fig, ax = plt.subplots()

# Plotting weapons used

sns.barplot(y=crime["Weapon Description"].value_counts().head(20).index,

                 x=crime["Weapon Description"].value_counts().head(20), ax=ax)

# Axes

ax.set_title("Weapon Used")

ax.set_xlabel("Total Crimes Occurred")

ax.set_ylabel("Weapon")

sns.despine()
fig, ax = plt.subplots()

# Plotting the arrest status

sns.barplot(y=crime["Status Description"].value_counts().index,

                 x=crime["Status Description"].value_counts(), ax=ax)

# Axes

ax.set_title("Status")

ax.set_xlabel("Total Crimes Occurred")

ax.set_ylabel("Arrest Status")

sns.despine()
# Making a new dataframe

CC_list = []

for i in range(1,5):

    for item in crime["Crime Code "+str(i)].dropna():

        CC_list.append("Code " +str(int(item)))

        

tempo_CC = np.array(CC_list)



CC_df = pd.DataFrame(tempo_CC)

CC_df = CC_df.rename(columns = {0 : "Crime Codes"})
fig, ax = plt.subplots()

# Plotting crime codes

ax = sns.barplot(CC_df["Crime Codes"].value_counts().head(20).index, CC_df["Crime Codes"].value_counts().head(20))

# Axes

ax.set_title("Crime Codes")

ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)

ax.set_xlabel("Crime Codes")

ax.set_ylabel("Total Crimes Occurring")

sns.despine()
# Dropping missing values and an out of place coordinate of (0, 0) and creating coordinates

crime_lats = []

crime_lons = []

for locs in crime["Location "].dropna():

    if locs.split(",")[0] != "(0" and locs.split(",")[1] != " 0)":

        crime_lats.append(locs.split(",")[0][1:])

        crime_lons.append(locs.split(",")[1][1:-2])

crime_lats = list(map(float, crime_lats))        

crime_lons = list(map(float, crime_lons))        
# Sets center of the map

# gmap = gmplot.GoogleMapPlotter(34.0522, -118.2437, 11)

# Plots the heatmap

# gmap.heatmap(crime_lats, crime_lons)
# Saves it

# gmap.draw("crime heatmap.html")
# Removing Entries for X and H and - (by elimination)

crime["Victim Gender"] = crime["Victim Sex"][crime["Victim Sex"] != "X"]

crime["Victim Gender"] = crime["Victim Gender"][crime["Victim Gender"] != "H"]

crime["Victim Gender"] = crime["Victim Gender"][crime["Victim Gender"] != "-"]
# Combining two columns into a dataframe

cc_vg = crime[["Crime Code Description", "Victim Gender"]]

# Dropping null values

cc_vg = cc_vg[pd.notnull(cc_vg["Victim Gender"])]
# Saving top 10 crimes

crimetop10 = cc_vg["Crime Code Description"].value_counts().head(10).index

# Choosing data that is included in the top 10 crimes (by selection)

crimecc = cc_vg.loc[cc_vg["Crime Code Description"].isin(crimetop10)]
# Group by Crime Code Description and Victim Gender

cc_gender = crimecc.groupby(["Crime Code Description", "Victim Gender"]).size().reset_index(name="Count")

cc_gender
# Factorplot Crime and Gender based on count

ax = sns.factorplot(x="Crime Code Description", hue="Victim Gender", kind="count", data=crimecc, size=5, aspect=3, 

                    palette=["red", "blue"])

# Axes

plt.title("Victim Gender by Crime")

ax.set_xticklabels(rotation=-90)

ax.set_xlabels("Victim of Crime")

ax.set_ylabels("Count")

sns.despine()
# Filtering only rows with Hand guns or Semi-automatic pistols (by equal to)

crime["Guns Only"] = crime["Weapon Description"][(crime["Weapon Description"] == "HAND GUN") | 

                                                 (crime["Weapon Description"] == "SEMI-AUTOMATIC PISTOL")]
# Group by Guns Only and Hour Occurred

cc_gender = crime.groupby(["Hour Occurred", "Guns Only"]).size().reset_index(name="Count")

cc_gender.tail(6)
# Factorplot Crimes by Weapon used

ax = sns.factorplot(x="Hour Occurred", hue="Guns Only", kind="count", data=crime, size=5, aspect=3)

# Axes

plt.title("Crime By Weapon")

ax.set_xlabels("Hours")

ax.set_ylabels("Count")

sns.despine()
crime["Premise Description"].value_counts().head(5)
# Filtering crimes that happen by top 5 by (by equal to)

crime["Public Premise"] = crime["Premise Description"][(crime["Premise Description"] == "STREET") |

                                                       (crime["Premise Description"] == "SINGLE FAMILY DWELLING") | 

                                                       (crime["Premise Description"] == "MULTI-UNIT DWELLING (APARTMENT, DUPLEX, ETC)") |

                                                       (crime["Premise Description"] == "PARKING LOT") |

                                                       (crime["Premise Description"] == "SIDEWALK")]
# Factorplot Crime and Gender based on count

ax = sns.factorplot(x="Public Premise", hue="Hour Occurred", kind="count", data=crime, size=5, aspect=3)

# Axes

plt.title("Crime in Premises by Hour")

ax.set_xticklabels(rotation=-60)

ax.set_xlabels("Hours")

ax.set_ylabels("Count")

sns.despine()
# Saving top 10 types of crime

crimetoptype = crime["Crime Code Description"].value_counts().head(16).index

# Choosing data that is included in the top 10 types of crimes (by selection)

crimepremtype = crime.loc[crime["Crime Code Description"].isin(crimetoptype)]
# Type of Crime by Location

sns.set_palette("hls", n_colors=16)

ax = sns.factorplot(x="Public Premise", hue="Crime Code Description", kind="count", data=crimepremtype, size=5, aspect=3)

# Axes

plt.title("Crime by Location")

ax.set_xticklabels(rotation=-60)

ax.set_xlabels("Hours")

ax.set_ylabels("Count")

sns.despine()
# Filtering the dataset with juvenile arrests (by selection)

crimejuv = crime.loc[crime["Status Description"].isin(["Juv Arrest"])]
crimejuv.shape
# Resetting color and size from above

sns.set()

rcParams['figure.figsize'] = (12,8)
# Juvenile Victims Age Distribution

fig, ax = plt.subplots()

ax = sns.distplot(crimejuv["Victim Age"].dropna(), bins=90)

sns.set_style("whitegrid")

ax.set_title("Victim Age")

ax.set_ylabel("Probability of Being a Victim with a Juvenile Offender")

ax.set_xlabel("Age")

sns.despine()
# Plotting top 10 types of crime committed by a juvenile.

fig, ax = plt.subplots()

# Axes

ax = sns.barplot(y=crimejuv["Crime Code Description"].value_counts().index[0:10], 

                 x=crimejuv["Crime Code Description"].value_counts().head(10))

ax.set_title("Types of Crime Committed by a Juvenile")

ax.set_xlabel("Crime Count")

ax.set_ylabel("Crime Committed")

sns.despine()
fig, ax = plt.subplots()

# Top 10 premises of crime by a juvie

ax = sns.barplot(y=crimejuv["Premise Description"].value_counts().head(10).index,

                 x=crimejuv["Premise Description"].value_counts().head(10))

# Axes

sns.set_style("whitegrid")

ax.set_title("Crime Premises by Juvies")

ax.set_xlabel("Total Crimes Occurred")

ax.set_ylabel("Premise")

sns.despine()
fig, ax = plt.subplots()

# Plotting crimes occurred by day

sns.barplot(x=crimejuv["Day Occurred"].value_counts().index, y=crimejuv["Day Occurred"].value_counts())

# Axes

ax.set_title("Crimes by Juvie by Day")

ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], fontsize=9)

ax.set_xlabel("Day of the Week")

ax.set_ylabel("Total Crimes Occurred")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=9, color='black', ha='center', va='bottom')

sns.despine()
fig, ax = plt.subplots()

# Plotting crime through the hours

ax = plt.plot(crimejuv["Hour Occurred"].value_counts().sort_index().index, crimejuv["Hour Occurred"].value_counts().sort_index())

# Axes

plt.title("Crimes by Juvie by the Hour")

plt.xticks(range(24))

plt.xlim(0,23)

plt.xlabel("Time (24 hour format)")

plt.ylabel("Total Crimes Occurring")

sns.despine()
# Taking the max and min value of crimes occuring by Area (by equal to)

crime["Dang and Safe Area"] = crime["Area Name"][(crime["Area Name"] == "77th Street") |

                                           (crime["Area Name"] == "Hollenbeck")]
# Grouping by the counts

areahour = crime.groupby(["Dang and Safe Area", "Hour Occurred"]).size().reset_index(name="Count")

areahour.head(10)
fig, ax = plt.subplots()

# Plotting crimes in the most dangerous and safest area by hour

sns.pointplot(x="Hour Occurred", y="Count", hue="Dang and Safe Area", data=areahour, ax=ax)

ax.set_title("Crimes in the Most Dangerous and Safest Area")

ax.set_ylabel("Count")

sns.despine()
# Filtering data for only 12pm (by selection)

crimenoon = crime.loc[crime["Hour Occurred"].isin(["12"])]



# Taking only top 6 crimes

top6crimes = crimenoon["Crime Code Description"].value_counts().head(6).index

crimenoon = crimenoon.loc[crimenoon["Crime Code Description"].isin(top6crimes)]



# Taking only top 6 premises

top6premises = crimenoon["Premise Description"].value_counts().head(6).index

crimenoon = crimenoon.loc[crimenoon["Premise Description"].isin(top6premises)]
print("The shape is {}".format(crimenoon.shape))
ccpremnoon = crimenoon.groupby(["Crime Code Description", "Premise Description"]).size().reset_index(name="Count")
ccpremnoon.head()
ccpremise = ccpremnoon.pivot("Crime Code Description", "Premise Description", "Count")
# Prepping data for heatmap

ccpremise = ccpremnoon.pivot("Crime Code Description", "Premise Description", "Count")



# Draw a heatmap with the numeric values in each cell

fig, ax = plt.subplots()

sns.heatmap(ccpremise, annot=True, linewidths=.5, ax=ax, fmt="2g")

fig.tight_layout()
# Get list of top 6 crimes

noidentheft = list(crimenoon["Crime Code Description"].value_counts().head(6).index)



# Remove identitfy theft since it doesn't really matter with time

try:

    noidentheft.pop(noidentheft.index("THEFT OF IDENTITY"))

except:

    print("Can't find THEFT OF IDENTITY")
# Make new dataset without identity theft

crimenoon2 = crimenoon.loc[crimenoon["Crime Code Description"].isin(noidentheft)]
# Groupby crime code and premise

ccpremnoon2 = crimenoon2.groupby(["Crime Code Description", "Premise Description"]).size().reset_index(name="Count")
# Prepping data for heatmap

ccpremise2 = ccpremnoon2.pivot("Crime Code Description", "Premise Description", "Count")



# Draw a heatmap with the numeric values in each cell

fig, ax = plt.subplots()

sns.heatmap(ccpremise2, annot=True, linewidths=.5, ax=ax, fmt="2g")

fig.tight_layout()
# Filter data with only Identity Theft crimes

identheftvic = crime[crime["Crime Code Description"] == "THEFT OF IDENTITY"]
# Create subset with victim gender and age, then drop Na Values

identheftvic = identheftvic[["Victim Gender", "Victim Age"]]

identheftvic = identheftvic.dropna()
# Plot victims by gender and age

sns.boxplot(y="Victim Age", x="Victim Gender", data=identheftvic, palette={"M": "b", "F": "mediumvioletred"})

# Axes

plt.title("Victims of Identity Theft")
# Filtering 2017 out of the dataframe

crimeno17 = crime.loc[crime["Year Occurred"].isin(range(2010, 2017))]
# Making a new dataframe

df4 = pd.DataFrame({

    'Month': list(crimeno17["Month Reported"].value_counts().index),

    'Crime Reported': list(crimeno17["Month Reported"].value_counts()),

    'Crime Occurred': list(crimeno17["Month Occurred"].value_counts())

})

monrepoccclean = df4.set_index("Month").stack().reset_index().rename(columns={"level_1" : "Variable", 0 : "Crime"})
fig, ax = plt.subplots(figsize=(14,11))

# Plotting side by side crime rep and occ by day

sns.barplot(x = "Month", y = "Crime", hue = "Variable", data=monrepoccclean, ax=ax)

# Axes

ax.set_title("Crime Reported and Occured by Month Excluding 2017")

ax.set_xticklabels(["January", "February", "March", "April", "May", "June",

                    "July", "August", "September", "October", "November", "December"], fontsize=9)

ax.set_ylabel("Crime")

# Adding values

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % round(int(p.get_height()), -2), 

            fontsize=8, color='black', ha='center', va='bottom')

sns.despine(fig)