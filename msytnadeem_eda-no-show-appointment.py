# importing packages and libraries and matplotlib for visualization

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")



%matplotlib inline
#importing CSV into data frame.

df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")

rows, columns = df.shape

print("The data frame has "+ str(rows) +" rows and " + str(columns) + " columns")
#browse sample of data values and formats of each feature. 

df.head()
#browse data frame columns data types

df.info()
#print out statistical details of the numeric data.

df.describe()
#check number of not showing up patinets to an appointment on scale of 100

#group by no-show column

no_show_percentage = pd.DataFrame(df.groupby(["No-show"])["PatientId"].count())

#calculate percentage of show up and no show and store it in column No-Show

no_show_percentage["No-show"] = no_show_percentage["PatientId"] / sum(no_show_percentage["PatientId"]) * 100

no_show_percentage.drop(columns="PatientId", inplace=True)

#plot the dataframe 

no_show_percentage.plot.bar(figsize=(10,5))

plt.ylim(top=100)

plt.title("Medical Appointments",{'fontsize': 20},pad=20)

plt.xlabel("Appointment Status")

plt.xticks(np.arange(2), ('Show-Up', 'No-Show'), rotation=0)

plt.legend(["Appointment Status Rate"])

#checking the age distripution

df["Age"].describe()
#Check number of duplicated records in the data frame. 

print("Number of duplicate recrods: " + str(sum(df.duplicated())))
#assure gender has only two unique values

df["Gender"].unique()
#check neighbourhood unique list

df["Neighbourhood"].unique()
#check number of wrong values of handcap that exceeds a value of 1

print("Number of wrong handicap values: " + str(df.query("Handcap > 1")["Handcap"].count()))
#check scheduled Day and Appointment Day description

df[["ScheduledDay","AppointmentDay"]].describe()
#new column names for columns requires word seperation with underscore or spelling mistakes

columnNames = {

            "PatientId":"patient_id", 

            "AppointmentID":"appointment_id",

            "ScheduledDay":"scheduled_day",

            "AppointmentDay":"appointment_day",

            "Hipertension":"hypertension",

            "Handcap":"handicap",

            "No-show":"no_show"

            }

df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")

#rename columns

df.rename(columns=columnNames, inplace=True)

#lower case all columns names

df.columns = df.columns.str.lower()

df.dtypes
#converting columns scheduled_day and appointment_day to datetime64

df['scheduled_day'] = pd.to_datetime(df['scheduled_day'], format="%Y-%m-%d %H:%M:%S")

df['appointment_day'] = pd.to_datetime(df['appointment_day'], format="%Y-%m-%d %H:%M:%S")

#confirm new data types, as well check no null values was generated because of the transition.

df[["scheduled_day","appointment_day"]].info()
#look at the description of the date time columns

df[["scheduled_day","appointment_day"]].describe()
df['appointment_day'].dt.time.describe()
#schedule_days = appointment_day - scheduled_day

df["schedule_days"] = (df["appointment_day"] - df["scheduled_day"]).dt.days



df["schedule_days"].describe()
#check ditribuption of the data for schedule_days with negative values

ax1 = plt.subplot(1,2,1)

df.query("schedule_days < 0")["schedule_days"].hist(bins=30,figsize=(13,4))

ax1.set_title("Days of Scheduling Before Appointment (All Negative)")

ax1.set_xlabel("Delta Days (Appointment - Schedule)")

ax1.legend(["Number of Appointments"])

#check ditribuption of the data for schedule_days below that -1

ax2 = plt.subplot(1,2,2)

df.query("schedule_days < -1")["schedule_days"].hist(bins=30, figsize=(13,4))

ax2.set_title("Days of Scheduling Before Appointment (Below -1)")

ax2.set_xlabel("Delta Days (Appointment - Schedule)")

ax2.legend(["Number of Appointments"])

plt.tight_layout()

#show the appointment date and schedule dates of appointments was scheulded 1 day after

df.query("schedule_days  == -1")[["schedule_days","scheduled_day", "appointment_day"]].head(10)
#apply the difference between scheduled day and appointment day with date only without time.

df["schedule_days"] = (df["appointment_day"].dt.date - df["scheduled_day"].dt.date).dt.days

#plot histogram of the negative schedule_days to confirm our results

df.query("schedule_days < 0")["schedule_days"].hist(bins=30,figsize=(10,5))

plt.title("Days of Scheduling Before Appointment (All Negative)")

plt.xlabel("Delta Days (Appointment - Schedule)")

plt.legend(["Number of Appointments"])
#filter our appointments which was scheduled after its day.

df = df.query("schedule_days >= 0")

#look at schedule days description

df["schedule_days"].describe()
#classifier function that returns the schedule_days group

def schedule_days_classifier(schedule_days):

    if schedule_days == 0:

        return "0 Days"

    elif schedule_days >= 1 and schedule_days < 5:

        return "1-4 Days"

    elif schedule_days >= 5 and schedule_days < 16:

        return "5-15 Days"

    else:

        return "16+ Days"

#apply classifier and store it in schedule_days    

df["schedule_days"] = df["schedule_days"].apply(schedule_days_classifier)
#static order of the schedule_days group

schedule_days_order = ["0 Days", "1-4 Days", "5-15 Days", "16+ Days"]

#group and plot

df.groupby(["schedule_days"]).count()[["no_show"]].loc[schedule_days_order].plot.bar()
#drop scheduled_day column

df.drop(columns=["scheduled_day"], inplace=True)

df.columns
#print appointment unique years

print("Appointments occured in years of: " + np.array2string(df['appointment_day'].dt.year.unique()))

#print appointment unique months

print("Appointments occured in months of: " + np.array2string(df['appointment_day'].dt.month.unique()))
#list of week_days

week_day_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#week day classifier get day and returns the day name.

def week_day_classifier(day):

    return week_day_list[day];

#apply classifier and store it in week_day

df["week_day"] = df["appointment_day"].dt.weekday.apply(week_day_classifier)

#print out week day sample data

df[[ "appointment_id", "week_day"]].head()
#drop appointment_day column

df.drop(columns=["appointment_day"], inplace=True)

df.columns
# get patient id for patients has negative age

df.query("age < 0")["patient_id"].astype(str).str[:-2]
#find other records for patient id =465943158731293

df.query("patient_id == '465943158731293'")
#filter our records with negative age.

df = df.query("age >= 0")
#let us see the age distribution

df["age"].describe()
#age classifier function

def age_classifier(age):

    if age >= 0 and age <18:

        return "Kids"

    elif age >= 18 and age < 37:

        return "Adults"

    elif age >= 37 and age < 55:

        return "Matures"

    else:

        return "Elders"

#apply age classifier and store into age_group    

df["age_group"] = df["age"].apply(age_classifier)

#drop age column

df.drop(columns=["age"], inplace=True)

#print out patinet information smaple data

df[["patient_id", "gender", "age_group"]].head()
#make handicap value above 1 to be equal to 1

df.loc[df.handicap >1 , 'handicap'] =1

df[["handicap"]].describe()
#convert scholarship, hypertension, diabetes, alcoholism, handicap, sms_received to boolean

df["scholarship"] = df["scholarship"].astype(bool)

df["hypertension"] = df["hypertension"].astype(bool)

df["diabetes"] = df["diabetes"].astype(bool)

df["alcoholism"] = df["alcoholism"].astype(bool)

df["handicap"] = df["handicap"].astype(bool)

df["sms_received"] = df["sms_received"].astype(bool)

df[["scholarship","hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]].dtypes
#Convert no_show column from Yes/No into True/False

def noshow_to_boolean(status):

    if status == 'No':

        return False

    else:

        return True

    

df["no_show"] = df["no_show"].apply(noshow_to_boolean)

df[["no_show"]].dtypes
#dropping patient_id and appointment_id

df.drop(columns=['patient_id', 'appointment_id'], inplace=True)

df.columns
#order data set columns

df = df[['gender', 'age_group', 'neighbourhood','scholarship','hypertension', 'diabetes',

       'alcoholism', 'handicap', 'week_day', 'schedule_days', 'sms_received','no_show']]

#store data frame into cleaned csv

df.to_csv('no_show_cleaned.csv', index=False)
#load cleaned Data Frame

df_clean = pd.read_csv('no_show_cleaned.csv')
#group by gender

gender_all = df_clean.groupby(["gender"])[["gender"]].count()

#Calculate percentage of appointments per gender

gender_all.columns = ["Gender Rate"]

gender_all["Gender Rate"] = gender_all["Gender Rate"] / sum(gender_all["Gender Rate"]) * 100

gender_all.reset_index(inplace=True)
#group by gender and no_show

gender_by_no_show = df_clean.groupby(["gender", "no_show"])[["gender"]].count()

#calculate percentage of appointment per gender per appointment show up status

gender_by_no_show.columns = ["no_show_count"]

gender_by_no_show.reset_index(inplace=True)

gender_by_no_show.columns = ["Gender", "No Show Status", "No Show Count"]

gender_by_no_show =  pd.DataFrame(gender_by_no_show.groupby(["Gender","No Show Status"])["No Show Count"].sum() / gender_by_no_show.groupby(["Gender"])["No Show Count"].sum() * 100)

gender_by_no_show = gender_by_no_show.unstack()
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Appointment per Gender', fontsize=16)



#plot percentage of appointments per gender

gender_all.plot.bar(ax=axs[0])

axs[0].set_xticklabels(("Female","Male"), rotation=0)

axs[0].set_ylim(top=100)

axs[0].set_xlabel("Gender")

axs[0].legend(["% of Appointments per Gender"])



#plot percentage of appointment per gender per appointment show up status

gender_by_no_show.plot.bar(ax=axs[1], stacked=True)

axs[1].set_xticklabels(("Female","Male"), rotation=0)

axs[1].set_ylim(top=100)

axs[1].set_xlabel("Gender")

axs[1].legend(["Show Up", "No Show"])
#group by age group

age_group_all = df_clean.groupby(["age_group"])[["age_group"]].count()

#calculate percentage of appointments per age group 

age_group_all.columns = ["Age Group Rate"]

age_group_all["Age Group Rate"] = age_group_all["Age Group Rate"] / sum(age_group_all["Age Group Rate"]) * 100

age_group_all.reset_index(inplace=True)
#group by age group per appointment show up status

age_group_no_show = df_clean.groupby(["age_group", "no_show"])[["age_group"]].count()

#calculate percentage of appointments per age group per appointment show up status

age_group_no_show.columns = ["age_group_count"]

age_group_no_show.reset_index(inplace=True)

age_group_no_show.columns = ["Age Group", "No Show Status", "No Show Count"]

age_group_no_show = pd.DataFrame(age_group_no_show.groupby(["Age Group","No Show Status"])["No Show Count"].sum() / age_group_no_show.groupby(["Age Group"])["No Show Count"].sum() * 100)

age_group_no_show = age_group_no_show.unstack()
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Appointment per Age Group', fontsize=16)



#plot percentage of appointments per age group 

age_group_all.plot.bar(ax=axs[0])

axs[0].set_xticklabels(("Under 18","19 to 37", "38 to 55", "Above 55"), rotation=0)

axs[0].set_ylim(top=30)

axs[0].set_xlabel("Age Group")

axs[0].legend(["% of Appointments per Age Group"])

#plot percentage of appointments per age group per appointment show up status

age_group_no_show.plot.bar(ax=axs[1], stacked=True)

axs[1].set_xticklabels(("Under 18","19 to 37", "38 to 55", "Above 55"), rotation=0)

axs[1].set_ylim(top=100)

axs[1].set_xlabel("Age Group")

axs[1].legend(["Show Up", "No Show"])
#group by schedule days group

schedule_days_all = df_clean.groupby(["schedule_days"])[["schedule_days"]].count().loc[schedule_days_order]

#calculate percentage of appointments per schedule day groups

schedule_days_all.columns = ["Schedule Days Rate"]

schedule_days_all["Schedule Days Rate"] = schedule_days_all["Schedule Days Rate"] / sum(schedule_days_all["Schedule Days Rate"]) * 100

schedule_days_all.reset_index(inplace=True)
#group by schedule days group and appointment show up status

schedule_days_no_show = df_clean.groupby(["schedule_days", "no_show"])[["schedule_days"]].count()

#calcualte percentage of appointments per schedule day group per appointment show up status

schedule_days_no_show.columns = ["schedule_days_count"]

schedule_days_no_show.reset_index(inplace=True)

schedule_days_no_show.columns = ["Schedule Days", "No Show Status", "No Show Count"]

schedule_days_no_show = pd.DataFrame(schedule_days_no_show.groupby(["Schedule Days","No Show Status"])["No Show Count"].sum() / schedule_days_no_show.groupby(["Schedule Days"])["No Show Count"].sum() * 100)

schedule_days_no_show = schedule_days_no_show.unstack().loc[schedule_days_order]
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Appointment per Schedule Days', fontsize=16)



#plot percentage of appointments per schedule day groups

schedule_days_all.plot.bar(ax=axs[0])

axs[0].set_xticklabels(schedule_days_order, rotation=0)

axs[0].set_ylim(top=30)

axs[0].set_xlabel("Schedule Days")

axs[0].legend(["% of Appointments per Schedule Days"])



#plot percentage of appointments per schedule day group per appointment show up status

schedule_days_no_show.plot.bar(ax=axs[1], stacked=True)

axs[1].set_xticklabels(schedule_days_order, rotation=0)

axs[1].set_ylim(top=100)

axs[1].set_xlabel("Schedule Days")

axs[1].legend(["Show Up", "No Show"])
#get only show up appointments and group by schedule days per sms_received per appointment show up status

schedule_days_sms_showed_up = df_clean.query("no_show == False").groupby(["schedule_days", "sms_received", "no_show"])[["schedule_days"]].count()

#calcualte the percentage of scheudle days per sms received per appointment show up status

schedule_days_sms_showed_up.columns = ["schedule_days_count"]

schedule_days_sms_showed_up.reset_index(inplace=True)

schedule_days_sms_showed_up.columns = ["Schedule Days","SMS Recieved", "No Show Status", "No Show Count"]

schedule_days_sms_showed_up = pd.DataFrame(schedule_days_sms_showed_up.groupby(["Schedule Days","SMS Recieved","No Show Status"])["No Show Count"].sum() / schedule_days_sms_showed_up.groupby(["Schedule Days"])["No Show Count"].sum() * 100)

#unstack twice the data

schedule_days_sms_showed_up = schedule_days_sms_showed_up.unstack().unstack().loc[schedule_days_order]
#get only no-show appointments and group by schedule days per sms_received per appointment show up status

schedule_days_sms_no_show = df_clean.query("no_show == True").groupby(["schedule_days", "sms_received", "no_show"])[["schedule_days"]].count()

#calcualte the percentage of scheudle days per sms received per appointment show up status

schedule_days_sms_no_show.columns = ["schedule_days_count"]

schedule_days_sms_no_show.reset_index(inplace=True)

schedule_days_sms_no_show.columns = ["Schedule Days","SMS Recieved", "No Show Status", "No Show Count"]

schedule_days_sms_no_show = pd.DataFrame(schedule_days_sms_no_show.groupby(["Schedule Days","SMS Recieved","No Show Status"])["No Show Count"].sum() / schedule_days_sms_no_show.groupby(["Schedule Days"])["No Show Count"].sum() * 100)

#unstack twice the data

schedule_days_sms_no_show = schedule_days_sms_no_show.unstack().unstack().loc[schedule_days_order]
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('SMS Reminder Affect on Early Scheduling', fontsize=16)



schedule_days_sms_showed_up.plot.bar(ax=axs[0],stacked=True);

axs[0].set_xticklabels(schedule_days_order, rotation=0)

axs[0].set_ylim(top=100)

axs[0].set_title("Show-Up Appointment")

axs[0].set_xlabel("Schedule Days")

axs[0].legend(["SMS Recieved", "SMS Not Recieved"])



schedule_days_sms_no_show.plot.bar(ax=axs[1], stacked=True)

axs[1].set_xticklabels(schedule_days_order, rotation=0)

axs[1].set_ylim(top=100)

axs[1].set_title("No-Show APpointment")

axs[1].set_xlabel("Schedule Days")

axs[1].legend(["SMS Recieved", "SMS Not Recieved"])
#group by week days

weekday_all = df_clean.groupby(["week_day"])[["week_day"]].count()

#calculate percentage of appointment per week day

weekday_all.columns = ["Week Day Rate"]

weekday_all["Week Day Rate"] = weekday_all["Week Day Rate"] / sum(weekday_all["Week Day Rate"]) * 100

#order index column by weekday order

weekday_all = weekday_all.reindex(week_day_list)
#group by week days per appointment show up status

week_day_no_show = df_clean.groupby(["week_day", "no_show"])[["week_day"]].count()

#calculate percentage of appointment per week day per appointment show up status

week_day_no_show.columns = ["week_day_count"]

week_day_no_show.reset_index(inplace=True)

week_day_no_show.columns = ["Week Day", "No Show Status", "No Show Count"]

week_day_no_show = pd.DataFrame(week_day_no_show.groupby(["Week Day","No Show Status"])["No Show Count"].sum() / week_day_no_show.groupby(["Week Day"])["No Show Count"].sum() * 100)

week_day_no_show = week_day_no_show.unstack()

#order index by weekday order

week_day_no_show = week_day_no_show.reindex(week_day_list)
fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Appointments per Weekday', fontsize=16)



#plot of percentage of appointment per week day

weekday_all.plot.bar(ax=axs[0],stacked=True);

axs[0].set_xticklabels(week_day_list, rotation=0)

axs[0].set_ylim(top=30)

axs[0].set_xlabel("Week Days")

axs[0].legend(["% of Appointments per Week Days"])



#plot of percentage of appointment per week day per appointment show up status

week_day_no_show.plot.bar(ax=axs[1], stacked=True)

axs[1].set_xticklabels(week_day_list, rotation=0)

axs[1].set_ylim(top=100)

axs[1].set_xlabel("Week Days")

axs[1].legend(["Show Up", "No Show"])
#group by hypertenstion per appointment show up status

hipertension_no_show = df_clean.groupby(["hypertension", "no_show"])[["no_show"]].count()

#calculate the percentage appointments of hypertentation  per appointment show up status

hipertension_no_show.columns = ["hypertension_count"]

hipertension_no_show.reset_index(inplace=True)

hipertension_no_show.columns = ["Hypertension", "No Show Status", "No Show Count"]

hipertension_no_show = pd.DataFrame(hipertension_no_show.groupby(["Hypertension","No Show Status"])["No Show Count"].sum() / hipertension_no_show.groupby(["Hypertension"])["No Show Count"].sum() * 100)

hipertension_no_show = hipertension_no_show.unstack()
#group by diabetes per appointment show up status

diabetes_no_show = df_clean.groupby(["diabetes", "no_show"])[["no_show"]].count()

#calculate the percentage appointments of diabetes per appointment show up status

diabetes_no_show.columns = ["diabetes_count"]

diabetes_no_show.reset_index(inplace=True)

diabetes_no_show.columns = ["Diabetes", "No Show Status", "No Show Count"]

diabetes_no_show = pd.DataFrame(diabetes_no_show.groupby(["Diabetes","No Show Status"])["No Show Count"].sum() / diabetes_no_show.groupby(["Diabetes"])["No Show Count"].sum() * 100)

diabetes_no_show = diabetes_no_show.unstack()
#group by diabetes per appointment show up status

alcoholism_no_show = df_clean.groupby(["alcoholism", "no_show"])[["no_show"]].count()

#calculate the percentage appointments of alcoholism per appointment show up status

alcoholism_no_show.columns = ["alcoholism_count"]

alcoholism_no_show.reset_index(inplace=True)

alcoholism_no_show.columns = ["Alcoholism", "No Show Status", "No Show Count"]

alcoholism_no_show = pd.DataFrame(alcoholism_no_show.groupby(["Alcoholism","No Show Status"])["No Show Count"].sum() / alcoholism_no_show.groupby(["Alcoholism"])["No Show Count"].sum() * 100)

alcoholism_no_show = alcoholism_no_show.unstack()
#group by handicap per appointment show up status

handcap_no_show = df_clean.groupby(["handicap", "no_show"])[["no_show"]].count()

#calculate the percentage appointments of handicap per appointment show up status

handcap_no_show.columns = ["handicap_count"]

handcap_no_show.reset_index(inplace=True)

handcap_no_show.columns = ["Handicap", "No Show Status", "No Show Count"]

handcap_no_show = pd.DataFrame(handcap_no_show.groupby(["Handicap","No Show Status"])["No Show Count"].sum() / handcap_no_show.groupby(["Handicap"])["No Show Count"].sum() * 100)

handcap_no_show = handcap_no_show.unstack()
fig, axs = plt.subplots(1,4,figsize=(20,5))

fig.suptitle('Halth Status VS No Show', fontsize=16)

#plot hypertenstion per appointment show up status

hipertension_no_show.plot.bar(ax=axs[0],stacked=True);

axs[0].set_xticklabels(("False", "True"),rotation=0)

axs[0].set_ylim(top=100)

axs[0].set_title("Hipertension VS No Show")

axs[0].set_xlabel("Hipertension")

axs[0].legend(["Show Up", "No Show"])



#plot diabetes per appointment show up status

diabetes_no_show.plot.bar(ax=axs[1], stacked=True)

axs[1].set_xticklabels(("False", "True"),rotation=0)

axs[1].set_ylim(top=100)

axs[1].set_title("Diabetes VS No Show")

axs[1].set_xlabel("Diabetes")

axs[1].legend(["Show Up", "No Show"])



#plot alcoholism per appointment show up status

alcoholism_no_show.plot.bar(ax=axs[2],stacked=True);

axs[2].set_xticklabels(("False", "True"),rotation=0)

axs[2].set_ylim(top=100)

axs[2].set_title("Alcoholism VS No Show")

axs[2].set_xlabel("Alcoholism")

axs[2].legend(["Show Up", "No Show"])



#plot handicaped per appointment show up status

handcap_no_show.plot.bar(ax=axs[3], stacked=True)

axs[3].set_xticklabels(("False", "True"), rotation=0)

axs[3].set_ylim(top=100)

axs[3].set_title("Handcap VS No Show")

axs[3].set_xlabel("Handcap")

axs[3].legend(["Show Up", "No Show"])



#group by neighbourhood per appointment show up status.

neighbourhood_all = df_clean.groupby(["neighbourhood", "no_show"])[["no_show"]].count()

neighbourhood_all.columns = ["no_show_count"]

neighbourhood_all.reset_index(inplace=True)

#Calculate percentage appointments per neighborhood per appointment show up status

neighbourhood_all["no_show_rate"] = pd.DataFrame(neighbourhood_all.groupby(["neighbourhood","no_show"])["no_show_count"].sum() / neighbourhood_all.groupby(["neighbourhood"])["no_show_count"].sum() * 100).reset_index()[["no_show_count"]]

neighbourhood_all = neighbourhood_all.groupby(["neighbourhood","no_show"])[["no_show_count", "no_show_rate"]].sum()

neighbourhood_all = neighbourhood_all.unstack()

#for neighbours has all patients showed up or all patients not showed to their appointment, substitute by 0

neighbourhood_all = neighbourhood_all.fillna(0)
#plot hypertenstion per appointment show up status

axs = neighbourhood_all["no_show_count"].sort_values(by=False).plot.bar(stacked=True, figsize=(20,5));

axs.set_xlabel("neighbourhood")

axs.legend(["Show Up", "No Show"])

axs.set_title("Appointment Per All Neigbourhoods", fontsize=16)

#excluding all neigbourhoods which has less than 1000 appointments

neighbourhood_above_1000_visits = neighbourhood_all[neighbourhood_all["no_show_count"][False] + neighbourhood_all["no_show_count"][True] > 1000]
# plot percentage of appointments per neighbourhood per appointment show up status.

axs = neighbourhood_above_1000_visits["no_show_rate"].sort_values(by=False).plot.bar(stacked=True, figsize=(20,5));

axs.set_xlabel("neighbourhood")

axs.legend(["Show Up", "No Show"])

axs.set_title("Appointment Per Neigbourhoods (1000+ Appointments)", fontsize=16)
#group by negibourhoods per scholarships, for only neigbourhoods has more than 1000 appointments. 

neighbourhood_scholarship = df_clean.query(f"neighbourhood in {neighbourhood_above_1000_visits.index.tolist()}").groupby(["neighbourhood", "scholarship"])[["scholarship"]].count()

neighbourhood_scholarship.columns = ["scholarship_count"]

neighbourhood_scholarship.reset_index(inplace=True)

#caclualte scholraship rate per neighbourhoods

neighbourhood_scholarship["scholarship_rate"] = pd.DataFrame(neighbourhood_scholarship.groupby(["neighbourhood","scholarship"])["scholarship_count"].sum() / neighbourhood_scholarship.groupby(["neighbourhood"])["scholarship_count"].sum() * 100).reset_index()[["scholarship_count"]]

neighbourhood_scholarship = neighbourhood_scholarship.groupby(["neighbourhood", "scholarship"])[["scholarship_rate"]].sum()

neighbourhood_scholarship.reset_index(inplace=True)

#find neigbourhood scholarships distribution

neighbourhood_scholarship.query("scholarship == True").describe()
#function to classify negbourhood by the scholraship rate.

def neighbourhood_social_classifier(row):

    x = row["scholarship_rate"]

    if(row["scholarship"] == False):

        x = 100 - x

    if x >= 0.283725 and x < 8.913911:

        return "Class A"

    elif x >= 8.913911 and x < 11.761120:

        return "Class B"

    elif x >= 11.761120 and x < 14.424395:

        return "Class C"

    else:

        return "Class D"

    

#apply classigication of neighbourhoods

neighbourhood_scholarship["neighbourhood_class"] = neighbourhood_scholarship.apply(neighbourhood_social_classifier,axis=1)

neighbourhood_scholarship_class = neighbourhood_scholarship.loc[:,["neighbourhood", "neighbourhood_class"]]

#drop dublicate records

neighbourhood_scholarship_class.drop_duplicates(inplace=True)

#function to get neigbourhood class

def get_neighbourhood_class(value):

    return neighbourhood_scholarship_class.query(f"neighbourhood == '{value}'")["neighbourhood_class"].values[0]



neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits.reset_index()

#Apply classification of neighbourhood

neighbourhood_above_1000_visits_classed["neighbourhood_class"] = neighbourhood_above_1000_visits_classed["neighbourhood"].apply(get_neighbourhood_class)

#group by neigbourhood class

neighbourhood_above_1000_visits_classed.drop(columns=['neighbourhood'], inplace=True, level=0)

neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits_classed.groupby(["neighbourhood_class"]).sum().stack()

neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits_classed[["no_show_count"]]

#caclulate percentage of appointments per neigbourhood class

neighbourhood_above_1000_visits_classed["no_show_rate"] = pd.DataFrame(neighbourhood_above_1000_visits_classed["no_show_count"] / neighbourhood_above_1000_visits_classed.groupby(["neighbourhood_class"])["no_show_count"].sum() * 100)[["no_show_count"]].values

neighbourhood_above_1000_visits_classed = neighbourhood_above_1000_visits_classed.unstack()
fig, axs = plt.subplots(1,2,figsize=(20,7))

fig.suptitle('Appointments per Neighbourhood Class', fontsize=16)



#plot neighbourhood class per appointment show up status.

neighbourhood_above_1000_visits_classed["no_show_count"].plot.bar(ax=axs[0],stacked=True);

axs[0].set_xlabel("Neighbourhood Class")

axs[0].legend(["Show Up", "No Show"])

# plot percentage of appointments class per neighbourhood per appointment show up status.

neighbourhood_above_1000_visits_classed["no_show_rate"].plot.bar(ax=axs[1],stacked=True);

axs[1].set_ylim(top=100)

axs[1].set_xlabel("Neighbourhood Class")

axs[1].legend(["Show Up", "No Show"])