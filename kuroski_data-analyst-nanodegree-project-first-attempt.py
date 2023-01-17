# first let's load our data

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df = pd.read_csv("../input/medicalappointmentnoshown/KaggleV2-May-2016.csv")

df.head(5)
# let's see from which period theese appointments are

df.AppointmentDay.min(), df.AppointmentDay.max()
# then let's see the shape of our data

df.shape
# and get general numeric attributes

df.describe()
# checking column information for missing values and strange types

df.info()
# checking for general data duplicates

df.duplicated().sum(), df.PatientId.duplicated().sum(), df.AppointmentID.duplicated().sum()
# checking all possible values on some columns

print(df.Gender.unique())

print(sorted(df.Age.unique()))

print(sorted(df.Neighbourhood.unique()))

print(df.Scholarship.unique())

print(df.Hipertension.unique())

print(df.Diabetes.unique())

print(df.Alcoholism.unique())

print(df.Handcap.unique())

print(df.SMS_received.unique())

print(df['No-show'].unique())
# let's remove some useless columns

# I think the appointmentID is useless for this analysis

df.drop(['AppointmentID'], axis=1, inplace=True)

df.columns
# renaming all columns to simpler names for our exploration

df.rename(columns={'PatientId': 'patient_id', 'ScheduledDay': 'scheduled_day', 'AppointmentDay': 'appointment_day', 'SMS_received': 'received_sms', 'No-show': 'no_show', 'Handcap': 'handicap' }, inplace=True)

df.rename(columns=lambda x: x.lower(), inplace=True)

df.columns
# formatting the patient_id column as string

df.patient_id = df.patient_id.apply(lambda patient: str(int(patient)))
# formatting the date time 'scheduled_day' and 'appointment_day' columns

# i'm just testing different forms of time conversion here

df.scheduled_day = pd.to_datetime(df.scheduled_day)

df.appointment_day = df.appointment_day.apply(np.datetime64)



df.scheduled_day.head(1), df.appointment_day.head(1)
# formatting the 'no_show' column with lower cases

df.no_show = df.no_show.map({ 'No': 'no', 'Yes': 'yes' })



df.no_show.unique()
# discart the ages bellow zero

df = df.query('age >= 0')

print(sorted(df.age.unique()))
# remove the weird values from handcap variable

df.loc[df.handicap > 1, 'handicap'] = 1

df.handicap.unique()
# creating the first column "appointment_week_day"

df['appointment_week_day'] = df.appointment_day.map(lambda day: day.day_name())

df.appointment_week_day.head()
# creating the second column "appointment_waiting_time"

df["appointment_waiting_days"] = df.appointment_day - df.scheduled_day

df.appointment_waiting_days.head()
# well it seams that some are treated on the same day that they scheduled

# we can prevent that weird value by calculating the the "absolute value" of this column

# and then converting the "time" to "days"

df.appointment_waiting_days = df.appointment_waiting_days.abs().dt.days

df.appointment_waiting_days.head(10)
# let's see how our data looks like after all cleanning

df.head(5)
# first let's re-see our dataset description

df.describe()
# and plot basic histogram charts

df.hist(figsize=(15, 8));
def show_no_show_trend(dataset, attribute, fit_reg = True):

    '''Prints a chart with no_show_rate explanation

    Syntax: show_no_show_trend(dataframe, attribute), where:

        attribute = the string representing the attribute;

        dataframe = the current dataframe;

    '''

    return sns.lmplot(data = dataset, x = attribute, y = 'no_show_rate', fit_reg = fit_reg, legend = True, height=8, aspect=2)    



def show_attribute_statistics(attribute, dataframe, scale = 0.06, sorter = False, verticalLabel = False):

    '''Prints basic statistics from the attribute also plotting the basic chart. 

    Syntax: show_attribute_statistics(dataframe, attribute), where:

        attribute = the string representing the attribute;

        dataframe = the current dataframe;

        scale = what's the scale you want to converto;

        sorter = array representing the sort reindex;

    '''

    

    # grouping by the patients by attribute and see if there is any interesting data related to their no showing

    # also stripping unwanted attributes with crosstab - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.crosstab.html

    dataset = pd.crosstab(index = dataframe[attribute], columns = dataframe.no_show).reindex(sorter).reset_index() if sorter else pd.crosstab(index = dataframe[attribute], columns = dataframe.no_show).reset_index()

    

    # replacing all none values with zero, since it's the count of patients on that categorie

    dataset['no'].fillna(value=0, inplace=True)

    dataset['yes'].fillna(value=0, inplace=True)



    # let's also record the rate of no-showing base on the attribute

    dataset["no_show_rate"] = dataset['yes'] / (dataset['no'] + dataset['yes'])

    dataset.no_show_rate.fillna(value=0.0, inplace=True)



    dataset["no_show_rate_value"] = dataset["no_show_rate"] * 100 

    dataset.no_show_rate_value.fillna(value=0.0, inplace=True)

    

    # plotting our data

    plt.figure(figsize=(30, 10))



    # scale data if needed

    dataset['no'] = dataset['no'] * scale

    dataset['yes'] = dataset['yes'] * scale



    # line chart

    plt.plot(dataset.no_show_rate_value.values, color="r")



    # bar chart

    plt.bar(dataset[attribute].unique(), dataset['no'].values, bottom = dataset['yes'].values)

    plt.bar(dataset[attribute].unique(), dataset['yes'].values)



    # configs

    if (verticalLabel):

        plt.xticks(rotation='vertical')

        

    plt.subplots_adjust(bottom=0.15)

    plt.xlabel(attribute, fontsize=16)

    plt.ylabel(f"amount of patients (scaled 1 to {scale * 100}%)", fontsize=16)

    plt.legend(["not attended rate", "attended", "not attended"], fontsize=14)



    plt.title("amount of patient by no show appointment groupped by %s" % attribute)



    plt.show();

    

    return dataset
age_dataset = show_attribute_statistics("age", df);

show_no_show_trend(age_dataset, "age");
appointment_waiting_days_dataset = show_attribute_statistics("appointment_waiting_days", df)

show_no_show_trend(appointment_waiting_days_dataset, "appointment_waiting_days")
categories = pd.Series(['same day: 0', 'week: 1-7', 'month: 8-30', 'quarter: 31-90', 'semester: 91-180', 'a lot of time: >180'])

df['waiting_days_categories'] = pd.cut(df.appointment_waiting_days, bins = [-1, 0, 7, 30, 90, 180, 500], labels=categories)

waiting_days_categories_dataset = show_attribute_statistics("waiting_days_categories", df, 0.005)

show_no_show_trend(waiting_days_categories_dataset, "waiting_days_categories", False)
# splitting data in groups

same_day_category = df[df.waiting_days_categories == categories[0]]

short_period_category = df.query(f"waiting_days_categories in ['{categories[1]}', '{categories[2]}']")

quarter_category = df[df.waiting_days_categories == categories[3]]

long_period_category = df[df.appointment_waiting_days > 90]



same_day_category.waiting_days_categories.unique(), short_period_category.waiting_days_categories.unique(), quarter_category.waiting_days_categories.unique(),  long_period_category.waiting_days_categories.unique()
print("Same day \n", same_day_category.mean(numeric_only=True))

print("\n")

print("Short period \n", short_period_category.mean(numeric_only=True))

print("\n")

print("Quarter \n", quarter_category.mean(numeric_only=True))

print("\n")

print("Long period \n", long_period_category.mean(numeric_only=True))
received_sms_dataset = show_attribute_statistics("received_sms", df, 0.005)

show_no_show_trend(received_sms_dataset, "received_sms")
appointment_week_day_dataset = show_attribute_statistics("appointment_week_day", df, 0.005, ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

show_no_show_trend(appointment_week_day_dataset, "appointment_week_day", False)
gender_dataset = show_attribute_statistics("gender", df, 0.001)

show_no_show_trend(gender_dataset, "gender", False)
neighbourhood_dataset = show_attribute_statistics("neighbourhood", df, 0.06, False, True);

neighbourhood_no_show_trend = show_no_show_trend(neighbourhood_dataset, "neighbourhood", False)

neighbourhood_no_show_trend.set_xticklabels(rotation='vertical')
df_groupped_by_neighborhood = df.groupby(['neighbourhood', 'no_show']).count().unstack().patient_id

df_groupped_by_neighborhood["sum"] = df_groupped_by_neighborhood['no'] + df_groupped_by_neighborhood['yes']

df_groupped_by_neighborhood.sort_values(by="sum", inplace=True)

df_groupped_by_neighborhood.dropna(inplace=True)



# plotting our data

plt.figure(figsize=(20, 30))



# bar chart

plt.barh(df_groupped_by_neighborhood.index, df_groupped_by_neighborhood['no'].values)

plt.barh(df_groupped_by_neighborhood.index, df_groupped_by_neighborhood['yes'].values)



# configs

plt.xlabel("amount of patients")

plt.ylabel("neighbourhood")

plt.legend(["attended", "not attended"])



plt.title("amount of patient by no show appointment groupped by neighbourhood")



plt.show();
# getting all neighbourhoods data from patients that no-showed groupped by waiting days categories

df_no_shows_by_neighbourhood_waiting_days_categories = df.query('no_show == "yes"').groupby(['neighbourhood', 'waiting_days_categories']).count().patient_id.fillna(value=0).unstack()

df_no_shows_by_neighbourhood_waiting_days_categories.head()
# normalizing values from the dataframe you can check out the method for this here: https://stackoverflow.com/a/31480994

df_no_shows_by_neighbourhood_waiting_days_categories = df_no_shows_by_neighbourhood_waiting_days_categories.div(df_no_shows_by_neighbourhood_waiting_days_categories.sum(axis=1), axis=0)

df_no_shows_by_neighbourhood_waiting_days_categories.head()
# converting the normalized values to percentage

df_no_shows_by_neighbourhood_waiting_days_categories = (df_no_shows_by_neighbourhood_waiting_days_categories * 100).round(2)

df_no_shows_by_neighbourhood_waiting_days_categories.head()
# get all necessary data for plotting

neighbourhoods = df_no_shows_by_neighbourhood_waiting_days_categories.index

waiting_days_categories = df_no_shows_by_neighbourhood_waiting_days_categories.columns.values



no_show_values_by_neighbourhood = np.array(df_no_shows_by_neighbourhood_waiting_days_categories.values)



neighbourhoods, waiting_days_categories
# plot the heatmap

figure, axes = plt.subplots(figsize=(60, 60))

axes.imshow(no_show_values_by_neighbourhood)



# show all the ticks

axes.set_xticks(np.arange(len(waiting_days_categories)))

axes.set_yticks(np.arange(len(neighbourhoods)))



# show all tick labels

axes.set_xticklabels(waiting_days_categories)

axes.set_yticklabels(neighbourhoods)



# Rotate the tick labels and set their alignment.

plt.setp(axes.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

for i in range(len(neighbourhoods)):

   for j in range(len(waiting_days_categories)):

       axes.text(j, i, no_show_values_by_neighbourhood[i, j], ha="center", va="center", color="w")



axes.set_title("no-show by neighbourhoods and waiting categories")

figure.tight_layout()

plt.show()
df_duplicated_patients = df[df.patient_id.duplicated() == True].groupby(['patient_id', 'no_show']).no_show.count().unstack()

df_duplicated_patients.fillna(0, inplace=True)

df_duplicated_patients["sum"] = df_duplicated_patients['no'] + df_duplicated_patients['yes']

df_duplicated_patients["no_show_rate"] = df_duplicated_patients['yes'] / (df_duplicated_patients['no'] + df_duplicated_patients['yes'])

df_duplicated_patients["no_show_rate_value"] = df_duplicated_patients["no_show_rate"] * 100

df_duplicated_patients.sort_values(by="sum", inplace=True)

df_duplicated_patients.dropna(inplace=True)



df_duplicated_patients.head()
df_duplicated_patients.describe()
duplicated_categories = pd.Series(['1', '2-5', '6-20', '21-40', '41-60', '>60'])

df_duplicated_patients['appointments_count_category'] = pd.cut(df_duplicated_patients['sum'], bins = [-1, 1, 5, 20, 40, 60, 500], labels=duplicated_categories)

df_duplicated_patients.head()
# see the distribution of the categories vs the no showing rate

show_no_show_trend(df_duplicated_patients, "appointments_count_category", False)
# check the rate of not attending by groups

df_duplicated_patients_group_by_category = df_duplicated_patients.groupby('appointments_count_category')

patients_attended = df_duplicated_patients_group_by_category.no.sum()

patients_not_attended = df_duplicated_patients_group_by_category.yes.sum()



patients_not_attended / (patients_attended + patients_not_attended)
from subprocess import call

call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])