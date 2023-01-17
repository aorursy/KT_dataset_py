# Setup

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import plotly.express as px

import plotly.graph_objects as go

import numpy as np

import math 



pd.plotting.register_matplotlib_converters()

sns.set_style("whitegrid")



# load data

patient_path = "../input/coronavirusdataset/patient.csv"

time_path = "../input/coronavirusdataset/time.csv"

route_path = "../input/coronavirusdataset/route.csv"

patient = pd.read_csv(patient_path, index_col="id")

time = pd.read_csv(time_path, index_col="176") # temporary fix orig. col name "date"

time.index.name="date"

route = pd.read_csv(route_path, index_col="id")
print(f"Last Update: {pd.datetime.today().strftime('%m/%d/%Y')}")
patient.head()
patient.info()
time.head()
# format date columns:

date_cols = ["confirmed_date", "released_date", "deceased_date"]

for col in date_cols:

    patient[col] = pd.to_datetime(patient[col])



time.index = pd.to_datetime(time.index)



# Derive features:



#status by gender:

patient["state_by_gender"] = patient["state"] + "_" + patient["sex"]



# age:

# approximation, using 2019 - assuming 3/4 are born after march/current month of 2020

patient["age"] = 2019 - patient["birth_year"]





def group_age(age):

    """This function is used to group patients by age

    in steps of 10 years. It returns the age range

    of the patient as a string.

    """

    if age >= 0: # not NaN

        if age % 10 != 0:

            lower = int(math.floor(age / 10.0)) * 10

            upper = int(math.ceil(age / 10.0)) * 10 - 1

            return f"{lower}-{upper}"

        else:

            lower = int(age)

            upper = int(age + 9) 

            return f"{lower}-{upper}"

    return "Unknown"





patient["age_range"] = patient["age"].apply(group_age)



# duration of infection:

patient["time_to_release_since_confirmed"] = patient["released_date"] - patient["confirmed_date"]

patient["time_to_death_since_confirmed"] = patient["deceased_date"] - patient["confirmed_date"]

patient["duration_since_confirmed"] = patient[["time_to_release_since_confirmed", "time_to_death_since_confirmed"]].min(axis=1)

patient["duration_days"] = patient["duration_since_confirmed"].dt.days



# for mortality rate:

patient["state_deceased"] = (patient["state"] == "deceased").astype("int8")
fig=go.Figure()

fig.add_trace((go.Scatter(x=time.index, y=time["acc_test"],

                    mode='lines',

                    name="Accumulated tests")))

fig.add_trace((go.Scatter(x=time.index, y=time["acc_negative"],

                    mode='lines',

                    name="Accumulated negative tests")))

fig.add_trace((go.Scatter(x=time.index, y=time["acc_confirmed"],

                    mode='lines',

                    name="Accumulated positive tests")))

fig.update_layout(title="Accumulated test results",

                   xaxis_title="Date",

                   yaxis_title="Count")

fig.show()



fig=go.Figure()

fig.add_trace((go.Scatter(x=time.index, y=time["new_test"],

                    mode='lines',

                    name="Daily tests")))

fig.add_trace((go.Scatter(x=time.index, y=time["new_negative"],

                    mode='lines',

                    name="Daily negative tests")))

fig.add_trace((go.Scatter(x=time.index, y=time["new_confirmed"],

                    mode='lines',

                    name="Daily positive tests")))

fig.update_layout(title="Daily test results",

                   xaxis_title="Date",

                   yaxis_title="Count")

fig.show()
fig=go.Figure()

fig.add_trace((go.Scatter(x=time.index, y=time["acc_confirmed"],

                    mode='lines',

                    name="Accumulated confirmed")))

fig.add_trace((go.Scatter(x=time.index, y=time["acc_released"],

                    mode='lines',

                    name="Accumulated released")))

fig.add_trace((go.Scatter(x=time.index, y=time["acc_deceased"],

                    mode='lines',

                    name="Accumulated deceased")))

fig.update_layout(title="Accumulated cases",

                   xaxis_title="Date",

                   yaxis_title="Count")

fig.show()



fig=go.Figure()

fig.add_trace((go.Scatter(x=time.index, y=time["new_confirmed"],

                    mode='lines',

                    name="New confirmed")))

fig.add_trace((go.Scatter(x=time.index, y=time["new_released"],

                    mode='lines',

                    name="New released")))

fig.add_trace((go.Scatter(x=time.index, y=time["new_deceased"],

                    mode='lines',

                    name="New deceased")))

fig.update_layout(title="New daily cases",

                   xaxis_title="Date",

                   yaxis_title="Count")

fig.show()
route_map = folium.Map(location=[36.5,128],

                       min_zoom=3,

                       max_zoom=10,

                       zoom_start=7,

                       tiles="cartodbpositron")

for lat, lon in zip(route["latitude"], route["longitude"]):

    folium.Circle([lat, lon],

                  color="crimson",

                  radius=3).add_to(route_map)

route_map
# Reason of infection

reason_order = list(patient["infection_reason"].value_counts().index)



plt.figure(figsize=(12, 8))

sns.countplot(y = "infection_reason",

              data=patient,

              order=reason_order)

plt.title("Known reasons of infection", fontsize=16)

plt.xlabel("Count", fontsize=16)

plt.ylabel("Reason of infection", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
states = pd.DataFrame(patient["state"].value_counts())

states["status"] = states.index

states.rename(columns={"state": "count"}, inplace=True)



fig = px.pie(states,

             values="count",

             names="status",

             title="Current state of patients",

             template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")

fig.show()
plt.figure(figsize=(10, 8))

sns.countplot(x = "sex",

            hue="state",

            hue_order=["isolated", "released", "deceased"],

            data=patient)

plt.title("Patient state by gender", fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
age_ranges = sorted(set([ar for ar in patient["age_range"] if ar != "Unknown"]))



plt.figure(figsize=(12, 8))

sns.countplot(x = "age_range",

            hue="state",

            order=age_ranges,

            hue_order=["isolated", "released", "deceased"],

            data=patient)

plt.title("State by age", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper right")

plt.show()
deceased_age_dist = patient.loc[patient["state_deceased"] == 1]["age"].describe()

mean_age_of_deceased = int(deceased_age_dist["mean"])

mean_age_of_deceased_std = int(deceased_age_dist["std"])

min_age_of_deceased = int(deceased_age_dist["min"])



print(f"The mean age of those who died is {mean_age_of_deceased} +/- {mean_age_of_deceased_std} years.")

print(f"The youngest patient who died was {min_age_of_deceased} years old.")
# define order

age_gender_hue_order =["isolated_female", "released_female", "deceased_female",

                       "isolated_male", "released_male", "deceased_male"]

# color list:

custom_palette = ["royalblue", "lightgreen", "orangered", "blue", "green", "red"]



plt.figure(figsize=(12, 8))

sns.countplot(x ="age_range",

              hue="state_by_gender",

              order=age_ranges,

              hue_order=age_gender_hue_order,

              palette=custom_palette,

              data=patient)

plt.title("State by gender and age", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper right")

plt.show()
# exclude post mortem confirmations:

excl_post_mortem = patient.loc[(patient["time_to_death_since_confirmed"].astype("int64") > 0) |

                               (patient["time_to_release_since_confirmed"].astype("int64") > 0)]



durations = excl_post_mortem[["time_to_release_since_confirmed", "time_to_death_since_confirmed"]].describe()

# durations
plt.figure(figsize=(12, 8))

sns.boxplot(x="state",

            y="duration_days",

            order=["released", "deceased"],

            data=excl_post_mortem)

sns.swarmplot(x="state",

            y="duration_days",

            order=["released", "deceased"],

            size= 8.0,  

            color=".25",  

            data=excl_post_mortem)

plt.title("Time from confirmation to release or death \n expluding post mortem confirmations",

          fontsize=16)

plt.xlabel("State", fontsize=16)

plt.ylabel("Days", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()

isolated = patient.loc[patient["state"] == "isolated"].shape[0]

upper_quartile_duration_death = durations["time_to_death_since_confirmed"]["75%"].days

max_duration_death = durations["time_to_death_since_confirmed"]["max"].days

median_duration_death = durations



patient["time_since_confirmation"] = pd.to_datetime("today") - patient["confirmed_date"]

patients_over_upper_quartile = patient.loc[(patient["time_since_confirmation"].dt.days > upper_quartile_duration_death) & 

                                           (patient["state"] == "isolated")].shape[0]

patients_over_max = patient.loc[(patient["time_since_confirmation"].dt.days > max_duration_death) & 

                                           (patient["state"] == "isolated")].shape[0]

upper_perc = round((patients_over_upper_quartile / isolated * 100), 2)

max_perc = round((patients_over_max / isolated * 100), 2)



print(f"Currently, {isolated} patients are isolated in South Korea.")

print(f"{patients_over_upper_quartile} ({upper_perc}%) of those have been isolated for more than {upper_quartile_duration_death} days (75% quartile of deceased).")

print(f"{patients_over_max} ({max_perc}%) of those have been isolated for more than {max_duration_death} days (max of deceased).")
#rate by gender:

mortality_rates_gender = pd.DataFrame(patient.groupby("sex")["state_deceased"].describe()[["count", "mean"]])

mortality_rates_gender.rename(columns={"count":"Number of patients","mean":"Mortality rate"}, inplace=True)

mortality_rates_gender



# add total rate for sex known:

total_patients_w_sex = patient.loc[patient["sex"].notna()].shape[0]

total_deceased_w_sex = patient.loc[(patient["sex"].notna()) & (patient["state"] == "deceased")].shape[0]

total_rate_w_sex = total_deceased_w_sex / total_patients_w_sex

total_w_sex_mortality_rate = pd.DataFrame({"Number of patients": total_patients_w_sex,

                                     "Mortality rate": total_rate_w_sex},

                                    index=["Total (gender known)"])





# add total incl. no age known:

total_patients = patient.shape[0]

total_deceased = patient.loc[patient["state"] == "deceased"].shape[0]

total_rate = total_deceased / total_patients

total_mortality_rate = pd.DataFrame({"Number of patients": total_patients,

                                     "Mortality rate": total_rate},

                                    index=["Total (all patients)"])



# df:

mortality_rates_gender = mortality_rates_gender.append(total_w_sex_mortality_rate)

mortality_rates_gender = mortality_rates_gender.append(total_mortality_rate)

mortality_rates_gender["Number of patients"] = mortality_rates_gender["Number of patients"].astype("int64")

mortality_rates_gender["Mortality rate"] = round(mortality_rates_gender["Mortality rate"],3) * 100

mortality_rates_gender.rename(columns={"Mortality rate": "Mortality rate [%]"}, inplace=True)

mortality_rates_gender.index.name = "Gender"

mortality_rates_gender
gender_order = ["female", "male"]

plt.figure(figsize=(10, 8))

sns.barplot(x=mortality_rates_gender.index,

            y=mortality_rates_gender["Mortality rate [%]"],

            order = gender_order,

            palette=["grey"])

plt.title("Mortality rate by gender", fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Mortality rate [%]", fontsize=16)

plt.axhline(y=total_rate_w_sex * 100,

            color="darkgrey",

            linestyle="--",

            label="mean of patients in South Korea with known gender")

plt.axhline(y= total_rate * 100,

            color="black",

            linestyle="--",

            label="mean of all patients in South Korea")

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper left")

plt.show()
#rate by age:

mortality_rates_age = pd.DataFrame(patient.groupby("age_range")["state_deceased"].describe()[["count", "mean"]])

mortality_rates_age.rename(columns={"count":"Number of patients","mean":"Mortality rate"}, inplace=True)

mortality_rates_age.drop("Unknown", axis=0, inplace=True)



# add total rate for age known:

total_patients_w_age = patient.loc[patient["age"].notna()].shape[0]

total_deceased_w_age = patient.loc[(patient["age"].notna()) & (patient["state"] == "deceased")].shape[0]

total_rate_w_age = total_deceased_w_age / total_patients_w_age

total_w_age_mortality_rate = pd.DataFrame({"Number of patients": total_patients_w_age,

                                     "Mortality rate": total_rate_w_age},

                                    index=["Total (age known)"])



# df:

mortality_rates_age = mortality_rates_age.append(total_w_age_mortality_rate)

mortality_rates_age = mortality_rates_age.append(total_mortality_rate)

mortality_rates_age["Number of patients"] = mortality_rates_age["Number of patients"].astype("int64")

mortality_rates_age["Mortality rate"] = round(mortality_rates_age["Mortality rate"],3) * 100

mortality_rates_age.rename(columns={"Mortality rate":"Mortality rate [%]"}, inplace=True)

mortality_rates_age.index.name = "Age range"

mortality_rates_age
plt.figure(figsize=(12, 8))

sns.barplot(x=mortality_rates_age.index,

            y=mortality_rates_age["Mortality rate [%]"],

            order = age_ranges,

            palette=["grey"])

plt.title("Mortality rate by age", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("Mortality rate [%]", fontsize=16)

plt.axhline(y=total_rate_w_age * 100,

            color="darkgrey",

            linestyle="--",

            label="mean of patients in South Korea with known age")

plt.axhline(y= total_rate * 100,

            color="black",

            linestyle="--",

            label="mean of all patients in South Korea")

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper left")

plt.show()
#rate by gender and age:

mortality_rates_gender_age = pd.DataFrame(patient.groupby(["age_range", "sex"])["state_deceased"].describe()[["count", "mean"]])

mortality_rates_gender_age.rename(columns={"count":"Number of patients","mean":"Mortality rate"}, inplace=True)

mortality_rates_gender_age.drop("Unknown", axis=0, inplace=True)



#formatting:

mortality_rates_gender_age["Number of patients"] = mortality_rates_gender_age["Number of patients"].astype("int64")

mortality_rates_gender_age["Mortality rate"] = round(mortality_rates_gender_age["Mortality rate"],3) * 100

mortality_rates_gender_age.rename(columns={"Mortality rate":"Mortality rate [%]"}, inplace=True)



# only for plotting:

mortality_rates_gender_age["age_range"] = list(x[0] for x in mortality_rates_gender_age.index)

mortality_rates_gender_age["gender"] = list(x[1] for x in mortality_rates_gender_age.index)



# show table

mortality_rates_gender_age[["Number of patients", "Mortality rate [%]"]]
mean_mortality_rate_female = mortality_rates_gender["Mortality rate [%]"]["female"]

mean_mortality_rate_male = mortality_rates_gender["Mortality rate [%]"]["male"]



plt.figure(figsize=(12, 8))

sns.barplot(x="age_range",

            y=mortality_rates_gender_age["Mortality rate [%]"],

            order = age_ranges,

            hue="gender",

            hue_order=["female", "male"],

            palette=["darkgrey", "black"],

            data=mortality_rates_gender_age)

plt.title("Mortality rate by age and gender", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("Mortality rate [%]", fontsize=16)

plt.axhline(y=total_rate_w_age * 100,

            color="darkgrey",

            linestyle="--",

            label="mean of patients in South Korea with known age")

plt.axhline(y=mean_mortality_rate_female,

            color="grey",

            linestyle=":",

            label="mean of female patients in South Korea with known gender")

plt.axhline(y=mean_mortality_rate_male,

            color="grey",

            linestyle="-.",

            label="mean of male patients in South Korea  with known gender")

plt.axhline(y= total_rate * 100,

            color="black",

            linestyle="--",

            label="mean of all patients in South Korea")

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper left")

plt.show()
# using data from patient.csv:

infected_patient = patient.shape[0]

recovered_patient = patient.loc[patient["state"] == "released"].shape[0]

deceased_patient = patient.loc[patient["state"] == "deceased"].shape[0]

dead_per_recovered_p = deceased_patient / recovered_patient

dead_per_recovered_p



# using data from time.csv:

infected_time = time.sort_values(by="date", ascending=False).iloc[0]["acc_confirmed"]

recovered_time = time.sort_values(by="date", ascending=False).iloc[0]["acc_released"]

deceased_time = time.sort_values(by="date", ascending=False).iloc[0]["acc_deceased"]

dead_per_recovered_t = deceased_time / recovered_time



outcome = pd.DataFrame({"Confirmed": [infected_patient, infected_time],

                                "Recovered":[recovered_patient, recovered_time],

                                "Deceased": [deceased_patient, deceased_time]}, index=["patient.csv", "time.csv"])

outcome.index.name="Data source"

outcome["Recovered [%]"] = round(outcome["Recovered"] / outcome["Confirmed"], 3)

outcome["Deceased [%]"] = round(outcome["Deceased"] / outcome["Confirmed"], 3)

outcome["Deceased / Recovered"] = round(outcome["Deceased"] / outcome["Recovered"], 3)

outcome["Data source"] = outcome.index # only for plotting

outcome.drop("Data source", axis=1)
outcome_fig = outcome.melt("Data source", var_name="columns",  value_name="values")



sns.catplot(x="columns",

            y="values",

            hue="Data source", 	

            kind="bar",   

            data=outcome_fig.iloc[2:6])

plt.title("Outcomes of disease so far", fontsize=16)

plt.xlabel("Outcome", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(rotation=45, fontsize=12)

plt.yticks(fontsize=12)

plt.show()