# Setup

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

import plotly.express as px

import plotly.graph_objects as go

import numpy as np

import math 



# for time series

from fbprophet import Prophet

from fbprophet.plot import plot_plotly

import plotly.offline as py



py.init_notebook_mode()

pd.plotting.register_matplotlib_converters()

sns.set_style("whitegrid")

pd.set_option("display.max_columns", 30)



# load data

patient_path = "../input/coronavirusdataset/patient.csv"

time_path = "../input/coronavirusdataset/time.csv"

route_path = "../input/coronavirusdataset/route.csv"

patient = pd.read_csv(patient_path, index_col="patient_id")

time = pd.read_csv(time_path, index_col="date")

route = pd.read_csv(route_path, index_col="patient_id")



print(f"Last Update: {pd.datetime.today().strftime('%m/%d/%Y')}")
patient.head()
patient.info()
time.head()
# format date columns:

date_cols = ["confirmed_date", "released_date", "deceased_date"]

for col in date_cols:

    patient[col] = pd.to_datetime(patient[col])



time.index = pd.to_datetime(time.index)



# correct single spelling mistake:

patient.loc[patient["sex"]=="feamle", "sex"] = "female"

# correct single empty birth_year entry

patient.loc[patient["birth_year"]==" ", "birth_year"] = np.nan

patient["birth_year"] = patient["birth_year"].astype("float")



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



# for case fetality rate:

patient["state_deceased"] = (patient["state"] == "deceased").astype("int8")



# for underlying diseases:

patient.loc[patient["disease"] == 1, "disease"] = "Underlying disease"

patient.loc[patient["disease"] == 0, "disease"] = "No underlying disease"

patient.loc[patient["disease"].isna(), "disease"] = "Unknown"
# Acc tests

fig=go.Figure()

fig.add_trace((go.Scatter(x=time.index, y=time["test"],

                    mode='lines',

                    name="Acc. tests")))

fig.add_trace((go.Scatter(x=time.index, y=time["negative"],

                    mode='lines',

                    name="Acc. negative tests")))

fig.add_trace((go.Scatter(x=time.index, y=time["confirmed"],

                    mode='lines',

                    name="Acc. positive tests")))

fig.update_layout(title="Accumulated test results over time",

                   xaxis_title="Date",

                   yaxis_title="Count",

                   #yaxis_type="log",

                   template="seaborn")

fig.show()



# Acc cases

fig=go.Figure()

fig.add_trace((go.Scatter(x=time.index, y=time["confirmed"],

                    mode='lines',

                    name="Acc. confirmed")))

fig.add_trace((go.Scatter(x=time.index, y=time["released"],

                    mode='lines',

                    name="Acc. released")))

fig.add_trace((go.Scatter(x=time.index, y=time["deceased"],

                    mode='lines',

                    name="Acc. deceased")))

fig.update_layout(title="Accumulated cases over time",

                   xaxis_title="Date",

                   yaxis_title="Count",

                   #yaxis_type="log",

                   template="seaborn")

fig.show()

regions = time.drop(["time", "test","negative", "confirmed", "released", "deceased"], axis=1)

regions = regions.transpose(copy=True)



# joining the dfs makes datetimes unaccessable, so convert them to str first, dicard hour info:

regions.columns = [str(col)[:10] for col in regions.columns]



# region coordinates for provinces are those of the province capital.

region_coordinates = pd.DataFrame({

                "latitude":[37.532600, 35.166668, 35.834236, 37.456257, 35.166668, 36.351002,

                      35.549999, 26.291321, 37.156000, 37.87472, 36.63722, 36.32139,

                      35.82194, 34.99014, 36.56556, 35.22806, 33.499621],

                "longitude": [127.024612, 129.066666, 128.534210, 126.705208,

                        126.916664, 127.385002, 129.316666, 127.165604, 127.006000,

                        127.73417, 127.48972, 127.41972, 127.14889, 126.47899, 128.725,

                        128.68111, 126.531188]}, 

                index=['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',

                        'Ulsan', 'Sejong', 'Gyeonggi-do', 'Gangwon-do',

                        'Chungcheongbuk-do', 'Chungcheongnam-do', 'Jeollabuk-do',

                        'Jeollanam-do', 'Gyeongsangbuk-do', 'Gyeongsangnam-do', 'Jeju-do'])



regions = region_coordinates.join(regions, how="left")



marker_scale_factor = 0.4 # scale marker size based on number confirmed to the power of this factor

route_map = folium.Map(location=[36.5,128],

                       min_zoom=3,

                       max_zoom=10,

                       zoom_start=6.5,

                       tiles="cartodbpositron")



for lat, lon, date in zip(regions["latitude"],

                          regions["longitude"],

                          regions.iloc[:, -1]): # iloc for last date

    folium.CircleMarker([lat, lon],

                  color="red",

                  radius=math.pow(date, marker_scale_factor),

                  fill=True,

                  fill_color="crimson",

                  fill_opacity=0.2).add_to(route_map)

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

              order=["female", "male"],

            hue_order=["isolated", "released", "deceased"],

            data=patient)

plt.title("Patient state by gender", fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
age_ranges = sorted(set([ar for ar in patient["age_range"] if ar != "Unknown"]))

custom_palette = ["blue", "green", "red"]

# figure

plt.figure(figsize=(12, 8))

sns.countplot(x = "age_range",

            hue="state",

            order=age_ranges,

            hue_order=["isolated", "released", "deceased"],

            palette=custom_palette,

            data=patient)

plt.title("State by age", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("Count", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper right")

plt.show()



# stats

deceased_age_dist = patient.loc[patient["state_deceased"] == 1]["age"].describe()

median_age_of_deceased = int(patient.loc[patient["state_deceased"] == 1]["age"].median())

mean_age_of_deceased = int(deceased_age_dist["mean"])

mean_age_of_deceased_std = int(deceased_age_dist["std"])

min_age_of_deceased = int(deceased_age_dist["min"])



print(f"The mean age of those who died is {mean_age_of_deceased} +/- {mean_age_of_deceased_std} years (median: {median_age_of_deceased}).")

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



# Duration boxplot

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

plt.title("Time from confirmation to release or death \n (expluding post mortem confirmations)",

          fontsize=16)

plt.xlabel("State", fontsize=16)

plt.ylabel("Days", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()



# boxplot by gender

order_duration_sex = ["female", "male"]

plt.figure(figsize=(12, 8))

sns.boxplot(x="sex",

            y="duration_days",

            order=order_duration_sex,

            hue="state",            

            hue_order=["released", "deceased"],

            data=excl_post_mortem)

plt.title("Time from confirmation to release or death by gender\n (expluding post mortem confirmations)",

          fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("Days", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()



# boxplot by age_range

order_duration_age = sorted(patient["age_range"].unique())[:-1]

plt.figure(figsize=(12, 8))

sns.boxplot(x="age_range",

            y="duration_days",

            order=order_duration_age,

            hue="state",            

            hue_order=["released", "deceased"],

            data=excl_post_mortem)

plt.title("Time from confirmation to release or death by age range\n (expluding post mortem confirmations)",

          fontsize=16)

plt.xlabel("Age range", fontsize=16)

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

print(f"{patients_over_upper_quartile} ({upper_perc}%) of those have been isolated for more than {upper_quartile_duration_death} days (75th quartile of deceased).")
disease_info_yes = patient.loc[patient["disease"] == "Underlying disease"].shape[0]

disease_yes_deceased = patient.loc[(patient["disease"] == "Underlying disease") & (patient["state"] == "deceased")].shape[0]

disease_info_no = patient.loc[patient["disease"] == "No underlying disease"].shape[0]

disease_info_all = disease_info_yes + disease_info_no

deceased_all = patient["state_deceased"].sum()

perc_disease = round((disease_info_yes / disease_info_all * 100),1) # percentage of disease known

perc_disease_died = round((disease_info_yes / disease_yes_deceased * 100),1) # pre-existing diseased who died

perc_deceased_disease = round((disease_yes_deceased / deceased_all * 100),1) # deceased with pre-existing disease





print(f"Disease information is available for {disease_info_all} patients.")

print(f"Out of these {disease_info_all} patients, {disease_info_yes} have/had an underlying disease ({perc_disease}%).")

print(f"Out of these {disease_info_yes} patients with an underlying disease, {disease_yes_deceased} have died ({perc_disease_died}%).")

print(f"Out of the {deceased_all} patients that died, {disease_yes_deceased} had an underlying disease ({perc_deceased_disease}%).")



# age of deceased with no known underlying disease:

age_mean = int(patient.loc[(patient["disease"]=="Unknown") & (patient["state"]=="deceased")]["age"].mean())

age_median = int(patient.loc[(patient["disease"]=="Unknown") & (patient["state"]=="deceased")]["age"].median())

age_min = int(patient.loc[(patient["disease"]=="Unknown") & (patient["state"]=="deceased")]["age"].min())

age_max = int(patient.loc[(patient["disease"]=="Unknown") & (patient["state"]=="deceased")]["age"].max())



print(f"The deceased with no underlying disease were between {age_min} and {age_max} years old (mean: {age_mean}, median: {age_median}).")



hue_order_disease = ["Underlying disease", "No underlying disease", "Unknown"]

plt.figure(figsize=(10, 8))

sns.countplot(data=patient,

            x="state_deceased",

            hue="disease",

            order=[1],

            hue_order=hue_order_disease)

plt.title("Underlying diseases of deceased", fontsize=16)

plt.xlabel("", fontsize=16)

plt.ylabel("Number of cases", fontsize=16)

plt.xticks(fontsize=0)

plt.yticks(fontsize=12)

plt.show()
total_confirmed = time.sort_values(by="date", ascending=False).iloc[0]["confirmed"]

total_deceased = time.sort_values(by="date", ascending=False).iloc[0]["deceased"]

total_recovered = time.sort_values(by="date", ascending=False).iloc[0]["released"]

total_cfr = round((total_deceased / total_confirmed * 100),1)



print(f"The current CFR for COVID-19 in South Korea is {total_cfr} %.")

print(f"This number is based on {total_confirmed} confirmed cases and {total_deceased} fetalities.")

print("Calculating the CFR using the number of released patients instead of confirmed cases yields a much higher - at this point in time unrealistic - number")
#calc numbers and rates

total_confirmed = patient.shape[0]

females = patient.loc[patient["sex"] == "female"].shape[0]

males = patient.loc[patient["sex"] == "male"].shape[0]



females_deceased = patient.loc[(patient["sex"] == "female") & (patient["state"] == "deceased")].shape[0]

males_deceased = patient.loc[(patient["sex"] == "male") & (patient["state"] == "deceased")].shape[0]

total_deceased = patient.loc[patient["state"] == "deceased"].shape[0]



female_cfr = round((females_deceased / females * 100),1)

male_cfr = round((males_deceased / females * 100),1)

total_cfr = round((total_deceased / total_confirmed * 100),1)

total_cfr_sex_known = round(((females_deceased + males_deceased) / (females + males) * 100),1)



#extrapolation

fraction_sex_known = patient.loc[patient["sex"].notna()].shape[0] / total_confirmed



estimated_female_patients = int(round(females / fraction_sex_known, 0))

estimated_male_patients = int(round(males / fraction_sex_known, 0))

extr_female_rate = round((female_cfr * fraction_sex_known),1)

extr_male_rate = round((male_cfr * fraction_sex_known),1)



# make dataframe

cfr_gender = pd.DataFrame({"Known number of patients": [females,

                                                             males,

                                                             total_confirmed],

                                      "Number deceased": [females_deceased,

                                                          males_deceased,

                                                          total_deceased],

                                      "CFR [%] (gender known)": [female_cfr,

                                                                              male_cfr,

                                                                              total_cfr_sex_known],

                                      "Estimated number of total patients": [estimated_female_patients,

                                                                       estimated_male_patients,

                                                                       estimated_female_patients + estimated_male_patients],

                                      "Estimated total CFR [%]": [extr_female_rate,

                                                                           extr_male_rate,

                                                                           total_cfr]}, 

                                     index=["Female", "Male", "Total"])

cfr_gender.index.name = "Gender"

cfr_gender
gender_order = ["Female", "Male"]

plt.figure(figsize=(10, 8))

sns.barplot(x=cfr_gender.index,

            y=cfr_gender["Estimated total CFR [%]"],

            order = gender_order,

            palette=["grey"])

plt.title("Estimated CFR by gender", fontsize=16)

plt.xlabel("Gender", fontsize=16)

plt.ylabel("CFR [%]", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
#rate by age:

cfr_age = pd.DataFrame(patient.groupby("age_range")["state_deceased"].describe()[["count", "mean"]])

cfr_age.rename(columns={"count":"Number of patients","mean":"CFR"}, inplace=True)

cfr_age.drop("Unknown", axis=0, inplace=True)



# add total rate for age known:

total_patients_w_age = patient.loc[patient["age"].notna()].shape[0]

total_deceased_w_age = patient.loc[(patient["age"].notna()) & (patient["state"] == "deceased")].shape[0]

total_rate_w_age = total_deceased_w_age / total_patients_w_age

total_w_age_cfr = pd.DataFrame({"Number of patients": total_patients_w_age,

                                     "CFR": total_rate_w_age},

                                    index=["Total (age known)"])





#extrapolation

fraction_age_known = patient.loc[patient["age"].notna()].shape[0] / total_confirmed



# df:

cfr_age["Number of patients"] = cfr_age["Number of patients"].astype("int64")

cfr_age["CFR"] = round(cfr_age["CFR"],3) * 100

cfr_age.rename(columns={"CFR":"CFR [%] (age known)"}, inplace=True)

cfr_age.index.name = "Age range"

cfr_age["Estimated number of total patients"] = (cfr_age["Number of patients"] / fraction_age_known).astype("int64")

cfr_age["Estimated total CFR [%]"] = round((cfr_age["CFR [%] (age known)"] * fraction_age_known), 1)



cfr_age



plt.figure(figsize=(12, 8))

sns.barplot(x=cfr_age.index,

            y=cfr_age["Estimated total CFR [%]"],

            order = age_ranges,

            palette=["grey"])

plt.title("Estimated CFR by age", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("CFR [%]", fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
#rate by gender and age:

cfr_gender_age = pd.DataFrame(patient.groupby(["age_range", "sex"])["state_deceased"].describe()[["count", "mean"]])

cfr_gender_age.rename(columns={"count":"Number of patients","mean":"CFR"}, inplace=True)

cfr_gender_age.drop("Unknown", axis=0, inplace=True)



#formatting:

cfr_gender_age["Number of patients"] = cfr_gender_age["Number of patients"].astype("int64")

cfr_gender_age["CFR"] = round(cfr_gender_age["CFR"],3) * 100

cfr_gender_age.rename(columns={"CFR":"CFR [%] (gender and age known)"}, inplace=True)



# only for plotting:

cfr_gender_age["age_range"] = list(x[0] for x in cfr_gender_age.index)

cfr_gender_age["gender"] = list(x[1] for x in cfr_gender_age.index)



#extrapolation

fraction_age_known = patient.loc[patient["age"].notna()].shape[0] / total_confirmed

fraction_sex_known = patient.loc[patient["sex"].notna()].shape[0] / total_confirmed



cfr_gender_age["Estimated total number of patients"] = (cfr_gender_age["Number of patients"] / fraction_age_known).astype("int64")

cfr_gender_age["Estimated total CFR [%]"] = round((cfr_gender_age["CFR [%] (gender and age known)"] * fraction_age_known), 1)

# show table

cfr_gender_age.drop(["age_range", "gender"], axis=1)
plt.figure(figsize=(12, 8))

sns.barplot(x="age_range",

            y=cfr_gender_age["Estimated total CFR [%]"],

            order = age_ranges,

            hue="gender",

            hue_order=["female", "male"],

            palette=["darkgrey", "black"],

            data=cfr_gender_age)

plt.title("Estimated CFR by age and gender", fontsize=16)

plt.xlabel("Age range", fontsize=16)

plt.ylabel("CFR [%]", fontsize=16)

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

infected_time = time.sort_values(by="date", ascending=False).iloc[0]["confirmed"]

recovered_time = time.sort_values(by="date", ascending=False).iloc[0]["released"]

deceased_time = time.sort_values(by="date", ascending=False).iloc[0]["deceased"]

dead_per_recovered_t = deceased_time / recovered_time



outcome = pd.DataFrame({"Confirmed": [infected_patient, infected_time],

                                "Recovered":[recovered_patient, recovered_time],

                                "Deceased": [deceased_patient, deceased_time]}, index=["patient.csv", "time.csv"])

outcome.index.name="Data source"

outcome["Recovered [%]"] = round((outcome["Recovered"] / outcome["Confirmed"] * 100), 1)

outcome["Deceased [%]"] = round((outcome["Deceased"] / outcome["Confirmed"] * 100), 1)

outcome["Deceased / Recovered [%]"] = round((outcome["Deceased"] / outcome["Recovered"] * 100), 1)

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
# utility function:



def prophet_prediction(fit_df, n_periods, **kwargs):

    """Fit FB Prophet to provided DataFrame, predict n_periods in the 

    future. Return tuple of fitted predictor and result as a

    pandas DataFrame containing ds, the target and upper and lower bounds.

    Provided df must be two columns 'ds' and 'y'. n_periods must be an int.

    Any provided keywords are passed on to Prophet.

    """

    # create instance of Prophet

    proph = Prophet(**kwargs)

    # fit model

    proph.fit(fit_df)

    # define future dataframe / length of prediction

    future_df = proph.make_future_dataframe(periods=n_periods)

    # make predictions and store as dataframe

    pred = proph.predict(future_df)

    

    #format df columns:

    pred["yhat"] = pred["yhat"].astype("int64")

    pred["yhat_lower"] = pred["yhat_lower"].astype("int64")

    pred["yhat_upper"] = pred["yhat_upper"].astype("int64")

    

    # return fitted model and predicted data as tuple

    return (proph, pred[["ds", "yhat", "yhat_lower", "yhat_upper"]])
#---------This is a test of the model based on historical data------------



# From the EDA we know that the confirmed cases started to increase sharply around 2020-02-20. 

# We will use this as the starting point to fit our model.

start_date = "2020-02-20"

end_date = "2020-03-05"



# prep input for prophet:

confirmed_data = pd.DataFrame({"ds": time.index, "y": time["confirmed"]})

confirmed_data = confirmed_data[start_date:end_date]



# use utility function for modeling:

pred_confirmed = prophet_prediction(confirmed_data, 5)[1]



# Add actual data to dataframe for comparison and calculate absolute percentage errors.

pred_confirmed["actual confirmed"] =  time[start_date:"2020-03-10"]["confirmed"].values

pred_confirmed["Difference [%]"] = round((abs(pred_confirmed["yhat"] - pred_confirmed["actual confirmed"]) 

                                          / pred_confirmed["actual confirmed"] * 100),2)



# Mean absolute percentage error (MAPE):

mape = round(np.mean(pred_confirmed["Difference [%]"].tail(5)),2)

print(f"MAPE of confirmed case predictions for the time period 2020-03-06 - 2020-03-10: {mape}%.")

pred_confirmed[["ds", "yhat", "yhat_lower", "yhat_upper", "actual confirmed", "Difference [%]"]].tail(5)
start_date = "2020-02-20"

end_date = "2020-03-10" # Date of modeling

days_to_predict = 10



actual_confirmed = list(time["2020-03-11":]["confirmed"].values)

while len(actual_confirmed) < 10:

    actual_confirmed.append(np.nan)



confirmed_data = pd.DataFrame({"ds": time.index, "y": time["confirmed"]})

confirmed_data = confirmed_data[start_date:end_date]

confirmed_w_pred = prophet_prediction(confirmed_data, days_to_predict)

result = confirmed_w_pred[1].tail(10).copy()

result.index = result["ds"]

result.index.name = "Date"

result["actual confirmed"] = actual_confirmed

result["Difference [%]"] = round((abs(result["yhat"] - result["actual confirmed"]) 

                                          / result["actual confirmed"] * 100),2)

result.drop("ds", axis=1)
# Plot modeled data:

fig = plot_plotly(confirmed_w_pred[0], confirmed_w_pred[1])

fig.update_layout(title={"text": "Predicted development of confirmed cases", "x": 0.5, "xanchor": "center"},

                  xaxis_title="Date",

                  yaxis_title="Number of cases")

py.iplot(fig)