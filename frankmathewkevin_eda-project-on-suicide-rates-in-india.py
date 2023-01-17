import numpy as np
import pandas as pd
pd.options.display.max_columns = 30
pd.options.display.max_rows = 30
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows",None)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("/kaggle/input/suicides-in-india/Suicides in India 2001-2012.csv")
data.head(3)
data.info()
df = data.copy()
df.rename({"Type":"TypeofCauses","Type_code":"Category"}, axis = "columns", inplace = True)
df.head(5)
df.State.value_counts()
#Replace Delhi(ut) with Delhi
df.replace("Delhi (Ut)", "Delhi", inplace = True)
#Removing Total(States), Total(Uts), Total (All India)
df = df.drop(df[(df.State == "Total (States)") |(df.State == "Total (Uts)") | (df.State == "Total (All India)") ].index)
df.State.value_counts()
f, ax = plt.subplots(1,2, figsize = (15,7))
df.groupby("State")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", color = "red", ax= ax[0])                                                                           
plt.ylabel("No.of Suicides")
df.groupby("State")["Total"].sum().sort_values(ascending = True)[:10].plot(kind = "bar",  color = "blue", ax= ax[1])
plt.ylabel("No.of Suicides")                                                                          
ax[0].title.set_text('Top 10 States with highest suicides')
ax[1].title.set_text('Top 10 States with lowest suicides')
df.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:10].plot(kind = "bar", color = "y", figsize = (15,7))
plt.ylabel("No. of suicides")
plt.title("Most common reasons for committing suicide")
f, ax = plt.subplots(2,2, figsize = (15,12), constrained_layout = True)
mh = df[df["State"] == "Maharashtra"]
mh_cause = mh[mh["Category"] == "Causes"]
mh_edu = mh[mh["Category"] == "Education_Status"]
mh_adop = mh[mh["Category"] == "Means_adopted"]
mh_prof = mh[mh["Category"] == "Professional_Profile"]
mh_socio = mh[mh["Category"] == "Social_Status"]

mh_edu.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,0], color = "b", title = "Reason for suicide based on Education")
mh_adop.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,1], color = "r",title ="Reason for suicide based on Means adopted")
mh_prof.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,0], color = "g",title = "Reason for suicide based on Professional Profile")
mh_socio.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,1], color = "c",title = "Reason for suicide based on Social Status")
plt.setp(ax[0,0], xlabel="")
plt.setp(ax[0,1], xlabel="")
plt.setp(ax[1,0], xlabel="")
plt.setp(ax[1,1], xlabel="")
f.suptitle("Most common Reasons for committing suicide in Maharashtra" + "\n")
f, ax = plt.subplots(2,2, figsize = (15,12), constrained_layout = True)
wb = df[df["State"] == "West Bengal"]
wb_cause = wb[wb["Category"] == "Causes"]
wb_edu = wb[wb["Category"] == "Education_Status"]
wb_adop = wb[wb["Category"] == "Means_adopted"]
wb_prof = wb[wb["Category"] == "Professional_Profile"]
wb_socio = wb[wb["Category"] == "Social_Status"]

wb_edu.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,0],color = "b", title = "Reason for suicide based on Education")
wb_adop.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,1],color = "r", title ="Reason for suicide based on Means adopted")
wb_prof.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,0],color = "g", title = "Reason for suicide based on Professional Profile")
wb_socio.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,1],color = "c", title = "Reason for suicide based on Social Status")
plt.setp(ax[0,0], xlabel="")
plt.setp(ax[0,1], xlabel="")
plt.setp(ax[1,0], xlabel="")
plt.setp(ax[1,1], xlabel="")
f.suptitle("Most common reasons for committing suicide in West Bengal" + "\n")
f, ax = plt.subplots(2,2, figsize = (15,12), constrained_layout = True)
tn = df[df["State"] == "Tamil Nadu"]
tn_cause = tn[tn["Category"] == "Causes"]
tn_edu = tn[tn["Category"] == "Education_Status"]
tn_adop = tn[tn["Category"] == "Means_adopted"]
tn_prof = tn[tn["Category"] == "Professional_Profile"]
tn_socio = tn[tn["Category"] == "Social_Status"]

tn_edu.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,0],color = "b", title = "Reason for suicide based on Education")
tn_adop.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[0,1],color = "r", title ="Reason for suicide based on Means adopted")
tn_prof.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,0],color = "g", title = "Reason for suicide based on Professional Profile")
tn_socio.groupby("TypeofCauses")["Total"].mean().sort_values(ascending = False)[:5].plot(kind = "bar", ax= ax[1,1],color = "c", title = "Reason for suicide based on Social Status")
plt.setp(ax[0,0], xlabel="")
plt.setp(ax[0,1], xlabel="")
plt.setp(ax[1,0], xlabel="")
plt.setp(ax[1,1], xlabel="")
f.suptitle("Most common reasons for committing suicide in Tamil Nadu" + "\n")
df_Age = df[df["Age_group"] != "0-100+"]
df_nonzero = df_Age[df_Age["Total"] != 0]
df_Age.groupby("Age_group")["Total"].sum().sort_values(ascending = False).plot(kind = "pie", explode = [0.02,0.02,0.02,0.02,0.02],
                                                                              autopct = "%3.1f%%", figsize = (15,7), shadow = False)
plt.title("Suicide rate based on Age group")
plt.ylabel("Different Age groups")
child = df[df["Age_group"] == "0-14"]
child.groupby(["TypeofCauses", "State"])["Total"].max().sort_values(ascending = False)[:10].plot(kind = "bar", figsize = (15,7), color = "m")
plt.title("Reasons for children who commit suicide")
plt.ylabel("No. of suicides")
f, ax = plt.subplots(2,2, figsize = (20,18), constrained_layout = True)
df[df["Age_group"] == "15-29"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[0,0], color = "g", title = "Age 15 to 29")
df[df["Age_group"] == "30-44"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[0,1], color = "c", title = "Age 30 to 44")
df[df["Age_group"] == "45-59"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[1,0], color = "b", title = "Age 45 to 59")
df[df["Age_group"] == "60+"].groupby("TypeofCauses")["Total"].sum().sort_values(ascending = False)[:10].plot(kind = "bar", fontsize = 10, ax = ax[1,1], color = "r", title = "Age 60+")
plt.setp(ax[0,0], xlabel = "")
plt.setp(ax[0,1], xlabel = "")
plt.setp(ax[1,0], xlabel = "")
plt.setp(ax[1,1], xlabel = "")
f.suptitle("Reasons for committing suicide among different Age groups" + "\n")
sns.barplot(data = df, x = "Category", y = "Total", hue = "Gender", palette = "viridis")
plt.xticks(rotation = 45)
plt.figure(figsize = (15,7))
edu = df[df["Category"] == "Education_Status"]

edu = edu[["TypeofCauses","Gender","Total"]]
edu_sort = edu.groupby(["TypeofCauses","Gender"],as_index = False).sum().sort_values(by="Total", ascending = False)
plt.figure(figsize=(15,7))
sns.barplot(data=edu_sort,x="TypeofCauses",y="Total",hue="Gender",palette = "viridis")
plt.xticks(rotation=45)
socio = df[df["Category"] == "Social_Status"]

socio = socio[["TypeofCauses", "Gender", "Total"]]
socio_sort = socio.groupby(["TypeofCauses","Gender"], as_index = False).sum().sort_values(by = "Total", ascending = False)
plt.figure(figsize = (15,7))
sns.barplot(data = socio_sort, x = "TypeofCauses", y = "Total", hue = "Gender", palette = "summer")
plt.xticks(rotation = 45)
df.groupby("Year")["Total"].sum().plot( kind = "line", figsize = (15,7))
