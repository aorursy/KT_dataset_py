import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

print("Loaded Libraries")
donors = pd.read_csv("../input/Donors.csv", low_memory=False)
donations = pd.read_csv("../input/Donations.csv", low_memory=False)
projects = pd.read_csv("../input/Projects.csv", low_memory=False)
schools = pd.read_csv("../input/Schools.csv", low_memory=False)
teachers = pd.read_csv("../input/Teachers.csv", low_memory=False)
resources = pd.read_csv("../input/Resources.csv", low_memory=False)

print("Loaded Data")
print("Every donator donates on average " + str(len(donations) / len(donors)) + " times")
donationsPerDonor = donations.groupby("Donor ID")["Donation ID"].nunique().sort_values(ascending=False)
print("How many times have the top donators donated?")
donationsPerDonor.head(10)
print("Histogram for the number of donations per donator.")
plt.figure(figsize=(20,10))
plt.hist(donationsPerDonor[donationsPerDonor < 20], bins=20)
donationsPerDonor[donationsPerDonor == 1] = 0
donationsPerDonor[donationsPerDonor != 0] = 1
donationsPerDonor = pd.DataFrame(donationsPerDonor).reset_index()
donationsPerDonor.columns = ["Donor ID", "Redonator"]
donors = donors.merge(donationsPerDonor, on="Donor ID", how="left")
donors["Redonator"] = donors["Redonator"].fillna(0).astype(int)
print("We have " + str(len(donors[donors["Redonator"] == 0])) + " donators who donated only once.")
print("We have " + str(len(donors[donors["Redonator"] == 1])) + " donators who donated more than one time.")
donors["Donor State"] = pd.Series(donors["Donor State"], dtype="category")
donors["Donor Is Teacher"] = pd.Series(donors["Donor Is Teacher"], dtype="category")
le = preprocessing.LabelEncoder()
le.fit(donors["Donor State"])
donors["Donor State"] = le.transform(donors["Donor State"])
le.fit(donors["Donor Is Teacher"])
donors["Donor Is Teacher"] = le.transform(donors["Donor Is Teacher"])
corr = donors.corr()
sns.heatmap(corr)
print("Merge all data into donations")
projects = projects.merge(schools, on = "School ID", how="left")
projects = projects.merge(teachers, on = "Teacher ID", how="left")
donations = donations.merge(projects,on="Project ID", how="left")
donations = donations.merge(donors, on = "Donor ID", how="left")
donations.head()
print("Preprocessing / Encoding the data, s.t. correlation calculation become feasible")

#donations["Project Cost"] = donations["Project Cost"].str.replace("$", "")
#donations["Project Cost"] = pd.to_numeric(donations["Project Cost"].str.replace(",", ""))

#donors = donors.drop(["School ID", "Teacher ID", "Project ID", "Donor ID", "Donation ID"], axis = 1)

donations["Project Subject Category Tree"] = donations["Project Subject Category Tree"].fillna("unknown")
le.fit(donations["Project Subject Category Tree"])
donations["Project Subject Category Tree"] = le.transform(donations["Project Subject Category Tree"])

donations["Donation Included Optional Donation"] = donations["Donation Included Optional Donation"].fillna("unknown")
le.fit(donations["Donation Included Optional Donation"])
donations["Donation Included Optional Donation"] = le.transform(donations["Donation Included Optional Donation"])

donations["Project Resource Category"] = donations["Project Resource Category"].fillna("unknown")
le.fit(donations["Project Resource Category"])
donations["Project Resource Category"] = le.transform(donations["Project Resource Category"])

donations["Project Type"] = donations["Project Type"].fillna("unkown")
le.fit(donations["Project Type"])
donations["Project Type"] = le.transform(donations["Project Type"])

donations["Project Current Status"] = donations["Project Current Status"].fillna("unknown")
le.fit(donations["Project Current Status"])
donations["Project Current Status"] = le.transform(donations["Project Current Status"])

donations["School Metro Type"] = donations["School Metro Type"].fillna("unknown")
le.fit(donations["School Metro Type"])
donations["School Metro Type"] = le.transform(donations["School Metro Type"])

donations["Teacher Prefix"] = donations["Teacher Prefix"].fillna("unknown")
le.fit(donations["Teacher Prefix"])
donations["Teacher Prefix"] = le.transform(donations["Teacher Prefix"])

donations["Project Grade Level Category"] = donations["Project Grade Level Category"].fillna("unknown")
le.fit(donations["Project Grade Level Category"])
donations["Project Grade Level Category"] = le.transform(donations["Project Grade Level Category"])

donations["Project Subject Subcategory Tree"] = donations["Project Subject Subcategory Tree"].fillna("unknown")
le.fit(donations["Project Subject Subcategory Tree"])
donations["Project Subject Subcategory Tree"] = le.transform(donations["Project Subject Subcategory Tree"])

donations["School State"] = donations["School State"].fillna("unkown")
le.fit(donations["School State"])
donations["School State"] = le.transform(donations["School State"])

donations["School County"] = donations["School County"].fillna("unknown")
le.fit(donations["School County"])
donations["School County"] = le.transform(donations["School County"])

print("Correlation Plot: ")

corr = donations.corr()
fig = plt.figure(figsize=(12,10))
sns.heatmap(corr)
print("Correlations shown as numerical values:")
pd.DataFrame(corr["Redonator"])
print("Plotting the development of donations")
plt.figure(figsize=(20,10))

time = donations[["Donation Received Date", "Donation Amount", "Redonator"]]
series0 = time.groupby("Donation Received Date")["Donation Amount"].sum()
series1 = time.groupby("Donation Received Date")["Donation Amount"].nunique()
time = time[time["Redonator"] == 1]
series2 = time.groupby("Donation Received Date")["Donation Amount"].sum()
series3 = time.groupby("Donation Received Date")["Donation Amount"].nunique()

ts = pd.Series(series0)
ts = ts.cumsum()
ts1 = pd.Series(series1)
ts1 = ts1.cumsum()
ts2 = pd.Series(series2)
ts2 = ts.cumsum()
ts3 = pd.Series(series3)
ts3 = ts3.cumsum()

plt.subplot(221)
plt.title("Total donation amount over time")
ts.plot()

plt.subplot(222)
plt.title("Total number of donations over time")
ts1.plot()

plt.subplot(223)
plt.title("Total donation amount of redonators over time")
ts2.plot()

plt.subplot(224)
plt.title("Total number of donations of redonators over time")
ts3.plot()
redonations = donations[["Donor ID", "School ID", "Teacher ID", "Project Subject Category Tree", "Donor Is Teacher", "Donation Received Date", "Donation Amount", "Project Type", "Redonator"]]
redonations = redonations[redonations["Redonator"]==1]
series0 = redonations.groupby("Donor ID")["School ID"].nunique()
series1 = redonations.groupby("Donor ID")["Donation Amount"].std()
#series2 = redonations.groupby("Donor ID")["Donation Amount"].mean()
series3 = redonations.groupby("Donor ID")["Teacher ID"].nunique()
series4 = redonations.groupby("Donor ID")["Project Subject Category Tree"].nunique()
print("Histogram: How many different schools does a redonator donate to?")
plt.figure(figsize=(20,7))
plt.hist(series0[series0 < 20], 20)
print("Histogram: How many different teachers does a redonator donate to?")
plt.figure(figsize=(20,7))
plt.hist(series3[series3 < 20], 20)
print("Average standard deviation of donated amount: " + str(series1.mean()))