import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import time, os, warnings
color = sns.color_palette()
warnings.filterwarnings("ignore")
%matplotlib inline

inspections = pd.read_csv("../input/restaurant-and-market-health-inspections.csv")
violations = pd.read_csv("../input/restaurant-and-market-health-violations.csv")
inspections.head()
inspections.describe()
inspections.info()
fig, ax  = plt.subplots(2, 1, figsize = (10, 8))
sns.boxplot(inspections["score"], ax = ax[0])
ax[0].set_title("Box plot of Score", fontsize = 14)
ax[0].set_xlabel("")
sns.distplot(inspections["score"], kde = True, bins = 20, ax = ax[1])
ax[1].set_xlabel("score")
ax[1].set_title("Distribution of Score", fontsize = 14)
plt.show()
inspections["activity_date"] = pd.to_datetime(inspections["activity_date"])

inspect_date = pd.DataFrame({"date":inspections["activity_date"].value_counts().index, 
                             "values":inspections["activity_date"].value_counts().values}).sort_values(by = "date")
plt.figure(figsize = (10, 5))
plt.plot(inspect_date["date"], inspect_date["values"])
plt.title("Inspections Overview By Date", fontsize = 14)
plt.show()
inspections["year"] = inspections["activity_date"].dt.year
inspections["month"] = inspections["activity_date"].dt.month
inspections["day"] = inspections["activity_date"].dt.day
fig, ax = plt.subplots(3, 1, figsize = (10, 15))

for idx, time in enumerate(["year", "month", "day"]):
    temp = inspections[time].value_counts()
    sns.barplot(temp.index, temp.values, order = temp.index, ax = ax[idx])
    ax[idx].set_xlabel(time)
    ax[idx].set_ylabel("frequency")
    ax[idx].set_title("Inspection {} Frequency".format(time), fontsize = 14)
    rects = ax[idx].patches
    labels = temp.values
    for rect, label in zip(rects, labels):
        ax[idx].text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
                     label, ha = "center", va = "bottom")
plt.show()
inspections["grade"].unique()
inspections.loc[inspections["grade"] == ' ', "grade"] = "Unknown"
inspections["grade"].unique()
plt.figure(figsize = (10, 5))
grade = inspections["grade"].value_counts()
ax = sns.barplot(grade.index, grade.values)
plt.xlabel("grade")
plt.ylabel("frequency")
plt.title("Inspection Grade Frequency", fontsize = 14)
rects = ax.patches
labels = grade.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
inspections[inspections["grade"] == "Unknown"]
inspections.loc[inspections["grade"] == "Unknown", "grade"] = "C"
plt.figure(figsize = (10, 5))
inspections["service_code"].astype(str)
service_code = inspections["service_code"].value_counts()
ax = sns.barplot(service_code.index, service_code.values)
plt.xlabel("service code")
plt.ylabel("frequency")
plt.title("Top 20 Service Code Frequency", fontsize = 14)
rects = ax.patches
labels = service_code.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
inspections["service_description"].unique()
plt.figure(figsize = (10, 5))
service_description = inspections["service_description"].value_counts()
ax = sns.barplot(service_description.index, service_description.values)
plt.xlabel("service description")
plt.ylabel("frequency")
plt.title("Service Description Frequency", fontsize = 14)
rects = ax.patches
labels = service_description.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
second_inspect = inspections[inspections["service_description"] == "OWNER INITIATED ROUTINE INSPECT."]
plt.figure(figsize = (10, 5))
second_grade = second_inspect["grade"].value_counts()
ax = sns.barplot(second_grade.index, second_grade.values)
plt.xlabel("grade")
plt.ylabel("frequency")
plt.title("Second Inspection Grade Frequency", fontsize = 14)
rects = ax.patches
labels = second_grade.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
len(inspections["facility_zip"].unique())
inspections["facility_zip_pri"] = inspections["facility_zip"].apply(lambda x:x[:5])
inspections["facility_zip_pri"].unique()
plt.figure(figsize = (10, 5))
facility_zip = inspections["facility_zip_pri"].value_counts()
ax = sns.barplot(facility_zip.index[:20], facility_zip.values[:20], order = facility_zip.index[:20])
plt.xlabel("facility zip")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("First 20 Facility Zip Frequency", fontsize = 14)
rects = ax.patches
labels = facility_zip.values[:20]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
inspections["facility_name"].unique()
len(inspections["facility_name"].unique())
plt.figure(figsize = (10, 5))
facility_name = inspections["facility_name"].value_counts()
ax = sns.barplot(facility_name.index[:20], facility_name.values[:20], order = facility_name.index[:20])
plt.xlabel("facility name")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("Top 20 Facility Name Frequency", fontsize = 14)
rects = ax.patches
labels = facility_name.values[:20]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
inspections["program_name"].unique()
len(inspections["program_name"].unique())
plt.figure(figsize = (10, 5))
program_name = inspections["program_name"].value_counts()
ax = sns.barplot(program_name.index[:20], program_name.values[:20], order = program_name.index[:20])
plt.xlabel("program name")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("Top 20 Program Name Frequency", fontsize = 14)
rects = ax.patches
labels = program_name.values[:20]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
inspections["program_status"].unique()
plt.figure(figsize = (10, 5))
program_status = inspections["program_status"].value_counts()
ax = sns.barplot(program_status.index, program_status.values)
plt.xlabel("program status")
plt.ylabel("frequency")
plt.title("Program Status Frequency", fontsize = 14)
rects = ax.patches
labels = program_status.values[:15]
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
plt.figure(figsize = (10, 5))
pe_description = inspections["pe_description"].value_counts()
ax = sns.barplot(pe_description.index, pe_description.values, order = pe_description.index)
plt.xlabel("pe description")
plt.xticks(rotation = 90)
plt.ylabel("frequency")
plt.title("Pe Description Frequency", fontsize = 14)
rects = ax.patches
labels = pe_description.values
for rect, label in zip(rects, labels):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 5,
           label, ha = "center", va = "bottom")
plt.show()
plt.figure(figsize = (10, 5))
sns.countplot(x = "year", hue = "grade", data = inspections)
plt.xlabel("year")
plt.ylabel("frequency")
plt.title("Inspections Grade by Year", fontsize = 14)
plt.show()
years = sorted(inspections["year"].unique())
fig, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "month", hue = "grade", data = temp, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("month")
    ax[int(i/2)][i%2].set_ylabel("frequency")
    ax[int(i/2)][i%2].set_title("Inspections Grade per Month, {}".format(year), fontsize = 14)
plt.show()
fig, ax = plt.subplots(1, 2, figsize = (15, 5))

for i, year in enumerate(years[:2]):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "pe_description", hue = "grade", data = temp, ax = ax[i])
    ax[i].set_xlabel("pe description")
    ax[i].set_ylabel("frequency")
    ax[i].set_title("Inspections Grade by Pe description, {}".format(year), fontsize = 14)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.show()
fig, ax = plt.subplots(1, 2, figsize = (15, 5))

for i, year in enumerate(years[2:]):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "pe_description", hue = "grade", data = temp, ax = ax[i])
    ax[i].set_xlabel("pe description")
    ax[i].set_ylabel("frequency")
    ax[i].set_title("Inspections Grade by Pe description, {}".format(year), fontsize = 14)

for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation = 90)
    
plt.show()
fig, ax = plt.subplots(2, 2, figsize = (15, 18))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "program_status", hue = "grade", data = temp, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("program status")
    ax[int(i/2)][i%2].set_title("Inspections Program Status in {}".format(year), fontsize = 14)
plt.show()
fig, ax = plt.subplots(2, 2, figsize = (15, 18))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    temp = temp.groupby(["activity_date", "grade"]).score.mean()
    temp.unstack().plot(stacked = False, colormap = plt.cm.Set3,
                        grid = False, legend = True, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("activity date")
    ax[int(i/2)][i%2].set_title("Inspections Average Grade in {}".format(year), fontsize = 14)
plt.show()
fig, ax = plt.subplots(2, 2, figsize = (15, 10))

for i, year in enumerate(years):
    temp = inspections[inspections["year"] == year]
    sns.countplot(x = "service_description", hue = "grade", data = temp, ax = ax[int(i/2)][i%2])
    ax[int(i/2)][i%2].set_xlabel("service description")
    ax[int(i/2)][i%2].set_ylabel("frequency")
    ax[int(i/2)][i%2].set_title("Inspections Service Description in {}".format(year), fontsize = 14)
plt.show()
