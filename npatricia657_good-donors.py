# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
donor = pd.read_csv("../input/Donors.csv")
print(donor.info())
# sample of donors
donor[["Donor ID","Donor City","Donor State","Donor Is Teacher","Donor Zip"]].head(5)
# donors missing data
missing = donor.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)
# Donor City column has many missing data, therefore we view Donor State instead
donor["Donor State"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.ylabel("Number of donors")
plt.xlabel("State")
plt.title("Top 30 States")
# donor is teacher or not
donor["Donor Is Teacher"].value_counts().plot(kind="pie", autopct='%1.2f%%')
plt.legend(["Non Teacher","Teacher"])
plt.title("Percentage of Teachers vs Non")
# number of donors in different States
ct = donor.groupby(["Donor State","Donor Is Teacher"]).agg({"Donor Is Teacher":"count"}).rename(columns={"Donor Is Teacher" : "Teacher Donor Counts"}).reset_index()
pd.pivot_table(ct,index="Donor State",columns="Donor Is Teacher",values="Teacher Donor Counts").plot(kind="bar",figsize=(14,11))
plt.xlabel("State")
plt.ylabel("Number of Donors")
plt.title("Donors in different States")
donation = pd.read_csv("../input/Donations.csv")
print(donation.info())
# sample of donations
donation[["Project ID","Donation ID","Donor ID","Donation Included Optional Donation","Donation Amount","Donor Cart Sequence","Donation Received Date"]].head(5)
# donations missing data
missing = donation.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)
# donations included any optional donation
donation["Donation Included Optional Donation"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.legend(["Included","Not Included"])
plt.title("Percentage of Optional Donation")
# distribution of donation amount from 2012-2018 at DonorsChoose
donation["Donation Received Date"] = pd.to_datetime(donation["Donation Received Date"])
donation["Donation Year"] = donation["Donation Received Date"].dt.year
donation["Donation Year"] = donation["Donation Year"].astype(int)
donation.groupby("Donation Year")["Donation Amount"].sum().plot(kind="bar",rot=0)
plt.xlabel("Year")
plt.ylabel("Total Donations")
plt.title("Total Donations From 2012-2018")
# distribution of donors  
donation["Donor ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Donor ID")
plt.ylabel("Number of Donations")
plt.title("Top 30 Donors")
# distribution of projects  
donation["Project ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Project ID")
plt.ylabel("Number of Donors")
plt.title("Top 30 Projects")
donation.groupby(["Project ID","Donor ID","Donation Year"])["Donation Amount"].sum().reset_index()
donation.groupby("Donor ID")["Donation Amount"].sum().nlargest(30).reset_index()
donation.groupby("Project ID")["Donation Amount"].sum().nlargest(30).reset_index()
resource = pd.read_csv("../input/Resources.csv")
print(resource.info())
# sample of resources
resource[["Project ID","Resource Item Name","Resource Quantity","Resource Unit Price","Resource Vendor Name"]].head(5)
# resources missing data
missing = resource.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)
# remove missing data
resource1 = resource.copy()
resource1 = resource1.dropna(axis=0)
# group resource by Project ID
resource1.groupby("Project ID")["Resource Quantity","Resource Unit Price"].sum().sort_values(ascending=False,by="Resource Quantity").reset_index()[:100]
# Top 30 Resource Vendor Name
resource1["Resource Item Name"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
# Top 30 Resource Vendor Name
resource1["Resource Vendor Name"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
teacher = pd.read_csv("../input/Teachers.csv")
print(teacher.info())
# sample of teachers data
teacher[["Teacher ID","Teacher Prefix","Teacher First Project Posted Date"]].head(5)
# teachers missing data
missing = teacher.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)
# remove Teacher Prefix missing data
teacher["Teacher Prefix"] = teacher["Teacher Prefix"].astype(str)
teacher["Teacher Prefix"] = teacher["Teacher Prefix"].dropna()
teacher["Teacher Prefix"].value_counts().plot(kind="pie",autopct="%1.2f%%")
# create Teacher Gender based on Teacher Prefix
# assume Mx., Dr., and Teacher as Male
teacher["Teacher Gender"] = teacher["Teacher Prefix"].astype(str)
teacher["Teacher Gender"] = teacher["Teacher Gender"].fillna("Mr.")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Mrs.","Female")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Ms.","Female")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Mr.","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Dr.","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Mx.","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("Teacher","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].replace("nan","Male")
teacher["Teacher Gender"] = teacher["Teacher Gender"].astype(str)
teacher["Teacher Gender"].value_counts().plot(kind="pie",autopct="%1.2f%%")
# Teacher First Project
teacher["Teacher First Project Posted Date"] = pd.to_datetime(teacher["Teacher First Project Posted Date"])
teacher["Teacher Year"] = teacher["Teacher First Project Posted Date"].dt.year
teacher["Teacher Year"] = teacher["Teacher Year"].astype(int)
ty = teacher["Teacher Year"].value_counts()#.plot.bar(figsize=(10,8),rot=0)
plt.figure(figsize=(10,8))
plt.bar(np.arange(17),ty[sorted(ty.index)].values)
plt.xticks(np.arange(17),sorted(ty.index))
plt.xlabel("Year")
plt.ylabel("Number of Contributions")
plt.title("Teacher First Project from 2002-2018")
school = pd.read_csv("../input/Schools.csv")
print(school.info())
# sample of schools
school[["School ID","School Name","School Metro Type","School Percentage Free Lunch","School State","School Zip","School City","School County","School District"]].head(5)
# schools missing data
missing = school.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)
school1 = school.copy()
# impute median to School Percentage Free Lunch column
school1["School Percentage Free Lunch"].fillna(school1["School Percentage Free Lunch"].median())
# remove missing data in schools
school1 = school1.dropna(axis=0)
print(school1.info())
# School Metro Type
school1["School Metro Type"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of School Type")
# School State
school1["School State"].value_counts().plot.bar(figsize=(9,6))
plt.xlabel("State")
plt.ylabel("Number of Contributions")
plt.title("Top 30 States")
# schools provide free lunch
sc = school1["School Percentage Free Lunch"].astype(int).value_counts()
plt.figure(figsize=(14,13))
plt.bar(np.arange(50),sc[sorted(sc.index)].values[:50],width=0.45)
plt.xticks(np.arange(50),sorted(sc.index[:50]))
plt.xlabel("Percentage of Free Lunch")
plt.ylabel("Number of Schools")
plt.title("Distribution of schools provided free lunch")
project = pd.read_csv("../input/Projects.csv")
print(project.info())
# sample of projects data
project[["Project ID","School ID","Teacher ID","Teacher Project Posted Sequence","Project Type","Project Title","Project Essay","Project Short Description","Project Need Statement","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","Project Posted Date","Project Expiration Date","Project Current Status","Project Fully Funded Date"]].head(5)
# projects missing data
missing = project.copy()
missing = missing.T
missed = missing.isnull().sum(axis=1)
print(missed)
# impute missing data
project1 = project.copy()
project1 = project1.dropna(axis=0)
# Project Type
project1["Project Type"] = project1["Project Type"].dropna(axis=0)
project1["Project Type"].value_counts().plot(kind="pie",autopct="%1.2f%%",figsize=(6,6))
plt.title("Percentage of Project Type")
# School ID
project1["School ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("School ID")
plt.ylabel("Number of Schools")
plt.title("Top 30 Schools involved in Projects")
project1["Teacher ID"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Teacher ID")
plt.ylabel("Number of Teachers")
plt.title("Top 30 Teachers involved in Projects")
project1["Project Title"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Title")
plt.ylabel("Number of Titles")
plt.title("Top 30 Titles")
project1["Project Subject Category Tree"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Subject Category")
plt.ylabel("Number of Categories")
plt.title("Top 30 Subjects")
project1["Project Subject Category Tree"].value_counts().nlargest(5).plot(kind="pie",autopct="%1.2f%%",figsize=(6,5))
plt.title("Percentage of Top 5 Subjects")
project1["Project Subject Subcategory Tree"].value_counts().nlargest(30).plot.bar(figsize=(10,8))
plt.xlabel("Subject Subcategory")
plt.ylabel("Number of Subcategory")
plt.title("Top 30 Subcategories in Projects")
project1["Project Subject Subcategory Tree"].value_counts().nlargest(5).plot(kind="pie",autopct="%1.2f%%",figsize=(6,5))
plt.title("Percentage of Top 5 Subcategories")
project1["Project Resource Category"].value_counts().plot.bar(figsize=(8,6))
plt.xlabel("Resource Category")
plt.ylabel("Number of Resources")
plt.title("Top Resources")
project1["Project Grade Level Category"].value_counts().plot(kind="pie",autopct="%1.2f%%",figsize=(6,5))
plt.title("Percentage of Grade Category")
project1["Project Posted Date"] = pd.to_datetime(project1["Project Posted Date"])
project1["Project Posted Year"] = project1["Project Posted Date"].dt.year
sn.distplot(project1["Project Posted Year"])
plt.title("Project distribution from 2013-2018")
py = project1.groupby("Project Posted Year")["Project Cost"].sum()
plt.figure(figsize=(6,5))
plt.bar(np.arange(6),py[sorted(py.index)].values)
plt.xticks(np.arange(6),sorted(py.index))
plt.xlabel("Year")
plt.ylabel("Total Cost")
plt.title("Project Cost from 2013-2018")
pt = project1.groupby(["Project Posted Year","Project Subject Category Tree"])["Project Cost"].sum().reset_index()
p1 = pt[pt["Project Subject Category Tree"]=="Literacy & Language"]
p2 = pt[pt["Project Subject Category Tree"]=="Math & Science"]
idx = np.arange(6)
width = 0.35
fig,ax = plt.subplots(figsize=(8,6))
ax.bar(idx-width/2,p1["Project Cost"],width,color="SkyBlue",label="Literacy & Language")
ax.bar(idx+width/2,p2["Project Cost"],width,color="IndianRed",label="Math & Science")
ax.set_xticks(idx)
ax.set_xticklabels(p2["Project Posted Year"])
ax.set_xlabel("Posted Year")
ax.set_ylabel("Total Cost")
ax.legend(["Literacy & Language","Math & Science"])
plt.title("Project Cost from 2013-2018")
project1.groupby(["Project Posted Year","Project Type"])["Project Cost"].sum()
ps = pd.concat([project1,school1,teacher],axis=1,join_axes=[project1.index]).dropna(axis=0)
ps["School Metro Type"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of School Type")
ps["School State"].value_counts().plot.bar(figsize=(9,6))
plt.xlabel("State")
plt.ylabel("Number of Projects")
plt.title("Top States involved in DonorsChoose")
ps["Project Grade Level Category"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Grade in Projects")
ps["Project Resource Category"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Resource in Projects")
ps["Teacher Gender"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Teacher Gender in Projects")
ps["Teacher Prefix"].value_counts().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Teacher Prefix in Projects")
cl = ps[ps["School State"]=="California"][["Project Title","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","School Metro Type","Teacher Gender","Teacher Year"]]
cl["Teacher Year"] = cl["Teacher Year"].astype(int)
cl.groupby("Project Subject Category Tree")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on different Categories")
cl.groupby("Project Title")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Project Title")
cl.groupby("School Metro Type")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on School Type")
cl.groupby("Project Grade Level Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Grade Level")
cl.groupby("Project Resource Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Resource Category")
cl.groupby("Teacher Gender")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Teacher Gender")
cl.groupby("Teacher Year")["Project Cost"].sum().plot(kind="bar",rot=0,figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Project Cost")
plt.title("Histogram of Teacher First Project")
tx = ps[ps["School State"]=="Texas"][["Project Title","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","School Metro Type","Teacher Gender","Teacher Year"]]
tx["Teacher Year"] = tx["Teacher Year"].astype(int)
tx.groupby("Project Subject Category Tree")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on different categories")
tx.groupby("Project Title")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Project Title")
tx.groupby("School Metro Type")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on School Type")
tx.groupby("Project Grade Level Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Grade Level")
tx.groupby("Project Resource Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Resource Category")
tx.groupby("Teacher Gender")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Teacher Gender")
tx.groupby("Teacher Year")["Project Cost"].sum().plot(kind="bar",rot=0,figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Project Cost")
plt.title("Histogram of Teacher First Project")
ny = ps[ps["School State"]=="New York"][["Project Title","Project Subject Category Tree","Project Grade Level Category","Project Resource Category","Project Cost","School Metro Type","Teacher Gender","Teacher Year"]]
ny["Teacher Year"] = ny["Teacher Year"].astype(int)
ny.groupby("Project Subject Category Tree")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on different subjects")
ny.groupby("Project Title")["Project Cost"].sum().nlargest(5).plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Project Title")
ny.groupby("School Metro Type")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on School Type")
ny.groupby("Project Grade Level Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Grade Level")
ny.groupby("Project Resource Category")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Resource Category")
ny.groupby("Teacher Gender")["Project Cost"].sum().plot(kind="pie",autopct="%1.2f%%")
plt.title("Percentage of Project Cost on Teacher Gender")
ny.groupby("Teacher Year")["Project Cost"].sum().plot(kind="bar",rot=0,figsize=(10,6))
plt.xlabel("Year")
plt.ylabel("Project Cost")
plt.title("Histogram of Teacher First Project")