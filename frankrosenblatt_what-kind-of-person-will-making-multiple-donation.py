import numpy as np
import pandas as pd

resources = pd.read_csv("../input/Resources.csv", sep=",").dropna()
schools = pd.read_csv("../input/Schools.csv", sep=",").dropna()
donors = pd.read_csv("../input/Donors.csv", sep=",").dropna()
donations = pd.read_csv("../input/Donations.csv", sep=",").dropna()
teachers = pd.read_csv("../input/Teachers.csv", sep=",").dropna()
projects = pd.read_csv("../input/Projects.csv", sep=",").dropna()
donor_value = donations["Donor ID"].value_counts()

once_donated_users = donor_value[donor_value == 1].index
multiple_donated_users = donor_value[donor_value > 1].index
once_donors_list = donors[donors["Donor ID"].isin(once_donated_users)]
multiple_donors_list = donors[donors["Donor ID"].isin(multiple_donated_users)]
import matplotlib.pyplot as plt

once_donors_list["Donor Is Teacher"].value_counts().plot(kind='bar')
plt.title("once_donated_users and is teacher")
multiple_donors_list["Donor Is Teacher"].value_counts().plot(kind='bar')
plt.title("multiple_donated_users and is teacher")
ov = once_donors_list["Donor State"].value_counts()
mv = multiple_donors_list["Donor State"].value_counts()
ov[0:30].plot(kind='bar')
plt.title("once_donated_users and state")
mv[0:30].plot(kind='bar')
plt.title("multiple_donated_users and state")
(ov / mv)[0:30].sort_values()[::-1].plot(kind='bar')
plt.title("Donation rate per states")