# import required libraries
import pandas as pd # for data wrangling
import numpy as np # for data processing
import matplotlib.pyplot as plt # for data visualization
# define filepath for dataset
filepath = "../input/simplilearn-projects/Comcast_telecom_complaints_data.csv"
# read file data into pandas dataframe
comcast_data = pd.read_csv(filepath, index_col="Ticket #")
# view the imported dataframe
comcast_data
# set the date format for manupaltion later, use dayfirst for correct formatting
comcast_data["Date"] = pd.to_datetime(comcast_data["Date"], format="%d-%m-%y")
comcast_data["Date"].dtype
# set month dataframe for plotting complaint volume against month
Month_complaints = comcast_data["Date"].dt.month
Month_complaints = Month_complaints.value_counts().sort_index()
Month_complaints
plt.figure(figsize=(24,10))
plt.plot(Month_complaints)
plt.show()
Date_complaints = comcast_data["Date"]
Date_complaints
Date_complaints = Date_complaints.value_counts().sort_index()
Date_complaints
plt.figure(figsize=(24,10))
plt.plot(Date_complaints.index.astype(str), Date_complaints.values)
plt.tick_params(axis="x", labelrotation=90)
plt.show()
comcast_data["Customer Complaint"].value_counts()
comcast_data["Status"].value_counts()
comcast_data["New Status"] = comcast_data["Status"].map({"Solved":"Closed", "Closed":"Closed", "Pending":"Open", "Open":"Open"})
comcast_data["New Status"].value_counts()
state = pd.crosstab(comcast_data["State"], comcast_data["New Status"])
state
state["Total"] = state.Open + state.Closed
state["Unresolved%"] = state.Open / state.Total
state
state[state["Total"] > 50]["Unresolved%"].idxmax()
state["Unresolved%"].idxmax()
channel = pd.crosstab(comcast_data["Received Via"], comcast_data["New Status"])
channel
channel["Total"] = channel.Open + channel.Closed
channel["Resolved%"] = channel.Closed / channel.Total
channel
channel["Resolved%"].idxmax()