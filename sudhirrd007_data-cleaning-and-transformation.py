from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

filepath = "../input/enron-email-dataset/emails.csv"
emails = pd.read_csv(filepath, skiprows=lambda x:x%2)
# >>> this file is too big(around 0.5M rows) and take much time in our analysis
# so if you are begineer then only extract half of data using given skiprows parameter
emails.shape
cols = emails.columns
emails.head(3)
print(emails.loc[0]["message"])
# >>> We can see lots of fields into message, but we only need some useful fields for our model. 
# so lets see which kind of fields a email contain
import email
message = emails.loc[0]["message"]
e = email.message_from_string(message)
e.items()

e.get("Date")
# >>> below are fields and corresponding value of those, 
# keep in mind here message text isn't extract into this var e
# we need only "Date", "Subject", "X-Folder", "X-From", "X-To"
# show message body
e.get_payload()
# now we add those fields into our emails DataFrame
def get_field(field, messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get(field))
    return column
emails["date"] = get_field("Date", emails["message"])
emails["subject"] = get_field("Subject", emails["message"])
emails["X-Folder"] = get_field("X-Folder", emails["message"])
emails["X-From"] = get_field("X-From", emails["message"])
emails["X-To"] = get_field("X-To", emails["message"])
emails.head(3)
def body(messages):
    column = []
    for message in messages:
        e = email.message_from_string(message)
        column.append(e.get_payload())
    return column

emails["body"] = body(emails["message"])
emails.head(3)
emails["file"][:10]
# >>> see row 0, allen-p is user name who sent email
# we will add column named employee into emails
def employee(file):
    column = []
    for string in file:
        column.append(string.split("/")[0])
    return column

emails["employee"] = employee(emails["file"])
emails.head(3)
"number of folders: ", emails.shape[0]
"number of unique folders: ", emails["X-Folder"].unique().shape[0]

# >>>>> means there are lots of same X-Folder used by employee
unique_emails = pd.DataFrame(emails["X-Folder"].value_counts())
unique_emails.reset_index(inplace=True)
unique_emails.columns = ["folder_name", "count"]
unique_emails
sns.barplot(x="count", y="folder_name", data=unique_emails.iloc[:20, :])
plt.xlabel("count")
plt.ylabel("folder_name")
plt.show();
# let's see top 20 highest email sender employee

top_20 = pd.DataFrame(emails["employee"].value_counts()[:20])
top_20.reset_index(inplace=True)
top_20.columns = ["employee_name", "count"]
top_20
sns.barplot(y="employee_name", x="count", data=top_20)
plt.xlabel("Number of emails send")
plt.ylabel("Employee Name")
plt.show();
# let's convert date column type(which is string) to date type
# for this we will use dateutil module : http://labix.org/python-dateutil#head-a23e8ae0a661d77b89dfb3476f85b26f0b30349c

import datetime
from dateutil import parser

# this is sample example
x = parser.parse("Fri, 4 May 2001 13:51:00 -0700 (PDT)")
print(x.strftime("%Y-%m-%d %H:%M"))
def change_type(dates):
    column = []
    for date in dates:
        column.append(parser.parse(date).strftime("%Y-%m-%d %H:%M"))
    return column

emails["date"] = change_type(emails["date"])
emails.head(3)

# Alternative way of doing (but takes more time :< )
# emails["date"] = pd.to_datetime(emails["date"])
# emails.head(3)
emails["X-Folder"][0]
# we only want last folder name
emails["X-Folder"][0].split("\\")[-1]
def preprocess_folder(folders):
    column = []
    for folder in folders:
        if(folder is None or folder == ""):
            column.append(np.nan)
        else:
            column.append(folder.split("\\")[-1].lower())
    return column

emails["X-Folder"] = preprocess_folder(emails["X-Folder"])
emails.head(3)
# emails["X-Folder"].unique()[:10]

# # Folders we can filter out
# unwanted_folders = ["all documents", "deleted items", "discussion threads", "sent", "deleted Items", "inbox",
#                    "sent items", "'sent mail", "untitled", "notes inbox", "junk file", "calendar"]

# emails["X-Folder"] = emails.loc[~emails["X-Folder"].isin(unwanted_folders)]["X-Folder"]
# emails.head(3)
# # >>>>> It will add NaN where unwanted_folder names occured
emails.isnull().sum()
emails.dropna(inplace=True)
emails.isnull().sum()
