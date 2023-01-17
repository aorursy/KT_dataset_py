# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

print(os.getcwd())
bills = pd.read_csv("../input/house-of-representatives-congress-116/house_legislation_116.csv", index_col="bill_id")

members = pd.read_csv("../input/house-of-representatives-congress-116/house_members_116.csv", index_col="name_id")

attendance = pd.read_csv("../input/house-of-representatives-congress-116/house_rollcall_info_116.csv")

votes = pd.read_csv("../input/house-of-representatives-congress-116/house_rollcall_votes_116.csv", index_col="name_id")



def summarize(df):

    print("Columns:")

    print(df.columns)

    print("\n\n\n\n\nFirst 5 entries:")

    print(df.head())

    print("\n\n\n\n\nDescriptive Stats:")

    print(df.describe())

    print("\n\n\n\n\nMissing Info:")

    print(df.info())
summarize(bills)
summarize(members)
summarize(attendance)
summarize(votes)
print("policy_areas")

print(pd.unique(bills.policy_area))





print("\n\n\n\n\nsubjects")

unique_subjs = []

for sub_list in bills.subjects:

    

    sub_list= sub_list[1:-1].split("'")

    sub_list = [i for i in sub_list if (i != "" and i != ", ")]

    

    for i in pd.unique(sub_list):

        if i not in unique_subjs:

            unique_subjs.append(i)

print(unique_subjs)
# select your topics of interest here

major_topics = [p for p in pd.unique(bills["policy_area"]) if pd.notnull(p)] 



# make sure summaries are available, limit to topics of interest

bills.index = [i.lower() for i in bills.index]

bills = bills[bills.summary.notna()]

bills = bills[bills.policy_area.apply(lambda x: x in major_topics)]

bills = bills.drop(columns = ['title', 'sponsor', 'cosponsors', 'related_bills','subjects', 'committees', 'bill_progress',]) #'bill_id', 'policy_area',  'summary', 'date_introduced', 'number', 'bill_type'



# take out bills that aren't laws

import re

p = re.compile("[HS].((R.)|(j.res.))?[0-9]*.$",re.IGNORECASE) # should only allow bills starting with "H.R.", "H.J.Res.", "H.J.RES.", "S.", "S.J.RES.", in any letter case

bills = bills.loc[[pd.notnull(p.match(x)) for x in bills.index]]







# update topic list

major_topics = list(pd.unique(bills.policy_area))

print(major_topics, "\n\n\n", bills.shape)

print(pd.unique(bills.bill_type))
# keep only the translation columns, and drop any missing entries

attendance = attendance[['rollcall_id', 'bill_id']].dropna()

attendance.bill_id = attendance.bill_id.apply(lambda x: x.lower())



# get rid of duplicates

attendance = attendance.drop_duplicates(keep="last", subset="bill_id")



print(len(attendance.index))

print(pd.unique(attendance.index))
members = members.drop(columns = ['state', 'url', 'chamber', 'current_party', 'committee_assignments']) # might want these later, but drop them for now
ls1 = list(pd.unique(attendance.bill_id))

print(ls1)

ls2 = list(pd.unique(bills.index))

print(ls2)

ls3 = ls1 + ls2

ls4 = list(set(ls3)) # the more matches there are between attendance and bills, the smaller ls2 will be

print(f"ls3: {len(ls3)}\tls4: {len(ls4)}\tdiff:{len(ls3) - len(ls4)}")
# cut out entries in attendance that are not in bills or votes

attendance = attendance[attendance.bill_id.apply(lambda x: x in bills.index)]

attendance = attendance[attendance.rollcall_id.apply(lambda x: x in votes.columns)]

print(attendance.shape)



print(len(pd.unique(attendance.rollcall_id))) # bill_id has 125 unique values



# cut out bills that are not in attendance, and therefore not in votes

bills = bills.loc[set(attendance.bill_id)] # bills has 2000 rows, should be less than attendance

print(bills.shape)



# cut out votes that are not in attendance, and therefore not in bills

votes = votes[attendance.rollcall_id] # filter out the votes irrelevant bills

print(votes.shape)



# append rollcall to bills, to avoid intermediary

a = attendance.copy()

a.index = a.bill_id

a = a.drop(columns="bill_id")

bills = bills.join(a, how="left")



# make bill_id vote columns, to avoid intermediary

#a = attendance.copy()

##a.index = a["rollcall_id"]

#a = a.drop(columns="bill_id")

#votes. = bills.join(a, how="left")
print(votes.head())



# What are all the unique terms for voting? We need to convert strings to booleans

ls = []

for col in votes.columns:

    ls += list(pd.unique(votes[col]))

ls = pd.unique(ls)

print(ls)



# Simplify answers

def simplify_answers(x):

    votes_to_bool = {"Yes": ["Aye", "Yea", "Yes"], "No": ["No", "Nay"]}

    for key, ls in votes_to_bool.items():

        if x in ls:

            return key

    return "Abstain"

    

for col in votes.columns:

    votes[col] = votes[col].apply(simplify_answers)
print(bills.summary.iloc[0])

print("\n\n\npolicy_area")

print(bills.policy_area.iloc[0])
def naive(bills):

    progressiveness = [1 for _ in range(bills.shape[0])]

    bills["progressiveness"] = progressiveness

    return bills



bills = naive(bills)

bills.head()
MI = pd.MultiIndex.from_product([votes.index, ["Average"] + major_topics], names=("pol_id", "major_topic"))

pol_per_topic = pd.DataFrame(index = MI, columns = ["Pos", "Neg", "Abs"])
pos, neg, ab = 0, 0, 0

for topic in major_topics:

    b = list(bills.rollcall_id.loc[bills.policy_area == topic])

    print(b)

    

    for politician in votes.index:

        ab  = sum(votes.loc[politician, b] == "Abstain")

        pos = sum(votes.loc[politician, b] == "Yes")

        neg = sum(votes.loc[politician, b] == "No")

    

        pol_per_topic.loc[(politician, topic)] = (pos, neg, ab)



        

for politician in votes.index:

    ab  = sum(votes.loc[politician, b] == "Abstain")/len(major_topics)

    pos = sum(votes.loc[politician, b] == "Yes")/len(major_topics)

    neg = sum(votes.loc[politician, b] == "No")/len(major_topics)

        

    pol_per_topic.loc[(politician, "Average")] = (pos, neg, ab)
pol_per_topic
heatmap_cols = []

for subj in major_topics:

    for stance in ["Yes", "No", "Abstain"]:

        combined = "_".join([subj, str(stance)])

        heatmap_cols.append(combined)



        

def avg(x, stance, n_cols):

    if n_cols > 0:

        return sum(x == stance)/n_cols

    else:

        return 0

        

vote_heatmap = pd.DataFrame(columns = heatmap_cols, index = votes.index)

nvotes_by_topic = {}

for topic in major_topics:

    

    # get the votes on a given topic

    b = bills.rollcall_id.loc[bills.policy_area == topic]

    v = votes[b]

    nvotes_by_topic[topic] = len(b)

    #if len(b.columns) == 0: # drops nan columns

    #    heatmap = heatmap.drop(columns=[f"{topic}_Yes", f"{topic}_No", f"{topic}_Abstain"])

    #    continue

        

    # aggregate positions

    for stance in ["Yes", "No", "Abstain"]:

        col = "_".join([topic, str(stance)])

        vote_heatmap.loc[:, col] = v.apply(lambda x: avg(x, stance, len(v.columns)), axis=1) # empty as zero

        

vote_heatmap.index = members.loc[votes.index, "name"]
topics = []

counts = []

for topic, count in nvotes_by_topic.items():

    topics += [topic]

    counts += [count]

    #print(f"{topic}: {count}")



sns.set()

fig, ax = plt.subplots(figsize=(33,5))

sns.barplot(topics, counts, ax=ax)

ax.tick_params(axis='x', labelrotation=90)

fig.show()
# add spacer b/w topics

fig, ax = plt.subplots(figsize=(33,90))

sns.heatmap(vote_heatmap, ax=ax, mask=vote_heatmap.isnull())

ax.xaxis.tick_top()

ax.tick_params(axis='x', labelrotation=90)

fig.show()