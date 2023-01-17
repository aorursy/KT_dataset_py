import datetime

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
records = pd.read_csv("../input/scirate_quant-ph_unnormalized.csv", dtype={"id": str},

                      index_col=0)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(13, 6))

sns.stripplot(x="day", y="scites", data=records, jitter=True, ax=ax1)

sns.stripplot(x="month", y="scites", data=records, jitter=True, ax=ax2)

sns.stripplot(x="year", y="scites", data=records, jitter=True, ax=ax3)

plt.show()
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(13, 6))

sns.regplot(x="day", y="scites", data=records, x_estimator=np.mean, ax=ax1)

sns.regplot(x="month", y="scites", data=records, x_estimator=np.mean, ax=ax2)

plt.show()
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

records.insert(4, "day of week", "")

records.insert(5, "nday of week", "")

for i, row in records.iterrows():

    date = datetime.datetime(row["year"], row["month"], row["day"])

    records.set_value(i, "day of week", days[date.weekday()])

    records.set_value(i, "nday of week", date.weekday())

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(13, 6))

sns.stripplot(x="scites", y="day of week", data=records, jitter=True, ax=ax1)

sns.violinplot(x="scites", y="day of week", order=days[:-2], data=records[records["nday of week"]<5], ax=ax2)

plt.show()    
def daterange(start_date, end_date):

    for n in range(int((end_date - start_date).days)):

        yield start_date + datetime.timedelta(n)



start_date = datetime.date(2012, 1, 1)

end_date = datetime.date(2016, 12, 31)

daily_order = []

for date in daterange(start_date, end_date):

    same_day = records[(records["year"] == date.year) &

                       (records["month"] == date.month) &

                       (records["day"] == date.day)].sort_values("id")

    total = len(same_day)

    daily_order += [[total, i, scites] for i, scites in enumerate(same_day["scites"])]

daily_order = pd.DataFrame(daily_order, columns=["total", "rank", "scites"])
fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(13, 6))

sns.stripplot(x="rank", y="scites", data=daily_order, jitter=True, ax=ax1)

sns.regplot(x="rank", y="scites", data=daily_order, x_estimator=np.mean, ax=ax2)

plt.show()
records.insert(7, "number of authors", 0)

for i, row in records.iterrows():

    records.set_value(i, "number of authors", len(row["authors"].split(",")))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))

records["number of authors"].hist(bins=20, ax=ax1)

sns.stripplot(x="number of authors", y="scites", data=records, jitter=True, ax=ax2)

sns.regplot(x="number of authors", y="scites", data=records, x_estimator=np.mean, ax=ax3)

sns.regplot(x="number of authors", y="scites", data=records, order=2, x_estimator=np.mean, ax=ax4)
records.insert(8, "length of title", 0)

records.insert(9, "words in title", 0)

for i, row in records.iterrows():

    records.set_value(i, "length of title", len(row["title"]))

    records.set_value(i, "words in title", len(row["title"].split(" ")))

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13, 6))

sns.regplot(x="length of title", y="scites", data=records, x_estimator=np.mean, ax=ax1)

sns.regplot(x="words in title", y="scites", data=records, x_estimator=np.mean, ax=ax2)