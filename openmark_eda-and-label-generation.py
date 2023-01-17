import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/ptb-diagnostic-ecg-database/meta.csv", 

                 usecols=["patient", "record_id", "fs", "sig_len", "age", "Reason_for_admission"])
df.head()
print("{} recordings from {} users have NaN diagnosis".format(

    df.loc[df.Reason_for_admission.isnull(), "patient"].size,

    df.loc[df.Reason_for_admission.isnull(), "patient"].nunique()

))
before = df.shape[0]

df = df.drop(df[df.Reason_for_admission.isnull()].index)

print("Before: {}, After: {}, {} records removed".format(before, df.shape[0], before - df.shape[0]))
df.groupby(["patient"])['record_id'].nunique().value_counts().sort_index()
_t = df.groupby(["patient"])['Reason_for_admission'].nunique().eq(1).all()

print("Different recordings of same patient are marked with same diagnosis:", _t)
ages = df.groupby(["patient"])['age'].apply(lambda x: list(np.unique(x))[0])

print("Age is not specified for {} patients".format(ages.isna().sum()))
ages.describe()
ax = ages.plot.hist(title="Age distribution")

ax.set_xlabel("Age")
df["signal_duration"] = df["sig_len"] / df["fs"]



groupby_diagnosis = df.groupby(["Reason_for_admission"])['patient', 'signal_duration']

groupby_diagnosis = groupby_diagnosis.agg({"patient": 'nunique', "signal_duration": "sum"})



fig, axes = plt.subplots(nrows=1, ncols=2)



axes[0].yaxis.label.set_visible(False)

axes[1].yaxis.label.set_visible(False)



groupby_diagnosis["patient"].sort_values(ascending=True).plot(ax=axes[0], kind='barh', grid=True, figsize=(10,5), 

                                         title="Number of patients per diagnosis")



for p in axes[0].patches:

    axes[0].annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 0), textcoords='offset points')



groupby_diagnosis["signal_duration"].sort_values(ascending=True).plot(ax=axes[1], kind='barh', grid=True, figsize=(10,5), 

                                         title="Total recording time per diagnosis, sec.")



for p in axes[1].patches:

    val = "{:.1f}".format(p.get_width())

    axes[1].annotate(val, (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 0), textcoords='offset points')

    

fig.tight_layout()
df["myocardial_infarction"] = (df.Reason_for_admission == "Myocardial infarction").astype(int)
_t = df.groupby(["myocardial_infarction"])["patient"].nunique()

ax = _t.sort_values(ascending=True).plot(kind='barh', grid=True, 

                                      title="Number of patients with MI (1) and no MI (0)")

# ax.set_yticklabels(["MI", "Non-MI"], rotation=0)

for p in ax.patches:

    val = "{:.1f}".format(p.get_width())

    ax.annotate(val, (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 5), textcoords='offset points')
_t = df.groupby(["myocardial_infarction"])["signal_duration"].sum()

ax = _t.sort_values(ascending=True).plot(kind='barh', grid=True, 

                                      title="Total recording time for patients with MI (1) and no MI (0), sec.")

# ax.set_yticklabels(["MI", "Non-MI"], rotation=0)

for p in ax.patches:

    val = "{:.1f}".format(p.get_width())

    ax.annotate(val, (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 5), textcoords='offset points')
df['age'].hist(by=df['myocardial_infarction'], stacked=True)

plt.suptitle('Age distribution for patients with no MI (0) and MI (1)', y=1.05)
df["dysrhythmia"] = (df.Reason_for_admission == "Dysrhythmia").astype(int)
_t = df.groupby(["dysrhythmia"])["patient"].nunique()

ax = _t.sort_values(ascending=True).plot(kind='barh', grid=True, 

                                      title="Number of patients with Dysrhythmia (1), and no Dysrhythmia (0)")

# ax.set_yticklabels(["MI", "Non-MI"], rotation=0)

for p in ax.patches:

    val = "{:.1f}".format(p.get_width())

    ax.annotate(val, (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 5), textcoords='offset points')
_t = df.groupby(["dysrhythmia"])["signal_duration"].sum()

ax = _t.sort_values(ascending=True).plot(kind='barh', grid=True, 

                                      title="Total recording time for patients with \n Dysrhythmia (1) and no Dysrhythmia (0), sec.")

# ax.set_yticklabels(["MI", "Non-MI"], rotation=0)

for p in ax.patches:

    val = "{:.1f}".format(p.get_width())

    ax.annotate(val, (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 5), textcoords='offset points')
df.head()
df.drop(labels=["Reason_for_admission", "signal_duration"], axis=1, inplace=True)

df.to_csv("../working/labels.csv", index=False)