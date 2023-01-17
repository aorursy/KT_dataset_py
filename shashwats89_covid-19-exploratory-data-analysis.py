import pandas as pd

case = pd.read_csv("../input/coronavirusdataset/Case.csv")

patient = pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

time = pd.read_csv("../input/coronavirusdataset/Time.csv")

timeage = pd.read_csv("../input/coronavirusdataset/TimeAge.csv")

timegender = pd.read_csv("../input/coronavirusdataset/TimeGender.csv")
#Using Seaborn to plot the graph

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
case.head(5)
patient.head(5)
patient['age'] = 2020 - patient['birth_year']

patient['age'].describe()
plt.figure(figsize=(10,6))

from scipy.stats import norm

sns.distplot(patient['age'], fit=norm);

#fig = plt.figure()

plt.show()
plt.figure(figsize=(10,6))

age_gender= pd.concat([patient['age'], patient['sex']], axis=1).dropna(); age_gender

sns.kdeplot(age_gender.loc[(age_gender['sex']=='female'), 

            'age'], color='g', shade=True, Label='female')



sns.kdeplot(age_gender.loc[(age_gender['sex']=='male'), 

            'age'], color='r', shade=True, Label='male')

plt.show()
df_patient=patient

infected_patient = patient.shape[0]

rp = patient.loc[patient["state"] == "released"].shape[0]

dp = patient.loc[patient["state"] == "deceased"].shape[0]

ip = patient.loc[patient["state"]== "isolated"].shape[0]

rp=rp/patient.shape[0]

dp=dp/patient.shape[0]

ip=ip/patient.shape[0]

print("The percentage of recovery is "+ str(rp*100) )

print("The percentage of deceased is "+ str(dp*100) )

print("The percentage of isolated is "+ str(ip*100) )
# Filtering patients that have released

released = df_patient[df_patient.state == 'released']

released.head()
plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the released")

sns.kdeplot(data=released['age'], shade=True,legend=True,cumulative=False,cbar=True,kernel='gau')

plt.show()
p = sns.countplot(data=patient,y = 'sex',saturation=1)
dead = df_patient[df_patient.state == 'deceased']

dead.head()
plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Age distribution of the deceased")

sns.kdeplot(data=dead['age'], shade=True,legend=True,cumulative=False,cbar=True)
p = sns.countplot(data=dead,y = 'sex',saturation=1)
p = sns.countplot(data=released,y = 'sex', saturation=1)

frames = [released, dead]

data1 = pd.concat(frames)

data1.head()
g = sns.catplot(x="sex", col="state",

                data=data1, kind="count",

                height=4, aspect=1);
dead.head()
released.head()
plt.figure(figsize=(6,4))

sns.set_style("darkgrid")

plt.title("Age distribution of the released vs dead")

ax = sns.kdeplot(data=released['age'],shade=True)

sns.kdeplot(dead['age'], ax=ax, shade=True,legend=True,cbar=True)

plt.show()
df_patient1 = df_patient[df_patient.state != 'isolated']

plt.figure(figsize=(6,4))

sns.set(style="whitegrid")

#tips = sns.load_dataset("dead")

#print(tips)

ax = sns.barplot(x="sex", y="age",hue='state', data=df_patient1)#,order=["age", "sex"])

df_patient2 = df_patient[df_patient.state == 'isolated']

plt.figure(figsize=(6,4))

sns.set(style="whitegrid")

#tips = sns.load_dataset("dead")

#print(tips)

ax = sns.barplot(x="sex", y="age", data=df_patient2)#,order=["age", "sex"])

patient.describe()
time.tail(5)
plt.figure(figsize=(15,6))

sns.set(style="darkgrid")

# Plot the responses for different events and regions

plt.xticks(rotation=90)

plt.title('seaborn-matplotlib time')

sns.lineplot(x="date", y="test",data=time)
timeage['age'] = timeage['age'].str.replace(r'\D', '').astype(int)

timeage.tail()
plt.figure(figsize=(15,6))

sns.set(style="darkgrid")

# Plot the responses for different events and regions

plt.xticks(rotation=15)

plt.title('seaborn-matplotlib timeage')

sns.lineplot(x="date", y="confirmed",data=timeage)
plt.figure(figsize=(15,6))

sns.set(style="darkgrid")

# Plot the responses for different events and regions

plt.xticks(rotation=15)

plt.title('seaborn-matplotlib timeage')

sns.lineplot(x="date", y="deceased",data=timeage)
timegender.tail()
plt.figure(figsize=(15,6))

sns.set(style="darkgrid")

# Plot the responses for different events and regions

plt.xticks(rotation=15)

plt.title('seaborn-matplotlib timeage')

sns.lineplot(x="date", y="confirmed",hue='sex',data=timegender)
plt.figure(figsize=(15,6))

sns.set(style="darkgrid")

# Plot the responses for different events and regions

plt.xticks(rotation=15)

plt.title('seaborn-matplotlib timeage')

sns.lineplot(x="date", y="deceased",hue='sex',data=timegender)