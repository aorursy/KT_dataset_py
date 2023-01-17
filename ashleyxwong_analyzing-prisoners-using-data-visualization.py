import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

crime = pd.read_csv("../input/daily-inmates-in-custody.csv")
crime.tail()
crime.info()
df = crime["AGE"].dropna()
df.plot.hist()
plt.title("Ages of Prisoners Distribution")
plt.xlabel("Age")
plt.savefig("AgeDistribution.png")
df = crime["AGE"].dropna()
sum(df.between(20,40))/len(df)
sns.countplot(x = "GENDER", data = crime)
plt.title("Genders of Prisoners Distribution")
plt.savefig("GenderDistribution.png")
df = crime["GENDER"].dropna()
df = df.map({'M' :1, 'F' :0})
sum(df)/len(df)
df = crime["RACE"].dropna()
race_df = pd.get_dummies(df)
race_df.head()
heights = []
bars = ['A', 'B', 'I', 'O', 'U', 'W']
for i in list(race_df.columns.values):
    heights.append(race_df[i].sum())
y_pos = np.arange(len(bars))
figure = plt.bar(y_pos, heights)
plt.xticks(y_pos, bars)
plt.title("Bar Plot of Prisoner Race")
plt.show()

plt.savefig('race_bar.png')
df = crime["RACE"].dropna()
df = df.map({'B' :1, 'A' :0, 'I' :0, 'O' :0, 'U' :0, 'W' :0})
sum(df)/len(df)
sns.countplot(x = "INFRACTION", data = crime)
plt.title("Number of Infractions Committed")
plt.savefig("NumInfractions.png")
df = crime['INFRACTION'].dropna()
df = df.map({'Y' :1, 'N' :0})
sum(df)/len(df)
sns.countplot(x = "BRADH", data = crime)
plt.title("Number of Prisoners under Mental Observation")
plt.xlabel("Mental Observation")
plt.savefig("MentalObservation.png")
df = crime['BRADH'].dropna()
df = df.map({'Y' :1, 'N' :0})
sum(df)/len(df)
sns.countplot(x = "SRG_FLG", data = crime)
plt.title("Number of Prisoners with Gang Affiliation")
plt.xlabel("Gang Affiliation")
plt.savefig("GangAffiliation.png")
df = crime['SRG_FLG'].dropna()
df = df.map({'Y': 1, 'N': 0})
sum(df)/len(df)
sns.countplot(x = 'CUSTODY_LEVEL', hue = 'RACE', data = crime)
plt.title('Level of Security of Prisoners by Race')
plt.savefig('securityvsrace.png')
sns.countplot(x = 'CUSTODY_LEVEL', hue = 'GENDER', data = crime)
plt.title('Level of Security of Prisoners by Gender')
plt.savefig('securityvsgender.png')
sns.countplot(x = "BRADH", hue = 'GENDER', data = crime)
plt.title("Number of Prisoners under Mental Observation by Gender")
plt.savefig("MentalObservationbyGender.png")
sns.countplot(x = "BRADH", hue = 'RACE', data = crime)
plt.title("Number of Prisoners under Mental Observation by Race")
plt.savefig("MentalObservationRace.png")
sns.countplot(x = "BRADH", hue = "INFRACTION", data = crime)
plt.title("Mental Observation by Infractions Committed")
plt.tight_layout()
plt.savefig("MentalObservationInfraction.png")
sns.countplot(x = "SRG_FLG", hue = "RACE", data = crime)
plt.title("Number of Prisoners with Gang Affiliation by Race")
plt.xlabel("Gang Affiliation")
plt.savefig("GangAffiliationRace.png")
sns.countplot(x = "SRG_FLG", hue = "GENDER", data = crime)
plt.title("Number of Prisoners with Gang Affiliation by Gender")
plt.xlabel("Gang Affiliation")
plt.savefig("GangAffiliationGender.png")
#### There exists a similar ratio between genders and having gang affiliations.  