import pandas as pd

import glob

import os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import holoviews

from IPython.display import display



%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data=pd.read_csv('/kaggle/input/speed-dating-experiment/Speed Dating Data.csv', encoding= 'unicode_escape')

#data=pd.read_csv("Speed Dating Data.csv", encoding= 'unicode_escape')

pd.options.display.max_columns = None



data.rename(columns={"gender": "Gender", 

                     "condtn":"Condition",

                     "mn_sat":"median_sat",

                     "age_o":"age of partner",

                     "race_o":"race of partner",

                     "pf_o_att":"partner stated preference",

                     "dec_o":"partner's decision",

                     "attr_o":"partner rating of attributes",

                     "imprace":"race importance",

                     "imprelig":"religion importance",

                     "from":"originally from",

                     "date":"frequency of date"}, inplace=True)



race_replacement={1:"Black/African American",

                 2: "European/Caucasion-American",

                 3: "Latino/Hispanic American",

                 4: "Asian/Pacific Islander/Asian-American",

                 5: "Native American",

                 6: "Other"}



field_replacement={1:"Law",

                   2:"Math",

                   3:"Social Science, Psychologist",

                   4:"Medican Science, Pharmaceuticals, and Bio Tech",

                   5:"Engineering",

                   6:"English/Creative Writing / Journalism",

                   7:"History/ Religion/ Philosophy",

                   8:"Business/Econ/Finance",

                   9:"Education, Academia",

                   10:"Biological Sciences / Chemistry/ Physics",

                   11:"Social Work",

                   12:"Undergrad/undecided",

                   13:"Political Science/ International Affairs",

                   14:"Film",

                   15:"Fine Arts / Arts Administration",

                   16:"Languages",

                   17:"Architecture",

                   18:"Other"}



career_replacement={1: "Lawyer",

                   2:  "Academic / Research",

                   3:  "Psychologist",

                   4:  "Doctor/Medicine",

                   5:  "Engineer",

                   6:  "Creative Arts/ Entertainment",

                   7:  "Banking / Consulting /Finance / Marketing/ Business /CEO/ Entrepreneur / Admin",

                   8:  "Real Estate",

                   9:  "International /Humanitarian Affairs",

                   10:  "Undecided",

                   11:  "Social Work",

                   12:  "Speech Pathology",

                   13:  "Politics",

                   14:  "Pro Sports / Athletics",

                   15:  "Other",

                   16:  "Journalism",

                   17:  "Architecture"}



decision_replacement={1:"Yes", 0:"No"}

length_replacement={1:"Too little",

                   2: "Too much",

                   3: "Just Right"}

goal_replacement={1:"Seemed like a fan night out",

                 2: "To meet new people",

                 3: "To get a date",

                 4:"Looking for serious relationship",

                 5:"To say I did it",

                 6: "Other"}

frequency_replacement={1:"Several times a week",

                      2: "Twice a week",

                      3: "Once a week",

                      4: "Twice a month",

                      5: "Once a month",

                      6: "Several times a year",

                      7: "Almost never"}





#data["Gender"].replace({0:"Female",1:"Male"}, inplace=True)

data["Condition"].replace({1:"Limited choice", 2:"Extensive choice"}, inplace=True)

#data["match"].replace({1:"Yes",0:"No"}, inplace=True)

#data["samerace"].replace({1:"Yes",0:"No"},inplace=True)

#data["race of partner"].replace(race_replacement, inplace=True)

#data["partner's decision"].replace(decision_replacement, inplace=True)

#data["field_cd"].replace(field_replacement, inplace=True)

data["race"].replace(race_replacement, inplace=True)

#data["career_c"].replace(career_replacement, inplace=True)

#data["length"].replace(length_replacement, inplace=True)

#data["numdat_2"].replace({1:"Too few", 2: "Too many", 3: "Just Right"}, inplace=True)

#data["date_3"].replace({1:"Yes", 2:"No", 0:np.nan}, inplace=True)

data["goal"].replace(goal_replacement, inplace=True)

data["frequency of date"].replace(frequency_replacement, inplace=True)

data["go_out"].replace(frequency_replacement, inplace=True)



display(data)
def missing_values(df):

    missing=pd.DataFrame(df.isnull().sum()/len(data))*100

    missing.columns = ['missing_values(%)']

    missing['missing_values(numbers)'] = pd.DataFrame(df.isnull().sum())

    return missing.sort_values(by='missing_values(%)', ascending=False)

missing_values(data)

print(data.columns.tolist())
general_data = data[data.columns[:69]]

general_data.head()
dfl=pd.melt(data, id_vars=data.columns[:69], value_vars=data.columns[69:], var_name="Survey", value_name="Response")

dfl.head()
fig, (ax1, ax2)=plt.subplots(ncols=2, figsize=[12,8])

sns.countplot(data=data, hue ="Gender", x="dec", ax=ax1).set_title("Female and Male saying No and Yes");

sns.countplot(data=data, hue ="race", x="dec", ax=ax2).set_title("People with different ethnicity saying yes and no to their matches");
data.describe()
cor=data.corr(method="spearman")

corr_target=abs(cor["partner rating of attributes"])

corr_target[corr_target>=0.50]
sns.jointplot(y="partner rating of attributes", x="fun_o", data=data);

sns.jointplot(y="partner rating of attributes", x="like_o", data=data, );

cor=data.corr(method="spearman")

corr_target=abs(cor["iid"])

corr_target[corr_target>=0.70]
t=(data

 .groupby("iid")

 .count()

)

len(t)

(data[(data["wave"]>=6) & (data["wave"]<=9)]

.groupby("attr4_1")

 [["iid"]]

.count()

 .head(10)

)

## Being Attractive is extremely important for the wave 6-9
(data[(data["wave"]>=10) & (data["wave"]<=21)]

.groupby("attr4_1")

 [["iid"]]

.count()

 .head(10)

)
(data[(data["wave"]>=6) & (data["wave"]<=9)]

.groupby("attr2_3")

 [["iid"]]

.count()

 .head(10)

)

## Choice
(data

 .groupby("Condition")

 [["iid"]]

 .count()

).plot.pie(y="iid", autopct='%1.0f%%')



## Only 17% of participants had limited choice
## Still attractiveness is important
cor=data.corr(method="spearman")

corr_target=abs(cor["partner's decision"])

corr_target[corr_target>=0.50]
sns.jointplot(data=data, x="like_o", y="partner's decision")



sns.regplot(data=data, x="like_o", y="partner's decision")
## More they liked the person, more they could say Yes
cor=data.corr(method="spearman")

corr_target=abs(cor["field_cd"])

corr_target[corr_target>=0.50]
## Which field got more match
(

data

    .groupby(["field", "wave"])

    [["match"]]

   .count()

    .sort_values("match", ascending=False)

    .head(100)

)



# For different waves, different professions were in demand: MBA. International Affairs, Law. Business and Social work----

# People with professions like Mathematics, Art Education, Chemistry, Medical Informatics got the least of matches
(

data

    .groupby(["race"])

    [["match", "iid"]]

   .count()

    .sort_values("match", ascending=False)

    .head(100)

).plot.pie(y="iid", autopct='%1.0f%%')
## Most of the matches were with European/Caucasion, American, The Least were Black/African american
(data

 .groupby("race importance")

 .size()

).plot.pie(y="iid",autopct='%1.0f%%')



## race is not so important for matching
(data

 .groupby("religion importance")

 .size()

).plot.pie(y="iid",autopct='%1.0f%%')



## religion is not so important for matching
(data

 .groupby("goal")

 [["match"]]

 .count()

).plot.pie(y="match", autopct='%1.0f%%')



# People who had a goal to have a fan night out and meet new people got the most of the matches
(data

 .groupby("frequency of date")

 [["match"]]

 .count()

).plot.pie(y="match", autopct='%1.0f%%')



#25% of people who go on a date twice a month, or several times a year got more matches

(data

 .groupby("go_out")

 [["match"]]

 .count()

).plot.pie(y="match", autopct='%1.0f%%')

#Those go out frequently got more matches
cor=data.corr(method="spearman")

corr_target=abs(cor["exphappy"])

corr_target[corr_target>=0.50]
(data

 .groupby("exphappy")

 .count()

).plot.pie(y="iid", autopct='%1.0f%%')



# Moderate and mosttly happy

d=(

data

.groupby("expnum")

    [["iid"]]

.count()

).plot.bar()



sns.distplot(data["expnum"])



#People expected to have approximately 3-4 dates after the experiment
(data

.groupby("date_3")

 .count()

).plot.pie(y="iid", autopct='%1.0f%%')



# 62% of participants did not get any dates from the experiments
(data

.groupby("you_call")

 .count()

).plot.pie(y="iid", autopct='%1.0f%%')



# Most of participants did not reach out their dates
(data

.groupby("them_cal")

 .count()

).plot.pie(y="iid", autopct='%1.0f%%')



#50% of participants did not receive calls from their matchers, 27% received only 1 call
