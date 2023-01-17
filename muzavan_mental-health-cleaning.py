# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/survey.csv")

data.head
data.drop(labels="Timestamp",inplace=True,axis=1)

data.head
for c in data.columns.values:

    arr = data[c].unique()

    print("{c:20} : {a}".format(c=c,a=arr))

# Age related

age_col = "Age"

data.loc[data[age_col] < 0, age_col] = None

data.loc[data[age_col] > 60, age_col] = None

data.loc[data[age_col].isnull(),age_col] = data[age_col].mean()

data[age_col] = np.floor(data[age_col])



# Gender related, will convert female to 0, and male to 1

gender_col = "Gender"

data[gender_col].fillna(value=0.5,inplace=True)# We are not sure

data.loc[data[gender_col].str.lower().isin(["female","woman"]), gender_col] = 0

data.loc[data[gender_col].str.lower().str.startswith("f",na=False), gender_col] = 0 # The typo and stuffs

data.loc[data[gender_col].str.lower().str.contains("fe",na=False), gender_col] = 0 # The typo and stuffs

data.loc[data[gender_col].str.lower().str.contains("wo",na=False), gender_col] = 0 # The typo and stuffs

data.loc[data[gender_col].str.lower().isin(["male","man"]), gender_col] = 1

data.loc[data[gender_col].str.lower().str.startswith("m",na=False), gender_col] = 1 # The typo and stuffs

data.loc[data[gender_col].str.lower().str.contains("ma",na=False), gender_col] = 1 # The typo and stuffs

data.loc[data[gender_col].str.lower().str.contains("guy",na=False), gender_col] = 1 # The typo and stuffs

data.loc[~data[gender_col].isin([0,1]), gender_col] = 0.5



# Yes/No Column

cols = [

"family_history","self_employed","treatment",

"remote_work","tech_company",

"benefits","wellness_program",

"seek_help","anonymity","mental_health_interview",

"mental_health_consequence","phys_health_consequence",

"phys_health_interview","mental_vs_physical","obs_consequence"

]

for col in cols:

    data.loc[data[col].str.lower().str.startswith("d",na=False) | data[col].str.lower().str.startswith("m",na=False) ,col] = 0.5 # don't know or maybe

    data.loc[data[col].str.lower().str.startswith("y",na=False),col] = 1

    data.loc[data[col].str.lower().str.startswith("n",na=False),col] = 0 

    data.loc[~data[col].isin([0,1]), col] = 0.5



# Work Interfere

col = "work_interfere"

data.loc[data[col].str.lower().str.contains("often",na=False) | data[col].str.lower().str.contains("sometime",na=False),col] = 1

data.loc[data[col].str.lower().str.contains("never",na=False) | data[col].str.lower().str.contains("rarely",na=False),col] = 0

data.loc[~data[col].isin([0,1]),col] = 0.5



# Leave

col = "leave"

data.loc[data[col].str.lower().str.contains("easy",na=False),col] = 0

data.loc[data[col].str.lower().str.contains("difficult",na=False),col] = 1

data.loc[~data[col].isin([0,1]),col] = 0.5



# Care Options

col = "care_options"

data.loc[data[col].str.lower().str.contains("yes",na=False),col] = 0

data.loc[data[col].str.lower().str.contains("not sure",na=False),col] = 0.5

data.loc[~data[col].isin([0,0.5]),col] = 1



# Other person related: coworkers, supervisor

cols = ["coworkers","supervisor"]

for col in cols:

    data.loc[data[col].str.lower().str.contains("yes",na=False),col] = 1

    data.loc[data[col].str.lower().str.contains("no",na=False),col] = 0

    data.loc[~data[col].isin([0,1]),col] = 0.5



# No of Employees, I will take the minimum of range as the value

col = "no_employees"

data.loc[data[col].str.startswith("100-",na=False), col] = 100

data.loc[data[col].str.startswith("26-",na=False), col] = 26

data.loc[data[col].str.startswith("6-",na=False), col] = 6

data.loc[data[col].str.startswith("500-",na=False), col] = 500

data.loc[data[col].str.startswith("1-",na=False), col] = 1

data.loc[data[col].str.startswith("M",na=False), col] = 1000

data.loc[~data[col].isin([1,6,26,100,500,1000]), col] = 100 # Somewhat median value

from nltk.sentiment.vader import SentimentIntensityAnalyzer as sid



# Define sentiment checker function

analyzer = sid()

def analyze(comment):

    global analyzer

    tcomment = str(comment)

    ss = analyzer.polarity_scores(tcomment)

    for k,v in sorted(ss.items(), key=lambda a : a[1], reverse=True):

        if k == "neu" or k == "compound" or v == 0.0:

            return 0.5

        if k == "neg":

            return 0

        if k == "pos":

            return 1

        

    return 0.5



col = "comments"

data[col] = data[col].apply(analyze)

data[col].fillna(0.5,inplace=True)
for c in data.columns.values:

    arr = data[c].unique()

    print("{c:20} : {a}".format(c=c,a=arr))