import numpy as np

import pandas as pd

import seaborn as sns

import warnings; warnings.simplefilter('ignore')



%matplotlib inline



data = pd.read_csv("../input/Reveal_EEO1_for_2016.csv")
data.head()
data.info()
%%capture

data["count"] = data["count"].convert_objects(convert_numeric=True)
companyGender = pd.DataFrame({"count": data.groupby(["company", "gender"])["count"].sum()}).reset_index()
genderData = companyGender[companyGender["gender"] == "male"]

genderData.drop(["gender"], inplace=True, axis=1)

genderData.columns.values[1] = "male_count"



femaleData = companyGender[companyGender["gender"] == "female"]

femaleData.columns = ["company", "gender", "female_count"]



genderData = genderData.assign(female_count = femaleData["female_count"].values)



male_prop = genderData["male_count"] / (genderData["male_count"] + genderData["female_count"])

genderData = genderData.assign(male_proportion = male_prop)



genderData = genderData.assign(female_proportion = (1 - male_prop))



genderScore = 1 - (genderData["male_proportion"]**2 + genderData["female_proportion"]**2) 

genderData = genderData.assign(gender_score = genderScore)

genderData.sort_values(["gender_score"], ascending=False)
np.unique(data["race"])
companyRace = pd.DataFrame({"count": data.groupby(["company", "race"])["count"].sum()}).reset_index()
asianData = companyRace[companyRace["race"] == "Asian"]

asianData.drop(["race"], inplace=True, axis=1)

asianData.columns.values[1] = "asian_count"



# femaleData = companyGender[companyGender["gender"] == "female"]

# femaleData.columns = ["company", "gender", "female_count"]



# genderData = genderData.assign(female_count = femaleData["female_count"].values)



# male_prop = genderData["male_count"] / (genderData["male_count"] + genderData["female_count"])

# genderData = genderData.assign(male_proportion = male_prop)



# genderData = genderData.assign(female_proportion = (1 - male_prop))



# genderScore = 1 - (genderData["male_proportion"]**2 + genderData["female_proportion"]**2) 

# genderData = genderData.assign(gender_score = genderScore)

# genderData.sort_values(["gender_score"], ascending=False)