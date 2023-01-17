import pandas as pd

import numpy as np

import unicodedata
data = pd.read_csv("../input/jeopardy.csv", dtype = {"round": np.int16, "value": np.int16})

data.head()
data.daily_double.describe()
data.daily_double.loc[data.daily_double == "no"] = False

data.daily_double.loc[data.daily_double == "yes"] = True

data.head()
data["answer"] = data["answer"].str.upper()

data["question"] = data["question"].str.upper()

data.head()
#for example

data["answer"].iloc[11]
data["answer"] = data["answer"].str.replace("\\\\", "")

data["question"] = data["question"].str.replace("\\\\", "")

data["category"] = data["category"].str.replace("\\\\", "")

data["answer"].iloc[11]
#notice the question

data.iloc[16:17]
def strip_accents(text):

    try:

        text = unicode(text, 'utf-8')

    except NameError: # unicode is a default on python 3 

        pass

    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    return str(text)



data["category"] = data["category"].apply(strip_accents)

data["answer"] = data["answer"].apply(strip_accents)

data["question"] = data["question"].apply(strip_accents)



#and now

data.iloc[16:17]
#notice

data.iloc[129490:129497, ]
data["value"].iloc[:129493] = 2*data["value"].iloc[:129493]

data.iloc[129490:129497, ]
ACCEPTABLE_VALUES = [200, 400, 600, 800, 1000, 1200, 1600, 2000]



data.loc[(data["round"] != 3) & ((~data["value"].isin(ACCEPTABLE_VALUES))|data["daily_double"])]
data.loc[349562:349571, ]
indexes_with_problems = [index for index in range(0, len(data)) if data["round"].iloc[index] != 3 \

                                                                and ((data["value"].iloc[index] not in ACCEPTABLE_VALUES)

                                                                     or data["daily_double"].iloc[index])]

len(indexes_with_problems)

assert(len(indexes_with_problems) == 17025)
ROUND_ONE_VALUES = set([200, 400, 600,  800,  1000])

ROUND_TWO_VALUES = set([400, 800, 1200, 1600, 2000])



index = 0

while(index < 349630):        #the last problem is at 349625

    current_category = data["category"].iloc[index]

    indexes_in_category = [index]

    index += 1

    while(data["category"].iloc[index] == current_category):

        indexes_in_category.append(index)

        index += 1

    for i in indexes_in_category:

        if i in indexes_with_problems:

            indexes_in_category.remove(i)

            possible_values = ROUND_ONE_VALUES.copy() if data["round"].iloc[i] == 1 else ROUND_TWO_VALUES.copy()

            for j in indexes_in_category:

                if data["value"].iloc[j] in possible_values:

                    possible_values.remove(data["value"].iloc[j])

                else:

                    print("Problem removing value from line", j, end= "\n")

            data["value"].iloc[i] = max(possible_values)

            break
data.iloc[4257:4263]
data.iloc[77347:77352]
data["value"].iloc[77347] = 400
data.iloc[79592:79597]
data["value"].iloc[79592] = 400
data.tail(6)
index = 0

while(index < 349636):        #since we see above there is no problem with the last round

    while data["round"].iloc[index] == 3: #some Final Jeopardy questions appear successively without any round content

        index += 1

    current_category = data["category"].iloc[index]

    indexes_in_category = [index]

    index += 1

    while(data["category"].iloc[index] == current_category):

        indexes_in_category.append(index)

        index += 1

    for i in indexes_in_category:

        possible_values = ROUND_ONE_VALUES.copy() if data["round"].iloc[i] == 1 else ROUND_TWO_VALUES.copy()

        for j in indexes_in_category:

            if data["value"].iloc[j] in possible_values:

                possible_values.remove(data["value"].iloc[j])

            else:

                print("Problem removing value from line ", j, end= "\n")

                break
data.iloc[44064:44067]
data["value"].iloc[44065] = 400

data.iloc[44064:44067]
data.to_csv("jeopardy_clean.csv", index = False)
data = pd.read_csv("jeopardy_clean.csv")
del data["daily_double"] #1



data["air_date"] = pd.to_datetime(data["air_date"], format = "%m/%d/%Y") #so that ids are in order by date



data["category_id"] = data.groupby(["air_date", "category"]).ngroup() + 1 #2



data.sort_values(by=["category_id", "value"], ascending=[False, False], inplace = True) #3 and #6



del data["value"] #4

del data["air_date"] #5



bad_strings = ["HEARD HERE", "SEEN HERE", "SHOWN HERE", "PICTURED HERE"]

for string in bad_strings:

    data = data[~data["answer"].str.contains(string)] #7



normal_questions = data.loc[data["round"] != 3] #8

final_questions = data.loc[data["round"] == 3]



del normal_questions["round"] #9

del final_questions["round"]



normal_questions.reset_index(inplace = True, drop = True)

final_questions.reset_index(inplace = True, drop = True)



normal_questions.head(8)
final_questions.head(5)
normal_questions.to_csv("jeopardy_normal.csv", index = False)

final_questions.to_csv("jeopardy_final.csv", index = False)
normals = pd.read_csv("jeopardy_normal.csv")

finals = pd.read_csv("jeopardy_final.csv")
num_categories = len(normals["category_id"].unique())

num_finals = len(finals)

num_clues = len(normals)

print("There are {:,} clues across {:,} categories and {:,} Final Jeopardy questions"\

          .format(num_clues,        num_categories,     num_finals), end=".")
finals["agg"] = finals["category"] + "|" + finals["answer"] + "|" + finals["question"]

del finals["category"]

del finals["answer"]

del finals["question"]

finals.head()
finals.to_csv("jeopardy_final_agg.csv", index = False)
normals["agg"] = normals["category"] + "|" + normals["answer"] + "|" + normals["question"]

del normals["category"]

del normals["answer"]

del normals["question"]

normals.head()
normals["freq"] = normals.groupby("category_id")["category_id"].transform("count")

normals["category_id"].loc[normals["freq"] < 5] = 0

del normals["freq"]

normals.head()
size = len(normals)

for quarter in range(1, 5):

    chunk = normals.iloc[int((quarter-1)*size/4) : int(quarter*size/4), ]

    chunk.to_csv("jeopardy_normal_agg" + str(5-quarter) + ".csv", index = False)