def get_dtype_lists(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output
from pandas import read_csv

data = read_csv("../input/survey.csv")
data.drop("Timestamp",1, inplace=True)
dtype = get_dtype_lists(data, data.columns)
numerics = dtype["int64"]
categories = dtype["object"]

categories.remove("comments")
for category in categories: data[category] = data[category].apply(str)
from pandas import cut
data["Age"] = data.Age.apply(lambda x: -1 if x < 18 or x > 72 else x)

data["Age"] = cut(data["Age"],[-1] + [i * 5 for i in range(3,16)], right=False).astype(str)
genders = data["Gender"]

genders.unique()
from nltk.metrics.distance import edit_distance



genders = data["Gender"]

genders = genders.apply(str.lower).apply(str.strip, " ")

genders = genders.apply(lambda x: "male" if edit_distance("male",x) < 2 else x)

genders = genders.apply(lambda x: "female" if edit_distance("female",x) < 2 else x)

genders = genders.apply(lambda x: x.replace("(cis)","").replace("cis", ""))

genders = genders.apply(lambda x: "male" if x.strip(" ") in ["m", "man", "mail",] else x)

genders = genders.apply(lambda x: "female" if x in ["f", "femail"] or "female" in x else x)

genders = genders.apply(lambda x: "trans female" if ("female" in x or "woman" in x) and "trans" in x else x)

genders = genders.apply(lambda x: "male" if ("male" in x or "guy" in x) and "female" not in x else x)

genders = genders.apply(lambda x: "genderqueer" if "female" not in x and "male" not in x else x)



data["Gender"] = genders
data.head()
def get_count(data, tuples):

    data = data.copy()

    data["Count"] = data.index

    ordered_categories = [category for (category, value) in tuples]

    counts = data.groupby(ordered_categories).count()["Count"]

    count = counts

    for tup in tuples: count = count[tup[1]]

    return count



def get_support(data, tuples):

    support = get_count(data,tuples)

    return support/len(data)



def get_confidence(data, predictors, outcomes):

    numerator = get_support(data,predictors+outcomes)

    denominator = get_support(data, predictors)

    return numerator / denominator



def get_lift(data,tuples):

    numerator = get_support(data,tuples)

    denominator = 1 

    for tup in tuples: denominator *= get_support(data,[tup])

    return numerator /denominator
def get_category_value_tuples(data, categories):

    tuples = []

    for category in categories:

        for value in data[category].unique():

            tuples += [(category,value)]

    return tuples
minimum_support = 30 / len(data)
outcome_categories = ["work_interfere", "treatment"]

predictor_categories = [category for category in categories if category not in outcome_categories] + ["Age"]
outcomes = get_category_value_tuples(data, outcome_categories)



rows = []



from itertools import combinations, product



predictor_tuples = {}

for category in predictor_categories: 

    predictor_tuples[category] = [unit for unit in get_category_value_tuples(data, [category]) if get_support(data,[unit]) > minimum_support]





length = 2



for predictor_combination in combinations(predictor_categories,length):



    for relationship in product(*[predictor_tuples[category] for category in list(predictor_combination)], outcomes):



        predictor = list(relationship[0:length])

        outcome = relationship[length]

        

        try:    

            

            support = get_support(data, list(relationship))

        

            if support < minimum_support: continue

            

            row = {}



            for i in range(0,length): row["Predictor_%s" % i]  = "/".join(predictor[i]) 



            row["Outcome"] = "/".join(list(outcome)) 

            row["Support"] = support

            row["Confidence"] = get_confidence(data, predictor, [outcome])

            row["Lift"] = get_lift(data, predictor + [outcome])

            

            """

            text = ""

            for i in range(0,len(predictor)): text += row["Predictor_%s" % i] + " "

            text = "=> " + row["Outcome"]

            print(text)

            """



            rows += [row]

            

        except:

            pass
from pandas import DataFrame

table = DataFrame(rows, columns=["Predictor_" + str(i) for i in range(0,length)] + ["Outcome","Support","Confidence","Lift"])
from pandas import set_option



set_option('display.max_rows', 500)
where = (table.Outcome == "treatment/Yes") & (table.Confidence > 0.6) & (table.Lift > 1.1)

table[where].round(2).sort_values("Confidence", ascending=False)
table[(table.Outcome == "treatment/No")].round(2).sort_values("Confidence", ascending=False)
table[table.Outcome == "work_interfere/Often"].round(2).sort_values("Confidence", ascending=False)
table[table.Outcome == "work_interfere/Sometimes"].round(2).sort_values("Confidence", ascending=False)
table[table.Outcome == "work_interfere/Rarely"].round(2).sort_values("Confidence", ascending=False)
table[table.Outcome == "work_interfere/Never"].round(2).sort_values("Confidence", ascending=False)