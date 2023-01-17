def get_dtype_lists(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()))

        else:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data,features):

    dtypes = get_dtype_lists(data,features)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
from pandas import read_csv

data = read_csv("../input/survey.csv")
data.head()
data.drop("Timestamp",1, inplace=True)
show_all_uniques(data, data.columns)
textual_features = ["comments"]

data.drop("comments",1,inplace=True)
dtype = get_dtype_lists(data, data.columns)
numerics = dtype["int64"]
categories = dtype["object"]
for category in categories: data[category] = data[category].apply(str)
data[categories].head()
from seaborn import countplot, color_palette, set_style, despine

from pandas import DataFrame

from matplotlib.pyplot import show

from IPython.display import display



set_style("whitegrid")

set_style({"axes.grid":False})



for c in categories:

    

    if data[c].unique().size > 10:

        ax = countplot(data=data, y=c, palette=color_palette("colorblind"))

        

        last_tick = int(round(ax.get_xticks()[-1]/len(data),1) * 10) + 1

        ax.set_xticks([i * (len(data) * 0.1) for i in range(0,last_tick)])

        ax.set_xticklabels(["{:.0f}%".format((tick / len(data)) * 100) for tick in ax.get_xticks()])

        

        despine(left=True)

        show()

        display(DataFrame(data[c].value_counts()).T)

        continue

        

    ax = countplot(data[c], palette=color_palette("colorblind"))

    

    last_tick = int(round(ax.get_yticks()[-1]/len(data),1) * 10) + 1

    ax.set_yticks([i * (len(data) * 0.1) for i in range(0,last_tick)])

    ax.set_yticklabels(["{:.0f}%".format((tick / len(data)) * 100) for tick in ax.get_yticks()])

    

    maximum_yticklabel_length = max([len(str(x)) for x in data[c].unique()])

    

    if maximum_yticklabel_length in range (5,7):

        ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

    elif maximum_yticklabel_length > 6:

        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

        

    despine(left=True)

    

    show()

    

    display(DataFrame(data[c].value_counts()).T)
from pandas import cut
countplot(data.Age, palette=color_palette("colorblind"))

show()
data.Age.unique()
data["Age"] = data.Age.apply(lambda x: -1 if x < 18 or x > 72 else x)
data["Age"].unique()
countplot(data.Age, palette=color_palette("colorblind"))

show()
countplot(cut(data[data.Age > 0]["Age"],[i * 5 for i in range(3,16)], right=False), palette=color_palette("colorblind"))

show()
genders = data["Gender"]

genders.unique()
from nltk import edit_distance
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

genders.unique()
countplot(genders)

show()