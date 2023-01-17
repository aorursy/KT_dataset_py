import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Load pokemon data

pokemon = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

columns = pokemon.columns

rename_dict = {}



for col in columns:

    rep = col

    rep = rep.replace(" ", "_")

    rep = rep.replace(".", "")

    rename_dict[col] = rep



pokemon.rename(columns=rename_dict, inplace=True)

pokemon["mix_type"] = pokemon["Type_1"] + "-" + pokemon["Type_2"]
#Load combats data

combats = pd.read_csv("../input/pokemon-challenge/combats.csv")

combats["Winner"] = (combats["Winner"] == combats["First_pokemon"])
m1 = combats.merge(pokemon, left_on="First_pokemon", right_on="#", how="left")

m2 = m1.merge(pokemon, left_on="Second_pokemon", right_on="#", how="left", suffixes=("_f", "_s"))
#Check pokemon null value

length = pokemon.shape[0] * 1.0



print("Column\t null(%)")

for col in pokemon.columns:

    print("%s\t%6.3f" % (col, pokemon[col].isnull().sum() / length))
attribute = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]
#Attribute to Type_1 and Type_2 boxplot

for i in [1, 2]:

    tp = "Type_" + str(i)

    for col in attribute:

        plt.figure(figsize=(20, 5))

        ax = sns.boxplot(x=tp, y=col, data=pokemon)

        ax.set_title("Boxplot for %s - %s" % (col, tp))

        plt.show()
#Attribute to mix_type boxplot

for col in attribute:

    plt.figure(figsize=(20, 20))

    ax = sns.boxplot(y="mix_type", x=col, data=pokemon)

    ax.set_title("Boxplot for %s - (Type_1-Type_2)" %col)

    plt.show()
#Attribute to Legendary boxplot

for col in attribute:

    plt.figure(figsize=(20, 5))

    ax = sns.boxplot(x="Legendary", y=col, data=pokemon)

    ax.set_title("Boxplot for %s - Legendary" % (col))

    plt.show()
#Attribute to Generation boxplot

for col in attribute:

    plt.figure(figsize=(20, 5))

    ax = sns.boxplot(x="Generation", y=col, data=pokemon)

    ax.set_title("Boxplot for %s - Generation" % (col))

    plt.show()
#Distribution of Legendary pokemon in each Generation

plt.figure(figsize=(20, 5))

sns.countplot(x="Generation", hue="Legendary", data=pokemon)

plt.show()
#Type_1 winrate distribution

index = pokemon["Type_1"].unique()

df = pd.DataFrame(0, index=index, columns=["win_count", "fight_count", "win_rate"])



for idx, row in m2.iterrows():

    type1_f = row["Type_1_f"]

    type1_s = row["Type_1_s"]



    if(row["Winner"]):

        df.loc[type1_f, "win_count"] += 1

        df.loc[type1_f, "fight_count"] += 1

        df.loc[type1_s, "fight_count"] += 1

    else:

        df.loc[type1_s, "win_count"] += 1

        df.loc[type1_s, "fight_count"] += 1

        df.loc[type1_f, "fight_count"] += 1

df["win_rate"] = 100.0 * df["win_count"] / df["fight_count"]



plt.figure(figsize=(20, 5))

ax = sns.barplot(x=df.index, y=df.win_rate, order=df.sort_values(by=["win_rate"]).index)

ax.set_title("Winrate of each category in Type_1")

plt.show()
#Type_2 winrate distribution

index = pokemon["Type_2"].unique()

df = pd.DataFrame(0, index=index, columns=["win_count", "fight_count", "win_rate"])



for idx, row in m2.iterrows():

    type2_f = row["Type_2_f"]

    type2_s = row["Type_2_s"]



    if(row["Winner"]):

        df.loc[type2_f, "win_count"] += 1

        df.loc[type2_f, "fight_count"] += 1

        df.loc[type2_s, "fight_count"] += 1

    else:

        df.loc[type2_s, "win_count"] += 1

        df.loc[type2_s, "fight_count"] += 1

        df.loc[type2_f, "fight_count"] += 1

df["win_rate"] = 100.0 * df["win_count"] / df["fight_count"]



plt.figure(figsize=(20, 5))

ax = sns.barplot(x=df.index, y=df.win_rate, order=df.sort_values(by=["win_rate"]).index)

ax.set_title("Winrate of each category in Type_2")

plt.show()
#mix_type winrate distribution

index = pokemon["mix_type"].unique()

df = pd.DataFrame(0, index=index, columns=["win_count", "fight_count", "win_rate"])



for idx, row in m2.iterrows():

    mix_type_f = row["mix_type_f"]

    mix_type_s = row["mix_type_s"]



    if(row["Winner"]):

        df.loc[mix_type_f, "win_count"] += 1

        df.loc[mix_type_f, "fight_count"] += 1

        df.loc[mix_type_s, "fight_count"] += 1

    else:

        df.loc[mix_type_s, "win_count"] += 1

        df.loc[mix_type_s, "fight_count"] += 1

        df.loc[mix_type_f, "fight_count"] += 1

df["win_rate"] = 100.0 * df["win_count"] / df["fight_count"]



plt.figure(figsize=(20, 20))

ax = sns.barplot(x=df.win_rate, y=df.index, order=df.sort_values(by=["win_rate"]).index)

ax.set_title("Winrate of each category in mix_type")

plt.show()
#Generation winrate distribution

index = pokemon["Generation"].unique()

df = pd.DataFrame(0, index=index, columns=["win_count", "fight_count", "win_rate"])



for idx, row in m2.iterrows():

    generation_f = row["Generation_f"]

    generation_s = row["Generation_s"]



    if(row["Winner"]):

        df.loc[generation_f, "win_count"] += 1

        df.loc[generation_f, "fight_count"] += 1

        df.loc[generation_s, "fight_count"] += 1

    else:

        df.loc[generation_s, "win_count"] += 1

        df.loc[generation_s, "fight_count"] += 1

        df.loc[generation_f, "fight_count"] += 1

df["win_rate"] = 100.0 * df["win_count"] / df["fight_count"]



ax = sns.barplot(x=df.index, y=df.win_rate, order=df.sort_values(by=["win_rate"]).index)

ax.set_title("Winrate of each category in Generation")

plt.show()
#Attribute - Winner scatterplot

for col in attribute:

    x = col + "_f"

    y = col + "_s"



    plt.figure(figsize=(10, 10))

    ax = sns.scatterplot(x=x, y=y, hue="Winner", data=m2)

    ax.set_title("%s - Winner scatterplot" % col)

    plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
def target_encoding(combats, col):

    replace_dict = {}



    res = combats.groupby([col])["Winner"].sum() / combats.groupby([col]).size()

    for idx, val in res.items():

        replace_dict[idx] = val



    combats[col].replace(replace_dict, inplace=True)



def label_encoding(pokemon, col):

    le = LabelEncoder()

    pokemon[col] = le.fit_transform(pokemon[col])

    

def run_cv_lgb(x, y):

    params = {

            "objective": "binary",

            "num_leaves": 31,

            "bagging_freq": 1,

            "bagging_fraction": 0.1,

            "feature_fraction": 0.1

            }

    train_data = lgb.Dataset(x, label=y)

    eval_hist = lgb.cv(params=params, train_set=train_data, metrics="auc")

    print("Lightgbm best accuracy: %f" % max(eval_hist["auc-mean"]))



def run_cv_lr(x, y):

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits)

    mean = 0.0



    for train_idx, test_idx in skf.split(x, y):

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



        lr = LogisticRegression(solver="lbfgs", max_iter=1000)

        lr.fit(x_train, y_train)



        score = lr.score(x_test, y_test)

        mean += score

        print(score)



    print("Logistic Regression mean accuracy: %f" % (mean / n_splits))



def run_cv_dtc(x, y):

    n_splits = 5

    skf = StratifiedKFold(n_splits=n_splits)

    mean = 0.0



    for train_idx, test_idx in skf.split(x, y):

        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]

        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



        dtc = DecisionTreeClassifier()

        dtc.fit(x_train, y_train)



        score = dtc.score(x_test, y_test)

        mean += score

        print(score)



    print("Decision Tree mean accuracy: %f" % (mean / n_splits))
pokemon.drop(columns=["Name"], axis=1, inplace=True)

pokemon.fillna("nan", inplace=True)



label_encoding(pokemon, "Type_1")

label_encoding(pokemon, "Type_2")

label_encoding(pokemon, "mix_type")



m1 = combats.merge(pokemon, left_on="First_pokemon", right_on="#", how="left")

m2 = m1.merge(pokemon, left_on="Second_pokemon", right_on="#", how="left", suffixes=("_f", "_s"))

m2["Speed_diff"] = m2["Speed_f"] - m2["Speed_s"]



y = m2["Winner"]

m2.drop(columns=["Winner","#_f", "#_s"], inplace=True, axis=1)
#Predict with the whole data

run_cv_lr(m2, y)

run_cv_dtc(m2, y)

run_cv_lgb(m2, y)
#Predict with only "Speed_diff" column

run_cv_lr(m2[["Speed_diff"]], y)

run_cv_dtc(m2[["Speed_diff"]], y)

run_cv_lgb(m2[["Speed_diff"]], y)