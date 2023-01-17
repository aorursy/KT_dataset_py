import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import re
choco = pd.read_csv("../input/chocolate-bar-ratings/flavors_of_cacao.csv")
choco.head()
# How many reviews we have in the dataset
choco.shape
# Get general info abt the dataset
choco.describe(include="all")
# What is datatype?
choco.dtypes
# Change the column name:
column_name = choco.columns
new_name = ["company", "species", "REF", "review_year", "cocoa_p", "company_location", "rating", "bean_type", "country"]
choco = choco.rename(columns=dict(zip(column_name, new_name)))
choco.head()
choco["cocoa_p"] = choco["cocoa_p"].apply(lambda n: n.replace("%","")).astype(float)/100
choco.head()
choco.dtypes
# What kind of species in Species column
choco["species"].value_counts()
# Is there any NAN value in "species"?
choco["species"].isna().value_counts()
# Is there any NAN value in origin country?
choco["country"].isna().value_counts()
# Replace origin country
choco["country"] = choco["country"].fillna(choco["species"])
choco["country"].isna().value_counts()
# Let's look at most frequent origin countries
choco["country"].value_counts()
# We see a lot of countries have " " value - it means that this is 100% of blend.
choco[choco["country"].str.len()==1]["species"].unique()
# Species contain "," are also blend
choco[choco["species"].str.contains(",")]["species"].nunique()
# Is there any misspelling in country?
choco["country"].sort_values().unique()
# Text preparation (correction) function
def text_prep(text):
    replacement = [
        ["-", ","], ["/", ", "], ["/ ", ", "], ["\(", ", "], [" and", ", "], [" &", ", "], ["\)", ", "],
        ["C Am", "Central America"],
        ["S America", "South America"],
        ["Ven,|Ven$|Venez,|Venez$", "Venezuela, "],
        ["DR|Dom Rep|Domincan Republic|Dominican Rep,|Domin Rep", "Dominican Republic"],
        ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
        ["Mad,|Mad$", "Madagasca, "],
        ["PNG", "Papua New Guinea"],
        ["Gre,|Gre$", "Grenada, "],
        ["Haw,|Haw$", "Hawaii, "],
        ["Guat,|Guat$", "Guatamala, "],
        ["Nic,|Nic$", "Nicaragua, "],
        ["Cost Rica", "Costa Rica"],
        ["Mex,|Mex$", "Mexico, "],
        ["Jam,|Jam$", "Jamaica, "],
        ["Tri,|Tri$", "Trinidad, "],
        [" Bali", " ,Bali"],
        [",  ", ", "],
        [", $", ""], [",  ", ", "], [", ,", ", "], ["\xa0"," "],  [",\s+", ","], 
    ]
    for i, j in replacement:
        text = re.sub(i, j, text)
    return text
choco["country"].str.replace(".","").apply(text_prep).unique()
choco["country"] = choco["country"].str.replace(".", "").apply(text_prep)
## Check it gain
choco["country"].value_counts().tail(10)
## How many countries may contain in Blend?
choco["country"].str.count(",").value_counts()
# Is there any misspelling in company location?
choco["company_location"].sort_values().unique()
choco["company_location"] = choco["company_location"]\
.str.replace("Amsterdam", "Netherlands")\
.str.replace("U.K.", "England")\
.str.replace("Niacragua","Nicaragua")\
.str.replace("Domincan Republic", "Dominican Republic")

choco["company_location"].sort_values().unique()
choco["is_blend"] = choco["species"].str.lower().str.contains(',|(blend)|;')
choco["len"] = choco["country"].str.len() == 1
choco["cblend"] = choco["country"].str.lower().str.contains(",")
choco["final"] = choco["is_blend"] | choco["len"] | choco["cblend"]
choco["final"].value_counts().unique()
choco.tail()
choco["isblend"] = choco["final"].apply(lambda y: 0 if y == False else 1)
## Number of blend chocolate in dataset
choco["isblend"].value_counts()
## How many chocolate is from domestic
choco["isdomestic"] = np.where(choco["company_location"] == choco["country"], 1, 0)
choco["isdomestic"].value_counts()
# DISTRIBUTION OF COCOA %
fig = plt.figure(figsize=(14,6))
sns.distplot(choco["cocoa_p"], hist=True)
plt.legend("Distribution of Cocoa%")
# DISTRIBUTION OF COCOA PERCENTAGE
fig, ax = plt.subplots(figsize=[16,4])
for i, c in choco.groupby("isdomestic"):
    sns.distplot(c["cocoa_p"], ax=ax, label=["Not Domestic", "Domestic"][i])
ax.set_title("Cocoa %, Distribution")
ax.legend()
plt.show()
# DISTRIBUTION OF RATING
fig = plt.figure(figsize=(14,6))
sns.distplot(choco["rating"], label="Rating")
# DISTRIBUTION OF RATING
fig = plt.figure(figsize=(14,6))
for i, c in choco.groupby("isdomestic"):
    sns.distplot(c["rating"], label=["Not Domestic", "Domestic"][i])
plt.legend()
# WHICH ONE IS BETTER? DOMESTIC OR NOT DOMESTIC?
figure = plt.figure(figsize=(8,6))
sns.boxplot(data=choco, x="isdomestic", y="rating")
# Which is better? Pure or blend?
figure = plt.figure(figsize=(8,6))
sns.boxplot(data=choco, x="isblend", y="rating")
choco.loc[choco["country"].str.contains(",")]

def choco_tidy(choco):
    data = []
    for i in choco.itertuples():
        for c in i.country.split(","):
            data.append({
                "company" : i.company,
                "species" : i.species,
                "REF" : i.REF,
                "review_year" : i.review_year,
                "cocoa_p" : i.cocoa_p,
                "company_location": i.company_location,
                "rating" : i.rating,
                "bean_type" : i.bean_type,
                "country" : c,
                "is_blend" : i.isblend,
                "is_domestic" : i.isdomestic
            })
    return pd.DataFrame(data)

choco_ = choco_tidy(choco)
print(choco.shape, choco_.shape)
choco_comp = choco_["company_location"].value_counts().sort_values(ascending=False)
figure = plt.figure(figsize=(10,20))
sns.countplot(y="company_location", data=choco_, orient="v")
choco_.groupby("is_domestic")["country"].value_counts()
figure = plt.figure(figsize=(10,20))
sns.countplot(y="country", data=choco_, hue="is_domestic")
blends = pd.crosstab(choco_["company_location"], choco_["is_blend"], choco_["rating"], aggfunc="mean")
blends['tot'] = blends.max(axis=1)
blends = blends.sort_values("tot", ascending=False)
blends = blends.drop("tot", axis=1)

fig, ax = plt.subplots(figsize=[15,10])
sns.heatmap(blends.head(20), cmap="RdBu_r", linewidths=.5)
review_year = pd.crosstab(choco_["company_location"], choco_["review_year"], choco_["rating"], aggfunc="mean")
review_year["tot"] = review_year.sum(axis=1)
review_year = review_year.sort_values("tot", ascending=False)
review_year = review_year.drop("tot", axis=1)

figure = plt.figure(figsize=(10,8))
sns.heatmap(review_year.head(20), cmap="RdBu_r", linewidths=.5)
figure = plt.figure(figsize=(10,6))
sns.scatterplot(x="cocoa_p", y="rating", hue="is_domestic", data=choco_)
