import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

import plotly.express as px

import re

from functools import reduce



from nltk.stem import PorterStemmer

from nltk.corpus import stopwords



from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.linear_model import ElasticNet

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score



sns.set()
# The original data

df = pd.read_csv("../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv")
df.head().T
df.duplicated().sum()
df.drop_duplicates(inplace=True)

raw_df = df.copy() # Copy of the original Data after dropping the duplicates
fig = px.bar(df.isna().sum().sort_values().reset_index(), x="index", y=0, title="Evaluating Missing values")

fig.update_layout({"yaxis": {"title": "Missing values"}})

fig.show()
df["merchant_profile_picture"].unique()
print(df.loc[df["merchant_profile_picture"].isna(), "merchant_has_profile_picture"].unique())

print(df.loc[~df["merchant_profile_picture"].isna(), "merchant_has_profile_picture"].unique())



df.drop(["merchant_profile_picture"], axis=1, inplace=True)
df["has_urgency_banner"].value_counts()
df["urgency_text"].value_counts()
print(df.loc[df["has_urgency_banner"].isna(), "urgency_text"].unique())

print(df.loc[~df["has_urgency_banner"].isna(), "urgency_text"].unique())



df.drop(["urgency_text"], axis=1, inplace=True)

df["has_urgency_banner"] = df["has_urgency_banner"].fillna(0)
rating_cols = ["rating_one_count", "rating_two_count", "rating_three_count", "rating_four_count", "rating_five_count", "rating", "rating_count"]



no_vote_df = df.loc[df[rating_cols].isna().any(axis=1), rating_cols]

df.loc[no_vote_df.index, rating_cols] = 0



df.loc[no_vote_df.index, rating_cols].head()
df["origin_country"].value_counts()
df["countries_shipped_to"].value_counts()
df.drop("origin_country", axis=1, inplace=True)
df["product_color"].value_counts()
# The urls of each product

df.loc[df["product_color"].isna(), ["product_url"]].head()
missing_color_vectors = [

    "#F6F7F1",

    "#09041E",

    "#D24D41",

    "#6EACE1",

    "#323232",

    "#CECECE",

    "#E1E1EB",

    "#1D0E25",

    "#9EA59D",

    "#272727",

    "#FFD1DE",

    "#B37264",

    "#050505",

    "#120F1A",

    "#F9F9F9",

    "#C6363F",

    "#B7EDEC",

    "#4DA4C8",

    "#E7E6E4",

    "#C7955E",

    "#D3560A",

        np.nan,

    "#68166B",

    "#97E8D7",

    "#A6C2D1",

        np.nan,

    "#374757",

    "#E94875",

    "#EC2A13",

    "#4C4C4C",

    "#B9B4A9",

    "#0E0809",

    "#EBEEF4",

    "#DEEEEF",

    "#7F7181",

    "#E1E1E1",

    "#CFE5D8",

    "#E0D5D4",

    "#F4F7FB",

    "#74636D",

    "#E1E6EE"

]

assert len(missing_color_vectors) == df.loc[df["product_color"].isna(),:].shape[0], f"There are missing colors, {len(missing_color_vectors)}"

df.loc[df["product_color"].isna(), ["product_color"]] = missing_color_vectors

df.loc[df["product_color"] == "multicolor", ["product_color"]] = np.nan
custom_color_rainbow = {

    "coffee": "#381E07",

    "floral": "#E4A593",

    "rose": "pink",

    "leopard": "#F2D24A",

    "leopardprint": "#F2D24A",

    "camouflage": "green",

    "army": "green",

    "camel": "#C19A6B",

    "wine": "#940F22",

    "apricot": "#F3C8AB",

    "burgundy": "#7D2F3D",

    "jasper": "#D0393C",

    "claret": "#940F22",

    "rainbow": "white",

    "star": "yellow",

    "nude": "pink"

}



def get_rgb(color):

    """ 

    Returns the rgb vector if the color exists on matplotlib. 

    This function is a bit messy with the nested try except statements, but performance is not critical 

    and it works for now, but I should clean it later """

    

    # TODO: THere is definitely a more elegant implementation

    if color in custom_color_rainbow.keys():

        return matplotlib.colors.to_rgb(custom_color_rainbow[color])



    try:

        return matplotlib.colors.to_rgb(color)

    except:

        base_colors = ["blue", "green", "red", "white", "black", "gold", "yellow", "pink", "purple", "orange", "grey", "khaki"]

        simplified_color = [c for c in base_colors if c in color]

        try:

            return matplotlib.colors.to_rgb(simplified_color[0])

        except:

            return np.nan

        

rgb_colors = df.loc[~df["product_color"].isna(), "product_color"].apply(get_rgb)
rgb_colors_dict = [{"r": r[0], "g": r[1], "b": r[2]} for r in rgb_colors]

rgb_colors_dict_df = pd.DataFrame(rgb_colors_dict, index=rgb_colors.index)

df["r"] = rgb_colors_dict_df["r"]

df["g"] = rgb_colors_dict_df["g"]

df["b"] = rgb_colors_dict_df["b"]



# Fill the missing values with the mean of the column

df["r"].fillna(df["r"].mean(), inplace=True)

df["g"].fillna(df["g"].mean(), inplace=True)

df["b"].fillna(df["b"].mean(), inplace=True)



# Drop the useless columns

df.drop(["product_color", "product_url", "product_picture"], axis=1, inplace=True)
df["product_variation_size_id"].value_counts()
def clean_sizes(s: str) -> str:

    return re.findall(r"M|X?[SsLl](?!\w+)", s)



def convert_us_to_eu(s: str) -> str:

    number = re.findall("\d+", s[0])[0]

    

    eu_to_letter = {

        (0, 36): "XS",

        (36, 40): "S",

        (40, 44): "M",

        (44, 48): "L",

        (48, 52): "XL",

        (52, 60): "XXL"

    }

    return [v for k, v in eu_to_letter.items() if k[0]<int(number)<k[1]][0]

     

original_sizes = df["product_variation_size_id"].dropna().unique()

changed_to_letter = [re.sub(r"EU\s*\d+", convert_us_to_eu, s) for s in original_sizes]

filtered_sizes = [clean_sizes(s) for s in changed_to_letter]
original_sizes = df["product_variation_size_id"].dropna()

changed_to_letter = [re.sub(r"EU\s*\d+", convert_us_to_eu, s) for s in original_sizes]

filtered_sizes = [clean_sizes(s) for s in changed_to_letter]
df.loc[original_sizes.index, "product_size"] = [c[0].lower() if c != [] else np.nan for c in filtered_sizes ]

df["product_size"].fillna("M", inplace=True)

df["product_size"].value_counts()
df["product_size"] = OrdinalEncoder().fit_transform(df["product_size"].values.reshape(-1, 1))

df["product_size"].value_counts()
df.drop("product_variation_size_id", axis=1, inplace=True)
df["merchant_id"].value_counts()
df["merchant_title"].value_counts()
# Surprisingly there are more unique values here than unique merchants

df["merchant_info_subtitle"].value_counts()
df[["merchant_info_subtitle", "merchant_rating_count", "merchant_rating"]]
df.describe()
df.drop(["merchant_id", "merchant_title", "merchant_name", "merchant_info_subtitle"], axis=1, inplace=True)
df["currency_buyer"].value_counts()
df.drop("currency_buyer", inplace=True, axis=1)
df["n_tags"] = df["tags"].apply(lambda x: len(x.split(",")))
swords = stopwords.words('english')



def clean_text(s: str) -> str:

    """ Cleans the strings from the titles and the tags"""

    

    # Only Keep letters

    processed_s = re.sub(r"[^a-z]", " ", s.lower())

    

    ps = PorterStemmer()

    

    # stemmed words with Porter Lemantizer

    stemmed_s = [ps.stem(s) for s in processed_s.split()]

    

    unique_tags = list(set(stemmed_s))

    

    # Filter stop words

    cleaned_text = [w for w in unique_tags if (w not in swords and len(w) > 2)]

    

    return cleaned_text



all_tags = (df["tags"] + df["title"] + df["title_orig"]).values

processed_tags = [clean_text(s) for s in all_tags]
len(set(reduce(lambda a,b : a+b, processed_tags)))
# Each of these will be a binary column

tags_list = [

    r"\bmen",

    r"\bwomen",

    "shirt",

    "robe",

    "dress",

    "skirt",

    "underwear",

    "swim",

    "nightwear",

    "sleepwear",

    "shorts"

]



def build_tags_dict(tags_list_per_product: list) -> dict:

    """ Returns a dict with 0 or 1, any of the tags_list were found  on the tags per sample"""

    return {tag.lstrip('\\b'): any(re.findall(tag, " ".join(tags_list_per_product))) for tag in tags_list}


testing_set = processed_tags[3]

print(build_tags_dict(testing_set))

print()

print(testing_set)
cols_df = pd.DataFrame([build_tags_dict(t) for t in processed_tags])

cols_df.head()
df = df.merge(cols_df, left_index=True, right_index=True)
test_index = np.random.randint(df.shape[0])

df.loc[test_index, cols_df.columns.values.tolist() + ["title", "tags", "title_orig"]].to_dict()
(df["title"].str.lower() == df["title_orig"].str.lower() ).sum()
df.drop(["tags", "title", "title_orig"], axis=1, inplace=True)
df["theme"].value_counts()
df["crawl_month"].value_counts()
df.drop(["theme", "crawl_month", "shipping_option_name"], axis=1, inplace=True)
df["product_id"].value_counts().value_counts()
n_id_counts = df["product_id"].value_counts()

duplicated_ids = n_id_counts[n_id_counts > 1].index

comp_ids = df[df["product_id"].isin(duplicated_ids)].sort_values("product_id").set_index("product_id")

comp_ids.T
raw_df.loc[raw_df["product_id"].isin(duplicated_ids), ["merchant_id", "product_id"]].groupby("merchant_id").count().squeeze().min()
df = df.sort_values("has_urgency_banner", ascending=False).drop_duplicates("product_id")

df.drop("product_id", inplace=True, axis=1)
df.describe().T


to_bool_cols = ["uses_ad_boosts", "shipping_is_express", "badge_local_product", "badge_product_quality", "has_urgency_banner", "merchant_has_profile_picture"]

df[to_bool_cols] = df[to_bool_cols].astype(bool)
df.dtypes
assert not df.isna().any().any()
df["inventory_total"].min()
# Just some usefull variables

cont_cols = df.select_dtypes(exclude="bool").columns

bool_cols = df.select_dtypes("bool").columns
px.imshow(df[cont_cols].corr(), width=1000, height=1000)
df.drop(["rating_five_count", "rating_four_count", "rating_three_count", "rating_two_count", "rating_one_count"], axis=1, inplace=True)
scatter_matrix_cols = ["price", "units_sold", "rating", "merchant_rating", "rating_count"]

px.scatter_matrix(df[scatter_matrix_cols], width=1000, height=1000)
px.box(df[bool_cols.values.tolist() + ["units_sold"]].melt(id_vars="units_sold"), x="variable", y="units_sold", color="value", title="Sold Unites Based on the Boolean Columns")
X = df.drop("units_sold", axis=1)

y = df["units_sold"]

reg = make_pipeline(StandardScaler(), ElasticNet(alpha=0.5))

reg.fit(X, y)

pd.Series(reg[-1].coef_, index=X.columns).sort_values().plot.bar(figsize=(20, 5))



r2 = r2_score(y, reg.predict(X))

plt.title(f"R2: {round(r2, 2)}")

plt.show()
no_counts_df = df.drop(["rating_count", "merchant_rating_count"], axis=1)
X = no_counts_df.drop("units_sold", axis=1)

y = no_counts_df["units_sold"]

reg = make_pipeline(StandardScaler(), ElasticNet(alpha=0.5))

reg.fit(X, y)

pd.Series(reg[-1].coef_, index=X.columns).sort_values().plot.bar(figsize=(20, 5))



r2 = r2_score(y, reg.predict(X))

plt.title(f"R2: {round(r2, 2)}")

plt.show()
df["profit"] = df["retail_price"] - df["price"]

df["profit"].hist(bins=40, figsize=(20, 5))

plt.title("Difference between retail_price and price")

plt.show()
X = df["rating_count"].to_frame()

y = df["units_sold"]





X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

reg = make_pipeline(StandardScaler(), ElasticNet(0.5))

reg.fit(X_train, y_train)

predition = reg.predict(X_test)

test_r2 = r2_score(y_test, predition)

print(f"Test r2_score: {round(test_r2, 2)}")
# Make sold units prediction

def get_results(reg, df: pd.DataFrame) -> pd.DataFrame:

    """

    reg is the pre trained pipelin (In this case with the included scaler)

    df is the original Data Frame

    """

    df["predicted_sold_units"] = reg.predict(df["rating_count"].to_frame())



    df["profit"] = (df["retail_price"] - df["price"]) * df["predicted_sold_units"]



    df["units_to_order"] = np.ceil(np.min(df["predicted_sold_units"] - df["inventory_total"], 0))

    return df



results_df = get_results(reg, raw_df)
profitable_products = results_df[["profit", "product_id"]].set_index("product_id").squeeze().sort_values(ascending=False)

profitable_products.head(10)
results_df["profit"].hist(bins=100, figsize=(20, 5))
results_df.loc[results_df["product_id"].isin(profitable_products.head().index), ["units_to_order", "product_id"]]