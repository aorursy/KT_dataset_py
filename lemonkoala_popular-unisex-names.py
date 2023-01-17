import numpy  as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from bq_helper import BigQueryHelper
from wordcloud import WordCloud, STOPWORDS

matplotlib.rcParams["figure.figsize"] = (20, 7)
db = BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="usa_names"
)
query_string = """
SELECT
    name,
    year,
    gender,
    sum(number) AS count
FROM
    `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY
    name,
    year,
    gender
"""

counts = db.query_to_pandas_safe(query_string)
counts.head()
counts_by_gender = pd.pivot_table(
    counts,
    index=["name", "year"],
    columns=["gender"],
    values="count",
    aggfunc=np.sum,
    fill_value=0
)
counts_by_gender["total"] = counts_by_gender["F"] + counts_by_gender["M"]
counts_by_gender.head()
proportions = pd.DataFrame(index=counts_by_gender.index)

proportions["Total"]   = counts_by_gender["total"]
proportions["F"] = counts_by_gender["F"] / counts_by_gender["total"]
proportions["M"] = counts_by_gender["M"] / counts_by_gender["total"]
proportions["AbsDiff"] = (proportions["F"] - proportions["M"]).abs()

proportions.head()
popular_names = proportions["Total"] > 10000
pretty_unisex = proportions["AbsDiff"] < 0.2
proportions[popular_names & pretty_unisex]
unisex = proportions[
    (proportions["Total"]   > 500) &
    (proportions["AbsDiff"] < 0.3)
].copy().reset_index()

unisex = unisex.groupby("name").sum()
unisex["Total"].head()
wordcloud = WordCloud(
    max_font_size=50, 
    stopwords=STOPWORDS,
    background_color='black',
    collocations=False,
    width=600,
    height=300,
)

image = wordcloud.generate_from_frequencies(unisex["Total"].to_dict())

plt.figure(figsize=(25, 10))
plt.title("Wordcloud for Popular Unisex Names", fontsize=35)
plt.imshow(image)
plt.axis('off')
plt.show()
proportions.to_csv("usa_names_gender_proportions.csv")