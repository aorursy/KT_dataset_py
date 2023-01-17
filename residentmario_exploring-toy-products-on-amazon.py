import numpy as np

import pandas as pd



products = pd.read_csv("../input/amazon_co-ecommerce_sample.csv")
products.head(3)
import numpy as np



def mapcategories(srs):

    if pd.isnull(srs):

        return []

    else:

        return [cat.strip() for cat in srs.split(">")]

    

category_lists = products['amazon_category_and_sub_category'].apply(mapcategories)

category_lists.map(lambda lst: len(lst)).value_counts()
import networkx as nx

DG = nx.DiGraph()



category_lists.map(lambda cats: DG.add_nodes_from(cats))

category_lists.map(lambda cats: [DG.add_edge(cats[i], cats[i + 1]) for i in range(len(cats) - 1)])



print("The number of categorical links possible is {0}.".format(len(DG.edges())))
# print(list(nx.simple_cycles(DG)))

# products.iloc[

#     category_lists[category_lists.map(lambda lst: 'Beach Toys' in lst)].index

# ].head()[['product_name', 'amazon_category_and_sub_category']]



top = category_lists[category_lists.map(lambda c: len(c) > 0)].map(lambda l: l[0])

bottom = category_lists[category_lists.map(lambda c: len(c) > 0)].map(lambda l: l[-1])



print("There are {0} possible top-level (root) categories in this dataset.".format(len(set(top))))

print("There are {0} possible bottom-level (leaf) categories in this dataset.".format(len(set(bottom))))



products = products.assign(root_category=top.astype('category'), 

                           leaf_category=bottom.astype('category'))
# Reformatting the number_of_reviews and price columns.

products['number_of_reviews'] = products['number_of_reviews'].str.replace(",", "").astype(float)



def mapprice(v):

    if pd.isnull(v):

        return 0

    try:

        return float(v[1:])

    except ValueError:

        return 0

    

products['price'] = products['price'].map(mapprice)
# Creating the plot.

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("white")



f, axarr = plt.subplots(2, 2)

f.subplots_adjust(hspace=1)

plt.suptitle('Top 10 Leaf Categories By...', fontsize=18)



bar_kwargs = {'figsize': (13, 11), 'fontsize': 14, 'color': 'goldenrod'}



products['leaf_category'].value_counts().head(10).plot.bar(

    title='...Number of Products', ax=axarr[0][0], **bar_kwargs

)



(products.groupby('leaf_category')

     ['number_of_reviews']

     .sum()

     .sort_values(ascending=False)

     .rename_axis(None)

     .head(10)

     .plot.bar(

        title='...Number of Reviews', ax=axarr[0][1], **bar_kwargs

     ))



(products.groupby('leaf_category')

     ['price']

     .sum()

     .sort_values(ascending=False)

     .rename_axis(None)

     .head(10)

     .plot.bar(

        title='...Gross Product Value (£)', ax=axarr[1][0], **bar_kwargs

     ))



(products

     .assign(

         pval=products.apply(lambda p: p['price'] * p['number_of_reviews'], axis='columns')

     )

     .groupby('leaf_category')

     .sum()

     .pipe(lambda df: df.assign(pval=df['pval'] / df['pval'].max()))

     .pval

     .sort_values(ascending=False)

     .rename_axis(None)

     .head(10)

     .plot.bar(

         title='...Product Value Sold (Index, Estimated)', 

         ax=axarr[1][1], **bar_kwargs

     )

)



sns.despine()



for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]:

    axarr[a][b].title.set_fontsize(16)

    axarr[a][b].set_xticklabels(axarr[a][b].get_xticklabels(), 

                                rotation=45, ha='right', fontsize=14)
from wordcloud import WordCloud



def wordcloudify(cat):

    return WordCloud().generate(

        " ".join(products.query('leaf_category == "{0}"'.format(cat))['product_name'].values)

    )



f, axarr = plt.subplots(4, 1, figsize=(14, 32))

# f.subplots_adjust(hspace=1)



# Display the generated image:

axarr[0].imshow(wordcloudify("Toys"), 

                interpolation='nearest', aspect='auto')

axarr[0].axis("off")

axarr[0].set_title("Characters & Brands > ... > Toys", fontsize=16)



axarr[1].imshow(wordcloudify("Vehicles"), 

                interpolation='nearest', aspect='auto')

axarr[1].axis("off")

axarr[1].set_title("Die Cast & Toy Vehicles > ... > Vehicles", fontsize=16)



axarr[2].imshow(wordcloudify("Educational Games"), 

                interpolation='nearest', aspect='auto')

axarr[2].axis("off")

axarr[2].set_title("Games > ... > Education Games", fontsize=16)



axarr[3].imshow(wordcloudify("Science Fiction & Fantasy"), 

                interpolation='nearest', aspect='auto')

axarr[3].axis("off")

axarr[3].set_title("Figures & Playsets > ... > Science Fiction & Fantasy", fontsize=16)

pass
df = (products[

        products['leaf_category'].isin(

            list(products['leaf_category'].value_counts().head(10).index.values)

        )

      ].query('price > 0')

       .pipe(lambda df: df.assign(leaf_category=df['leaf_category'].astype(str)))

       .query('price < 100') 

       .sample(1000)

     )



with sns.plotting_context("notebook", font_scale=1.5):

    g = sns.FacetGrid(df, row='leaf_category', size=2.5, aspect=4)



    g.map(sns.violinplot, "price", inner=None, color='lightyellow', linewidth=1)

    g.map(sns.swarmplot, "price", size=8, color='goldenrod')

    g.despine(bottom=True, left=True)
products['manufacturer'].value_counts().head().index
df = (products[

        products['manufacturer'].isin(

            ['Disney', 'Playmobil', 'LEGO', 'Hot Wheels', 'Mattel']

        )

      ]

         .query('price <= 100').query('price > 0'))



with sns.plotting_context("notebook", font_scale=1.5):

    g = sns.FacetGrid(df, row='manufacturer', hue='manufacturer', size=3, aspect=3.8)

    g.map(sns.kdeplot, "price", color='goldenrod', clip_on=False, shade=True)

    g.despine(bottom=True, left=True)

    g.set(yticks=[])
def splitrating(rating):

    try:

        return rating.split(" ")[0]

    except AttributeError:

        return np.nan



products['average_review_rating'] = products['average_review_rating'].map(splitrating)

products['average_review_rating'] = products['average_review_rating'].astype(float)



def recommend_me(item):

    tokens = item.split(" ")

    tokens = [t.lower() for t in tokens]

    df = products[products.product_name.map(lambda name: all(n in name.lower() for n in tokens))]

    

    if len(df[df['average_review_rating'] >= 4.0]) == 0:

        return df.sort_values(by='average_review_rating').iloc[0]

    else:

        return df[df['average_review_rating'] >= 4.0].sort_values(by='number_of_reviews').iloc[0]
recommend_me("Star Wars Action Figure").product_name
recommend_me("Fire Truck").product_name
recommend_me("Pony").product_name