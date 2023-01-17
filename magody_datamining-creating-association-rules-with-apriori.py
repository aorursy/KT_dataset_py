import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt

import numpy as np

pd.set_option('display.max_columns', 6)
path_dataset = '../input/19560-indian-takeaway-orders/'

file = 'restaurant-1-orders.csv'

# header: Order Number,Order Date,Item Name,Quantity,Product Price,Total products

df_orders = pd.read_csv(path_dataset+file)



# The data is already cleaned



print("Original dataframe")

df_orders.head(10)
min_support = 0.1  # i set it to 10%. Only search the sets with that support
metric = "lift"  # another usual metrics is confidence of the rule with at least 50%

min_threshold_for_metric = 1
number_rules_to_visualize = 3  # for the grouped bar chart
basket = (

    # group data (item name should be grouped in order to unstack later)

    df_orders.groupby([

        'Order Number', 'Item Name'

    ])['Quantity'].sum()  # Agregate quantity data just to apply unstack, the value doesnt change

    .unstack().reset_index()  # Transform to 1 transaction per row

    .fillna(0)  # fill the products that its not in the order with 0

    .set_index('Order Number')  # set the order number as index

)



print("One order per row with the quantity of each product")

basket.head()
basket_boolean_set = basket.applymap(lambda quantity: 1 if int(quantity) >= 1 else 0)



print("Converted quantity to boolean values")

basket_boolean_set.head()
frequent_itemsets = apriori(basket_boolean_set, min_support=min_support, use_colnames=True)



print("\nFrequent itemsets using apriori and minimun support equals to %.2f percent" % (min_support*100))

frequent_itemsets.head(10)
rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold_for_metric)
rules['antecedents'] = rules['antecedents'].apply(lambda frozen_set: str(set(frozen_set)))

rules['consequents'] = rules['consequents'].apply(lambda frozen_set: str(set(frozen_set)))

rules['support'] = rules['support'].apply(lambda value: round(value, 2))

rules['confidence'] = rules['confidence'].apply(lambda value: round(value, 2))

rules['lift'] = rules['lift'].apply(lambda value: round(value, 2))

rules['leverage'] = rules['leverage'].apply(lambda value: round(value, 2))

rules['conviction'] = rules['conviction'].apply(lambda value: round(value, 2))
pd.set_option('display.max_columns', 10)

print("Rules using the metric '%s' with a minimun threshold of %s equals to %.2f " % (metric, metric, min_threshold_for_metric))

rules.head(10)
rules_top = rules.iloc[0:number_rules_to_visualize, :]

labels = ["=>".join(rule) for rule in rules_top.iloc[:, 0:2].values]  # format {set1} => {set2} for labels

x = np.arange(len(labels))  # the label locations

width = 0.3

fig, ax = plt.subplots(figsize=(15,6))

rects1 = ax.bar(x - width/2, rules_top['conviction'].values, width, label='conviction')

rects2 = ax.bar(x + width/2, rules_top['confidence'].values, width, label='confidence')

rects3 = ax.bar(x + width*1.5, rules_top['lift'].values, width, label='lift')

# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Metrics')

ax.set_title('Metrics for top ' + str(number_rules_to_visualize) + ' rules evaluated with ' + metric)

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()







def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)

autolabel(rects2)

autolabel(rects3)



mng = plt.get_current_fig_manager()

# mng.window.state('zoomed')  # maximize the screen in windows

plt.show()