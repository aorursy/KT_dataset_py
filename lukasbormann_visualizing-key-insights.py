import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")



print(sales.isnull().sum())

sales.head()
print(shops.isnull().sum())

shops.head()
def print_lineplot(df, ycol, title="title", xlabel="", ylabel=""):

    '''

    This method takes an dataframe with 2 indices and plots.

    df = dataframe

    ycol = column that should be plot against the index

    title = subplot title

    xlabel = label of x axis

    ylabel = label of y axis

    '''

    

    num_plot = df.index.max()[0]//10

    fig, ax = plt.subplots(num_plot+1, figsize=(15,30))

    cmap = plt.get_cmap('jet')



    for ten_elem in range(0, num_plot):

        for elem in range(10*ten_elem,10*ten_elem + 11):

            color = cmap((elem-(10*ten_elem))/10)

            ax[ten_elem].plot(df.xs([elem], level=0)[ycol], c=color)





        ax[ten_elem].legend(range(10*ten_elem,10*ten_elem + 10), labelspacing=0.25, ncol=2, prop={'size': 9})

        ax[ten_elem].set_xlabel(xlabel)

        ax[ten_elem].set_ylabel(ylabel)



    for elem in range(10*num_plot,10*num_plot + df.index.max()[0] - num_plot*10):

        color = cmap((elem-(10*num_plot))/10)

        ax[num_plot].plot(df.xs([elem], level=0)[ycol],c=color)



    ax[num_plot].legend(range(10*num_plot,10*num_plot + df.index.max()[0] - num_plot*10), labelspacing=0.25, ncol=2, prop={'size': 9})

    ax[num_plot].set_xlabel(xlabel)

    ax[num_plot].set_ylabel(ylabel)





    fig.suptitle(title)

    plt.tight_layout()

    plt.show()

    plt.clf()
# Order data after each shop

check = sales.groupby(["shop_id", "date_block_num"]).sum()



print_lineplot(check, ycol="item_cnt_day", title="items sold of shops over a period of time", xlabel="Months", ylabel="items sold per day")
shop_outperformed = sales[sales["date_block_num"].isin([10, 11, 23, 24])]["date"].unique()

shop_outperformed = set([x[3:10] for x in shop_outperformed])

print(shop_outperformed)
check_economy = sales.groupby("date_block_num").sum()

check_price = sales.groupby("date_block_num").mean()

# calcualte the total revenue

check_earnings = sales.groupby("date_block_num").mean()

check_earnings["earnings"] = check_price["item_price"] * check_economy["item_cnt_day"]



fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,7))

check_economy.item_cnt_day.plot(ax=ax1)

sns.regplot(data=check_economy, x=check_economy.index, y="item_cnt_day", ax=ax1)



check_price.item_price.plot(ax=ax2)

sns.regplot(data=check_price, x=check_price.index, y="item_price", ax=ax2)



check_earnings.earnings.plot(ax=ax3)

sns.regplot(data=check_earnings, x=check_earnings.index, y="earnings", ax=ax3)
example_shop1 = sales[sales["shop_id"] == 10]

item1 = example_shop1[example_shop1["item_id"] == 6093]

#print(item1)



example_shop2 = sales[sales["shop_id"] == 9]

item2 = example_shop2[example_shop2["item_id"] == 6093]

#print(item2)



fig, ax = plt.subplots(figsize=(15,7))



item1.plot(x="item_price", y="item_cnt_day", ax=ax)

item2.plot(x="item_price", y="item_cnt_day", ax=ax)
sales_and_items = sales.merge(items, on="item_id")

check_categroy = sales_and_items.groupby(["item_category_id", "date_block_num"]).sum()

check_categroy["earnings"] = check_categroy["item_price"] * check_categroy["item_cnt_day"]



print_lineplot(check_categroy, ycol="earnings", title="Earnings of product category over time", ylabel="Earnings", xlabel="Months")
check_item = sales.groupby(["item_id"]).mean()

print(check_item.describe())

sns.jointplot(data=check_item, x="item_price", y="item_cnt_day")