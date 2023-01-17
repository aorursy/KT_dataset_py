import warnings

warnings.filterwarnings('ignore')



import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns

import matplotlib.gridspec as gridspec

%matplotlib inline
inp0 = pd.read_csv("../input/finance-accounting-courses-udemy-13k-course/udemy_output_All_Finance__Accounting_p1_p626.csv")
#glance at the columns and types of the data

inp0.info()
#glance at the data

inp0.head()
inp0.drop(["id","url","discount_price__currency","price_detail__currency","is_wishlisted"],axis = 1,inplace = True)

inp0.drop(["discount_price__price_string","price_detail__price_string"],axis = 1,inplace = True)

inp0.drop('created',axis = 1, inplace = True)
mis_val = 100*inp0.isnull().sum()/inp0.shape[0]

mis_val = mis_val.reset_index(name = 'Mis_perc')



plt.figure(figsize = [8,5])

plt.barh(mis_val['index'], mis_val['Mis_perc'])

a = np.arange(0,12,2)

b = ["{}%".format(i) for i in a]

plt.xticks(a,b)

plt.xlabel("$Percentage of Missing Values$")

plt.ylabel("$Attributes in dataset$")

plt.title("Missing Values comparision across attribues")

plt.show()
inp0.price_detail__amount.fillna(value = 0, inplace = True)



inp0['Discount'] = 100*(inp0['price_detail__amount'] - inp0["discount_price__amount"])/inp0['price_detail__amount'] if inp0["discount_price__amount"] is not np.nan else np.nan
inp0['PublishedYear'] = inp0.published_time.apply(lambda x: x[:4])

inp0['PublishedMonth'] = inp0.published_time.apply(lambda x: x[5:7])

inp0.drop('published_time',axis = 1, inplace = True)
fig = plt.figure(figsize = [10,6])

gs = gridspec.GridSpec(3,3)

f_ax1 = fig.add_subplot(gs[0, :])

plt.sca(f_ax1)

sns.boxplot(x = 'num_subscribers',data = inp0)

plt.title("Distribution for number of subscribers")

f_ax2 = fig.add_subplot(gs[1:,:])

plt.sca(f_ax2)

sns.distplot(inp0['num_subscribers'],bins = 100, hist_kws = {"edgecolor":"white"})

plt.ylabel('density')

plt.show()
inp0[inp0.num_subscribers > 150000]["title"]
inp0.is_paid.value_counts(normalize = True).plot.pie(figsize = [5,5], explode = [0.2,0.] )

plt.show()
fig = plt.figure(figsize = [10,6])

gs = gridspec.GridSpec(3,3)

f_ax1 = fig.add_subplot(gs[0, :])

plt.sca(f_ax1)

sns.boxplot(x = 'avg_rating',data = inp0)

plt.title("Distribution for average rating")

f_ax2 = fig.add_subplot(gs[1:,:])

plt.sca(f_ax2)

sns.distplot(inp0['avg_rating'],bins = 10, hist_kws = {"edgecolor":"white"})

plt.ylabel('density')

plt.show()
fig = plt.figure(figsize = [10,6])

gs = gridspec.GridSpec(3,3)

f_ax1 = fig.add_subplot(gs[0, :])

plt.sca(f_ax1)

sns.boxplot(x = 'num_reviews',data = inp0)

plt.title("Distribution for number of reviews")

f_ax2 = fig.add_subplot(gs[1:,:])

plt.sca(f_ax2)

sns.distplot(inp0['num_reviews'],bins = 100, hist_kws = {"edgecolor":"white"})

plt.ylabel('density')

plt.show()
fig = plt.figure(figsize = [10,6])

gs = gridspec.GridSpec(3,3)

f_ax1 = fig.add_subplot(gs[0, :])

plt.sca(f_ax1)

sns.boxplot(x = 'price_detail__amount',data = inp0)

plt.title("Distribution for price of the courses")

f_ax2 = fig.add_subplot(gs[1:,:])

plt.sca(f_ax2)

sns.distplot(inp0['price_detail__amount'],bins = 50, hist_kws = {"edgecolor":"white"})

plt.ylabel('density')

plt.show()
fig = plt.figure(figsize = [10,6])

gs = gridspec.GridSpec(3,3)

f_ax1 = fig.add_subplot(gs[0, :])

plt.sca(f_ax1)

sns.boxplot(x = 'Discount',data = inp0)

plt.title("Distribution for Discount percentages")

plt.show()
inp0.PublishedYear.value_counts().plot.barh(figsize = [8,4])

plt.title('Count of the courses published')

plt.show()
inp0.PublishedMonth.value_counts().plot.barh(figsize = [8,4],width = 0.5)

plt.title('Count of the courses published')

plt.show()
res = inp0[['num_subscribers','avg_rating','num_reviews','num_published_lectures','price_detail__amount','Discount']].corr()



sns.heatmap(res, annot = True, cmap = 'RdYlGn')

plt.show()
def q5(x):

    return np.quantile(x,0.05)

def q90(x):

    return np.quantile(x,0.9)
fig = plt.figure(figsize = [10,10])

gs = gridspec.GridSpec(2,2)

f_ax1 = fig.add_subplot(gs[0, 0])

plt.sca(f_ax1)

sns.boxplot(x = 'is_paid',y = 'avg_rating',data = inp0)

plt.title("Rating for paid and free courses")

f_ax2 = fig.add_subplot(gs[0,1])

plt.sca(f_ax2)

sns.boxplot(x = 'is_paid', y = 'num_published_lectures', data = inp0)

plt.title("Number of lectures for free and paid lectures")

f_ax3 = fig.add_subplot(gs[1,0])

sns.barplot(x = 'is_paid', y = 'avg_rating', data = inp0, estimator = q5)

plt.title("5th Quantile Rating for paid and free courses")

f_ax4 = fig.add_subplot(gs[1,1])

sns.barplot(x = 'is_paid',y = 'num_published_lectures', data = inp0, estimator = q90)

plt.title("90th Quantile Lectures for paid and free courses")

plt.show()
inp0['lec_cat'] = pd.qcut(inp0.num_published_lectures, q = [0, .25, .75, 1], labels = ['low','medium','high'])



fig = plt.figure(figsize = [11,10])

gs = gridspec.GridSpec(2,2)

ax1 = fig.add_subplot(gs[0,0])

plt.sca(ax1)

sns.boxplot(x = 'lec_cat', y = 'avg_rating', data = inp0)

ax2 = fig.add_subplot(gs[0,1])

plt.sca(ax2)

sns.boxplot(x = 'lec_cat', y = 'price_detail__amount', data = inp0)

f_ax3 = fig.add_subplot(gs[1,0])

sns.barplot(x = 'lec_cat', y = 'avg_rating', data = inp0, estimator = q5)

plt.title("5th Quantile Rating for lecture categories")

plt.show()