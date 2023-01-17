import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
%matplotlib inline 
#shows plot from matplotlib and seaborn in the Jupyter notebook
df = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
df.head()
df = pd.merge(df, items, on="item_id")

df = pd.merge(df, item_categories, on="item_category_id")

df.head()
revenueByDate = pd.DataFrame(df.groupby('date', as_index=False)['item_price'].sum())
revenueByDate["day"] = revenueByDate.date.str.extract("([0-9][0-9]).", expand = False)
revenueByDate["month"] = revenueByDate.date.str.extract(".([0-9][0-9]).", expand = False)
revenueByDate["year"] = revenueByDate.date.str.extract(".([0-9][0-9][0-9][0-9])", expand =False)
revenueByDate.head()
revenueByDate["date"] = pd.to_datetime(revenueByDate[["year", "month", "day"]])

plt.plot( "date", "item_price", data = revenueByDate.sort_values(by="date"))
plt.xticks(rotation=45)
plt.show()
revenueByDate["day"] = 1 #rewrite this column using ones so I can use the DatetimeIndex and aggregate sales by month
revenueByDate['monthlyDate'] = pd.to_datetime(revenueByDate[["year", "month", "day"]])

revenueByDate.head()
revenueByMonth = pd.DataFrame(revenueByDate.groupby("monthlyDate", as_index=False)["item_price"].sum())
revenueByMonth.head()
plt.plot("monthlyDate", "item_price", data = revenueByMonth.sort_values(by="monthlyDate"))
plt.xticks(rotation = 90);
revenueByMonth[revenueByMonth["item_price"] == revenueByMonth["item_price"].max()]

#gets the 2 highest item prices and their respective row index.
revenueByMonth["item_price"].nlargest(2)

revenueByMonth.iloc[[11,23],:]
monthlyRev = pd.DataFrame(revenueByDate.groupby(["month", "year"], as_index=False)["item_price"].sum())
monthlyRev.head()


g = sns.FacetGrid(data = monthlyRev.sort_values(by="month"), hue = "year", size = 5, legend_out=True)
g = g.map(plt.plot, "month", "item_price")
g.add_legend()
g;
df["item_id"] = df["item_id"].astype("category")
df["item_name"] = df["item_id"].astype("category")
df["item_category_id"] = df["item_category_id"].astype("category")
df["item_category_name"] = df["item_category_name"].astype("str")
#apparently, if item_category_name is set as category, we cannot use googletrans in the next section

#a nice way to visualize the content of a dataset and also its shape. It is built to be similar to str function in R.
def rstr(df): 
    return df.shape, df.apply(lambda x: [x.unique()])

rstr(df)
sales_by_category = df.groupby("item_category_name", 
                             as_index = False)["item_price"].sum().sort_values(by = "item_price")

top_sales = sales_by_category.nlargest(n=15, columns=["item_price"])

top_sales.set_index(np.arange(0,len(top_sales),1))
#The code below is used to translate each row of "item_category_name"
#However, this library needs to use the internet (to access Google)
#in order to make its translation. Kaggle doesn't let kernels to access
#the web, so in order to overcome this issue, I've uploaded top_sales
#dataframe after the translation, the same you should get after running this 
#piece of commented code in your local machine
'''
from googletrans import Translator

translator = Translator()

i = 0
for row in top_sales["item_category_name"]:
    english_word = translator.translate(row)
    top_sales.iloc[i,0] = english_word.text
    i+=1
'''

top_sales = pd.read_csv("../input/additional-data-for-competition/topsales.csv")

sns.barplot(y = "item_category_name", x = "item_price",
             data = top_sales)
plt.title("Sales for each one of the top 15 categories-products")
plt.xlabel("Sales")
plt.ylabel("Category-Product")
top_sales["item_category"] = top_sales.item_category_name.str.extract('([A-Za-z\ ]+)', expand=False) 

top_sales.head()
sns.barplot(y = "item_category", x = "item_price",
             data = top_sales)
plt.title("Sales for each one of the top 8 categories")
plt.xlabel("Sales")
plt.ylabel("Category-Product")
dailyItensByCat = pd.DataFrame(df.groupby(["item_category_name", "date"], as_index=False)["item_cnt_day"].sum())

dailyItensByCat["month"] = dailyItensByCat.date.str.extract(".([0-9][0-9]).", expand = False)
dailyItensByCat["year"] = dailyItensByCat.date.str.extract(".([0-9][0-9][0-9][0-9])", expand =False)
dailyItensByCat["day"] = 1 #create this column so I can use the DatetimeIndex
dailyItensByCat['monthlyDate'] = pd.to_datetime(dailyItensByCat[["year", "month", "day"]])

monthlyItensByCat = pd.DataFrame(dailyItensByCat.groupby(["monthlyDate", "item_category_name"],
                                                         as_index = False)["item_cnt_day"].sum())
monthlyItensByCat.head()
monthlyItensByCat = pd.DataFrame(dailyItensByCat.groupby(["monthlyDate", "item_category_name"],
                                                         as_index = False)["item_cnt_day"].sum())

monthlyItensByCat.head()
monthlyItensByCat["item_category"] = (
    monthlyItensByCat.item_category_name.str.extract(r'((?i)[А-Яa-я\ ]+)', expand=False))

monthlyItensByCat.head()

#As a record: I've spent hours looking for a way to extract 
#cyrillic alphabet, but there is no simple answer on the web.
#In the end, turned out the solution was simple indeed and achieved by 
#trial and error. I hope this will help someone when they need a way
#to extract cyrillic alphabet words so they won't spend such a long time
#looking for an answer :P 


countByCatByMonth = monthlyItensByCat.groupby(["item_category", "monthlyDate"], as_index=False)["item_cnt_day"].sum()
countByCatByMonth['item_category_trans'] = None
countByCatByMonth.head()
#Here again, we need internet access to perform the translations
#and Kaggle doesn't let us. So I uploaded the exact dataframe that
#should be generated after running the code below

'''
from googletrans import Translator

cat_translator = Translator()

obs = 0
already_translated = {'word': [], 'translation':[]}
#creates a new column in countByCatByMonth
countByCatByMonth['item_category_trans'] = None

#for each row...
for cat_name in countByCatByMonth["item_category"]:
    
    #check if it has already been translated
    if cat_name in already_translated['word']:
        #if it is, get the index of the original word on the dictionary...
        word_index = already_translated["word"].index(cat_name)
        #and use it to get the translated word
        countByCatByMonth.iloc[obs,3] = already_translated["translation"][word_index]
        #if the word was not translated yet...
    else:
        try:
            #translate it
            english_word = cat_translator.translate(cat_name)
            #append in dictionary for later use
            already_translated['word'].append(cat_name)
            already_translated['translation'].append(english_word.text)
            #write the translated word into dataframe
            countByCatByMonth.iloc[obs,3] = english_word.text
    
        except:
            print ("Error in row "+ cat_name)
    obs+=1
'''
countByCatByMonth = pd.read_csv("../input/additional-data-for-competition/countbycatbymonth.csv")
countByCatByMonth.head()
g = sns.FacetGrid(data = countByCatByMonth.sort_values(by="monthlyDate"), hue = "item_category_trans", legend_out=True, size = 8)
g = (g.map(plt.plot, "monthlyDate", "item_cnt_day").set(xticks=[0, 5, 11, 17, 23, 29, 35],
                                                        xticklabels=['2013-01-01', '2013-06-01', '2013-12-01', 
                                                                     '2014-06-01','2014-12-01', '2015-06-01','2015-12-01']))
g.add_legend()
plt.xlabel("Monthly Date")
plt.ylabel("Number of itens sold");

ts = revenueByMonth.set_index("monthlyDate")
ts.head()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey = 'row')
ax1.plot(ts)
ax1.set_title("original")
ax2.plot(trend)
ax2.set_title("trend")
ax3.plot(seasonal)
ax3.set_title("seasonal")

ax4.plot(residual)
ax4.set_title("residual")
f.set_figheight(7)
f.set_figwidth(10)

#let me rotate the labels of x-axis. 
#Otherwise they get mixed and very hard to read.
for ax in f.axes:
    plt.sca(ax)
    plt.xticks(rotation = 45)
revenueByMonth["item_price_x"] = revenueByMonth["item_price"].shift(1)

slideRevenueByMonth = revenueByMonth.drop(columns = "monthlyDate")

sns.lmplot(data = slideRevenueByMonth, x = "item_price_x", y = "item_price", ci=False);

slideRevenueByMonth.head()
slideRevenueByMonth["log_itemprice"] = np.log(slideRevenueByMonth["item_price"])

slideRevenueByMonth["log_itemprice_x"] = np.log(slideRevenueByMonth["item_price_x"])
sns.lmplot(data = slideRevenueByMonth, x = "log_itemprice_x", y = "log_itemprice", ci=False);

slideRevenueByMonth.head()
xtrain = slideRevenueByMonth.iloc[1:20,3].values.reshape(-1,1)
ytrain = slideRevenueByMonth.iloc[1:20,2]

xtest = slideRevenueByMonth.iloc[20:,3].values.reshape(-1,1)
ytest = slideRevenueByMonth.iloc[20:, 2]
#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state = 42)

rf_reg.fit(xtrain, ytrain)

ypred_rf = rf_reg.predict(xtest)

#LinearRegressor

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(xtrain,ytrain)

ypred_linear = linear.predict(xtest)
#Just putting all predictions in one dataframe to get things more organized
testing = pd.concat([pd.DataFrame(ypred_linear),
                     pd.DataFrame(ypred_rf),pd.DataFrame(ytest).reset_index().drop(["index"], axis = 1)], axis = 1)

testing.columns = ["log_itemprice_pred_ln","log_itemprice_pred_rf", "log_itemprice"]

#making the plot and also drawing a 45 degree line so we can see the ideal situation
grid = sns.JointGrid(y = testing.log_itemprice_pred_rf, x = testing.log_itemprice, space=0, size=6, ratio=50)
grid.plot_joint(plt.scatter, color="g")
plt.plot([17.8, 19.0], [17.8, 19.0], linewidth=2);
grid = sns.JointGrid(y = testing.log_itemprice_pred_ln, x = testing.log_itemprice, space=0, size=6, ratio=50)
grid.plot_joint(plt.scatter, color="g")
plt.plot([17.8, 19.0], [17.8, 19.0], linewidth=2);
slideRevenueByMonth["UpOrDown"] = np.sign(slideRevenueByMonth["item_price"] - slideRevenueByMonth["item_price_x"])
slideRevenueByMonth["PreviousUpOrDown"] = slideRevenueByMonth["UpOrDown"].shift(1)
#drop two rows that have null values in UpOrDown and PreviousUpOrDown
slideRevenueByMonth = slideRevenueByMonth.dropna()
slideRevenueByMonth.head()
xtrain = slideRevenueByMonth.iloc[2:22,[3,5]]
ytrain = slideRevenueByMonth.iloc[2:22,2]

xtest = slideRevenueByMonth.iloc[22:,[3,5]]
ytest = slideRevenueByMonth.iloc[22:, 2]

##########Random Forest###########
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state = 42)

rf_reg.fit(xtrain, ytrain)

ypred_rf = rf_reg.predict(xtest)

##########LinearRegressor###########

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(xtrain,ytrain)

ypred_linear = linear.predict(xtest)
#Creating the predictions dataframe
testing = pd.concat([pd.DataFrame(ypred_linear),
                     pd.DataFrame(ypred_rf),pd.DataFrame(ytest).reset_index().drop(["index"], axis = 1)], axis = 1)

testing.columns = ["log_itemprice_pred_ln","log_itemprice_pred_rf", "log_itemprice"]

grid = sns.JointGrid(y = testing.log_itemprice_pred_rf,x = testing.log_itemprice, space=0, size=6, ratio=50)
grid.plot_joint(plt.scatter, color="g")
plt.plot([17.8, 19.0], [17.8, 19.0], linewidth=2)
plt.title("Random Forest x True", fontsize = 20);
grid = sns.JointGrid(y = testing.log_itemprice_pred_ln,x = testing.log_itemprice, space=0, size=6, ratio=50)
grid.plot_joint(plt.scatter, color="g")
plt.plot([17.8, 19.0], [17.8, 19.0], linewidth=2)
plt.title("Linear Regression x True", fontsize = 20);