import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')


df_store = pd.read_csv("/kaggle/input/retaildataset/stores data-set.csv")

df_feature = pd.read_csv("/kaggle/input/retaildataset/Features data set.csv",parse_dates=["Date"])

df_sales = pd.read_csv("/kaggle/input/retaildataset/sales data-set.csv",parse_dates=["Date"])
print(df_store.info())

print(df_store.head())
print(df_feature.info())

print(df_feature.head())
print(df_sales.info())

print(df_sales.head())
data_date = df_feature.groupby("Date").agg({"Temperature":"mean"

                                            ,"Fuel_Price":"mean"

                                            ,"IsHoliday":"sum"

                                            ,"CPI":"mean"

                                           ,"Unemployment":"mean"})

data_date = data_date.sort_index()

temp_date_data = data_date[:'2012-12-10']



data_sales_date = df_sales.groupby("Date").agg({"Weekly_Sales":"sum"})

data_sales_date.sort_index(inplace=True)

data_sales_date.Weekly_Sales = data_sales_date.Weekly_Sales/1000000

data_sales_date.Weekly_Sales = data_sales_date.Weekly_Sales.apply(int)

data = pd.merge(data_sales_date, temp_date_data, left_index=True,right_index=True, how='left')

data["IsHoliday"] = data["IsHoliday"].apply(lambda x: True if x == 45.0 else False )

print(data.describe())
plt.style.use('fivethirtyeight')

#plt.figure(figsize=(15,4))

fig, ax = plt.subplots(5,1,figsize=(15,10),sharex=True) 

data["Weekly_Sales"].plot(ax=ax[0],title="Weekly Sales/sales on Holiday")

data[data.IsHoliday==True]["Weekly_Sales"].plot(marker="D",ax=ax[0],legend="Holiday Week sale")

data["Temperature"].plot(ax=ax[1], title="Temperature")

data["Fuel_Price"].plot(ax=ax[2],title="Fuel_Price")

data["CPI"].plot(ax=ax[3],title="CPI")

data["Unemployment"].plot(ax=ax[4],title="Unemployment")

sns.heatmap(data.corr(),annot=True)
data_sales_month = data.groupby(data.index.month).agg({"Weekly_Sales":"sum"})

plt.figure(figsize=(10, 5))

sns.barplot(x=data_sales_month.index,y=data_sales_month.Weekly_Sales)

plt.title("Month wise Sales")

plt.xlabel("Month")

plt.ylabel("Sales")
data_sales_year = data.groupby(data.index.year).agg({"Weekly_Sales":"sum"})



sns.barplot(x=data_sales_year.index,y=data_sales_year.Weekly_Sales)

plt.title("Year wise Sales")

plt.xlabel("Year")

plt.ylabel("Sales")

from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data["Weekly_Sales"], period=45) 
plt.figure(figsize=(15, 7))

plt.plot(decomposition.trend)

plt.plot(decomposition.seasonal)

plt.plot(decomposition.resid)

plt.legend(["Trend", "Seasonal","Resid"], loc ="upper right") 
data_Store = df_feature.groupby("Store").agg({"Temperature":"mean","Fuel_Price":"mean","IsHoliday":"sum"})



temp_store = df_sales.groupby("Store").agg({"Weekly_Sales":"sum"})

temp_store.Weekly_Sales = temp_store.Weekly_Sales/1000000

temp_store.Weekly_Sales = temp_store.Weekly_Sales.apply(int)

data_Store.set_index(np.arange(0,45),inplace=True)

df_store["temp"] = data_Store.Temperature

df_store["Fuel_Price"] = data_Store.Fuel_Price

df_store["holiday"] = data_Store.IsHoliday

df_store["Weekly_Sales"] = temp_store.Weekly_Sales
df_store.describe()
fig,ax = plt.subplots(1,3,figsize=(15, 4))

sns.countplot(df_store.Type,ax=ax[0])

sns.swarmplot(data = df_store,y="Size",x="Type",ax=ax[1])



sns.boxplot(data = df_store,y="Weekly_Sales",x="Type",ax=ax[2])



len(df_sales["Dept"].unique())
data_Dept = df_sales.groupby("Dept").agg({"Weekly_Sales":"sum"})

data_Dept.Weekly_Sales = data_Dept.Weekly_Sales/10000

data_Dept.Weekly_Sales = data_Dept.Weekly_Sales.apply(int)

data_Dept.sort_values(by="Weekly_Sales")
fig1, ax1 = plt.subplots(figsize=(15, 4))

#ordered_df = data_Dept.sort_values(by='Weekly_Sales')

plt.vlines(x=data_Dept.index, ymin=0, ymax=data_Dept['Weekly_Sales'], color='skyblue')

plt.plot(data_Dept.index,data_Dept['Weekly_Sales'], "o")

plt.title("Departmentwise Sales")

plt.ylabel("Sales")

plt.xlabel("Department")
sales_date_store = df_sales.groupby(["Date","Store"]).agg({"Weekly_Sales":"sum"})

sales_date_store.sort_index(inplace=True)

sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales/10000

sales_date_store.Weekly_Sales = sales_date_store.Weekly_Sales.apply(int)

data_table = pd.merge(df_feature,sales_date_store ,  how='left', on=["Date","Store"])

data_table = pd.merge(data_table,df_store[["Store","Type"]] ,  how='left', on=["Store"])

data_table.head(20)

data_train = data_table[data_table.Weekly_Sales.notnull()]

data_test = data_table[data_table.Weekly_Sales.isnull()]
plt.figure(figsize=(15, 5))

sns.barplot(x=data_train.Date.dt.year, y=data_train.Weekly_Sales,hue=data_train.Type)
plt.figure(figsize=(15, 7))

sns.barplot(x=data_train.Date.dt.month, y=data_train.Weekly_Sales,hue=data_train.Type)
plt.figure(figsize=(15,4))

train_markdown = data_table[data_table.MarkDown2.notnull()]

train_markdown = train_markdown.groupby("Date").agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})





plt.plot(train_markdown.index,train_markdown.MarkDown1)

plt.plot(train_markdown.index,train_markdown.MarkDown2)

plt.plot(train_markdown.index,train_markdown.MarkDown3)

plt.plot(train_markdown.index,train_markdown.MarkDown4)

plt.plot(train_markdown.index,train_markdown.MarkDown5)

plt.title("Timeline Markdown")

plt.ylabel("Markdown")

plt.xlabel("Date")
train_markdown.hist(figsize=(10,8),bins=6,color='Y')



plt.tight_layout()

plt.show()
train_markdown_month = train_markdown.groupby(train_markdown.index.month).agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})



train_markdown_month.plot(kind='bar', stacked=True,figsize=(15,6))

plt.title("Stacked Monthwise Morkdown")

plt.ylabel("Markdown")
train_markdown_1 = data_table[data_table.MarkDown2.notnull()]

train_markdown_type = train_markdown_1.groupby("Type").agg({"MarkDown1":"mean","MarkDown2":"mean","MarkDown3":"mean","MarkDown4":"mean","MarkDown5":"mean"})



train_markdown_type.plot(kind='bar', stacked=True,figsize=(10,4))

plt.title("Stacked StoreType Wise")

plt.ylabel("Markdown")


from fancyimpute import IterativeImputer

from sklearn.metrics import mean_squared_error



from sklearn.svm import SVR, LinearSVR, NuSVR

from sklearn.linear_model import ElasticNet, Lasso, RidgeCV,LinearRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor

import xgboost as xgb

import lightgbm as lgb
def createdummies(data,cols):

    for col in cols:

        one_hot = pd.get_dummies(data[col],prefix=col)

        data = data.join(one_hot)

        data.drop(col,axis = 1,inplace=True)

    

    return data

        


# imputing the missing value

itt = IterativeImputer()

df = itt.fit_transform(data_table[["MarkDown1","MarkDown2","MarkDown3","MarkDown4","MarkDown5"]]) 

data_table.MarkDown1 = df[:,0]

data_table.MarkDown2 = df[:,1]

data_table.MarkDown3 = df[:,2]

data_table.MarkDown4 = df[:,3]

data_table.MarkDown5 = df[:,4]



data_table['CPI'].fillna((data_table['CPI'].mean()), inplace=True)

data_table['Unemployment'].fillna((data_table['Unemployment'].mean()), inplace=True)

data_table['IsHoliday'] = data_table['IsHoliday'].map({True:0,False:1})



#create new column

data_table["Month"] = data_table.Date.dt.month

data_table["Year"] = data_table.Date.dt.year

data_table["WeekofYear"] = data_table.Date.dt.weekofyear

data_table.drop(['Date'],axis=1,inplace=True)



#create dummies out of categorical column

data_table = createdummies(data_table,["Type","Month","Year","WeekofYear"])



data_table.columns
data_train = data_table[data_table.Weekly_Sales.notnull()]

data_test = data_table[data_table.Weekly_Sales.isnull()]

X = data_train.drop('Weekly_Sales', axis=1)

y = data_train['Weekly_Sales']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)






classifiers = [

    LinearRegression(),

    ElasticNet(),

    RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]),

    KernelRidge(alpha=0.6, kernel='polynomial', degree=3, coef0=2.5),

    Lasso(alpha =16, random_state=100),

    ElasticNet(alpha=0.8),

    DecisionTreeRegressor(),

    RandomForestRegressor(),

    GradientBoostingRegressor(),

    AdaBoostRegressor(),

    SVR(), 

    LinearSVR(), 

    NuSVR(),

    xgb.XGBRegressor(),

    lgb.LGBMRegressor()

    ]



name = []

score = []

models = []

rmse = []

i = 0

for classifier in classifiers:

    classifier.fit(X_train, y_train)   

    name.append(type(classifier).__name__)

    score.append(classifier.score(X_test, y_test))

    models.append(classifier)

    rmse.append(np.sqrt(mean_squared_error(classifier.predict(X_test), y_test)))

df_score = pd.DataFrame(list(zip(name,rmse, score, models)),columns=['name','rmse','score',"model"])

df_score.set_index('name',inplace=True)

df_score.sort_values(by=['score'],inplace=True)

df_score
model = df_score.loc["XGBRegressor","model"]

data_test.drop(['Weekly_Sales'],axis=1,inplace=True)

predict = model.predict(data_test)

predict