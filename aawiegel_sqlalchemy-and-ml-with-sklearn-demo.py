# import relevant libraries



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import sqlalchemy as sqla
# Connect to SQLite database found in database.sqlite



metadata = sqla.MetaData()



engine =  sqla.create_engine("sqlite:///../input/database.sqlite")



table_names = engine.table_names()



tables = dict()



for table in table_names:

    print("\n"+table+" columns:\n")

    tables[table] = sqla.Table(table, metadata, autoload=True, autoload_with=engine)

    for column in tables[table].c.keys():

        print(column)
# Create SQL query function (We'll use a similar query again later)



# Select distinct IndicatorNames (to avoid repeating the same indicator for each country)



def find_indicator_strings(indicator):

    stmt = sqla.select([tables["Indicators"].c.IndicatorName.distinct(), 

                        tables["Indicators"].c.IndicatorCode,

                        tables["Series"].c.LongDefinition])



    # Use JOIN to find description under series name



    stmt = stmt.select_from(

        tables["Indicators"].join(tables["Series"], 

                                 tables["Indicators"].c.IndicatorCode == tables["Series"].c.SeriesCode)

    )



    # Find indicators that have indicator somewhere



    return stmt.where(tables["Indicators"].c.IndicatorName.ilike("%"+indicator+"%"))



stmt = find_indicator_strings("life expectancy")



# Connect to the engine and execute the statement.



conn = engine.connect()



for result in conn.execute(stmt):

    print(result.IndicatorName, result.IndicatorCode)

    print(result.LongDefinition)

    print("\n")



conn.close()
%matplotlib inline

# Create SQL query



stmt = sqla.select([tables["Indicators"].c.CountryName, tables["Indicators"].c.Value.label("LifeExpectancy")])





# Use to avoid selecting regions instead of countries

stmt = stmt.select_from(tables["Indicators"].join(tables["Country"],

                                                 tables["Country"].c.CountryCode == tables["Indicators"].c.CountryCode))



stmt = stmt.where(sqla.and_(tables["Country"].c.Region.isnot(""),

                            tables["Indicators"].c.IndicatorCode == "SP.DYN.LE00.IN",

                            tables["Indicators"].c.Year == 2010)

    )





conn = engine.connect()



# Load into pandas dataframe



life_exp = pd.read_sql_query(stmt, conn)



print(life_exp.info())

print(life_exp.describe())





plt.hist(life_exp["LifeExpectancy"], bins = 14)

plt.xlabel("Life Expectancy from Birth")

plt.ylabel("Count")



conn.close()
# Create list to look through

string_list = ["infant", "fertility", "population density", "GDP per capita", "Inflation", "PM2.5", "CO2 emissions"]



conn = engine.connect()



for string in string_list:

    print(string+"\n")

    stmt = find_indicator_strings(string)

    for result in conn.execute(stmt):

        print(result.IndicatorName, result.IndicatorCode)

        print(result.LongDefinition)

        print("\n")



conn.close()

    
%matplotlib inline



# Create dictionary of codes we wish to query to human readable names



code_dict = {"SP.DYN.LE00.IN" : "LifeExpectancy", 

             "SP.DYN.IMRT.IN" : "InfantMort",

             "SP.DYN.TFRT.IN" : "Fertility",

             "EN.POP.DNST" : "PopDens",

             "NY.GDP.PCAP.PP.KD" : "GDPperCap",

             "NY.GDP.DEFL.KD.ZG" : "Inflation",

             "EN.ATM.PM25.MC.M3" : "PM2.5Exp",

             "EN.ATM.CO2E.KT" : "CO2"}



# Create SQL query



stmt = sqla.select([tables["Indicators"].c.CountryName, 

                    tables["Indicators"].c.IndicatorCode.label("Indicator"),

                    tables["Indicators"].c.Value])





# Use to avoid selecting regions instead of countries

stmt = stmt.select_from(tables["Indicators"].join(tables["Country"],

                                                 tables["Country"].c.CountryCode == tables["Indicators"].c.CountryCode))



stmt = stmt.where(sqla.and_(tables["Country"].c.Region.isnot(""),

                            tables["Indicators"].c.IndicatorCode.in_(list(code_dict.keys())),

                            tables["Indicators"].c.Year == 2010)

    )



conn = engine.connect()



life_exp = pd.read_sql_query(stmt, conn)



# Change codes to readable names

life_exp["Indicator"].replace(code_dict, inplace=True)



# Change from long to wide format

life_exp = life_exp.pivot(index="CountryName", columns = "Indicator", values = "Value")



# Remove NA values

life_exp.dropna(inplace=True)



print(life_exp.info())

print(life_exp.describe())







conn.close()
%matplotlib inline



sns.heatmap(life_exp.corr(), square=True, cmap='RdYlBu')
from sklearn import model_selection

from sklearn import linear_model



# Split data into dependent and independent variables.

y = life_exp["LifeExpectancy"].values

X = life_exp.drop("LifeExpectancy", axis = 1).values



# Pick l1 ratio hyperparameter space



l1_space = np.linspace(0.01, 1, 30)



# Setup cross validation and parameter search

elastic = linear_model.ElasticNetCV(l1_ratio = l1_space,

                                    normalize = True, cv = 5)



# Create train and test sets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state = 23)



# Fit to the training set



elastic.fit(X_train, y_train)



# Check linear regression R^2



r2 = elastic.score(X_test, y_test)



print("Tuned ElasticNet Alpha {}".format(elastic.alpha_))

print("Tuned ElasticNet l1 ratio {}".format(elastic.l1_ratio_))

print("Tuned R squared {}".format(r2))
# create lasso model with cross validation



lasso = linear_model.LassoCV(normalize = True, cv = 5)



# Fit to the training set



lasso.fit(X_train, y_train)



# Find regression R^2



r2 = lasso.score(X_test, y_test)



print("Tuned Lasso Alpha {}".format(lasso.alpha_))

print("Tuned R squared {}".format(r2))
%matplotlib inline



lasso_coef = lasso.coef_



predictors = life_exp.drop("LifeExpectancy", axis = 1).columns



plt.plot(range(len(predictors)), lasso_coef)

plt.xticks(range(len(predictors)), predictors.values, rotation = 60)

plt.ylabel("Coefficient")

%matplotlib inline

# Set up ridge regression



alpha_space = np.logspace(-2, 2, 100)



ridge = linear_model.RidgeCV(alphas = alpha_space, 

                             normalize = True, cv = 5)



# Fit to the training set



ridge.fit(X_train, y_train)



# Find regression R^2



r2 = lasso.score(X_test, y_test)



print("Tuned Ridge Alpha {}".format(ridge.alpha_))

print("Tuned R squared {}".format(r2))



ridge_coef = ridge.coef_



plt.plot(range(len(predictors)), ridge_coef)

plt.xticks(range(len(predictors)), predictors.values, rotation = 60)

plt.ylabel("Coefficient")