# Importing Essential Packages

import numpy as np 

import pandas as pd 

import sqlalchemy as sqla

import matplotlib.pyplot as plt



# Connecting to SQLite Database

metadata = sqla.MetaData()

engine = sqla.create_engine("sqlite:///../input/database.sqlite")

table_names = engine.table_names()

connection = engine.connect()

tables = dict()



# Examining the tables and their respective columns

for table in table_names:

#    print("\n"+table+" columns:\n")

    tables[table] = sqla.Table(table, metadata, autoload=True, autoload_with=engine)

#    for column in tables[table].c.keys():

#        print(column)



# Designing our initial Dataframes

query = sqla.select([tables['Indicators'].c.CountryName, 

                     tables['Indicators'].c.Year,

                     tables['Indicators'].c.IndicatorName,

                     tables['Indicators'].c.IndicatorCode, 

                     tables['Indicators'].c.Value])
# DataFrame for Foreign Direct Investment Indicators

query1 = query.where(sqla.and_(sqla.or_(tables['Indicators'].c.IndicatorCode == 'BM.KLT.DINV.GD.ZS',

                                        tables['Indicators'].c.IndicatorCode == 'BN.KLT.DINV.CD',

                                        tables['Indicators'].c.IndicatorCode == 'BX.KLT.DINV.CD.WD',

                                        tables['Indicators'].c.IndicatorCode == 'BX.KLT.DINV.WD.GD.ZS',

                                        tables['Indicators'].c.IndicatorCode == 'FM.AST.NFRG.CN',

                                        tables['Indicators'].c.IndicatorCode == 'GC.FIN.FRGN.CN',

                                        tables['Indicators'].c.IndicatorCode == 'GC.FIN.FRGN.GD.ZS',

                                        ),

                               sqla.and_(tables['Indicators'].c.CountryName != 'Arab World', 

                                        tables['Indicators'].c.CountryName != 'Caribbean small states',

                                        tables['Indicators'].c.CountryName != 'Central Europe and the Baltics',

                                        tables['Indicators'].c.CountryName != 'East Asia & Pacific (all income levels)',

                                        tables['Indicators'].c.CountryName != 'East Asia & Pacific (developing only)',

                                        tables['Indicators'].c.CountryName != 'Euro area',

                                        tables['Indicators'].c.CountryName != 'Europe & Central Asia (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Europe & Central Asia (developing only)',

                                        tables['Indicators'].c.CountryName != 'European Union',

                                        tables['Indicators'].c.CountryName != 'Fragile and conflict affected situations',

                                        tables['Indicators'].c.CountryName != 'Heavily indebted poor countries (HIPC)',

                                        tables['Indicators'].c.CountryName != 'High income',

                                        tables['Indicators'].c.CountryName != 'High income: nonOECD',

                                        tables['Indicators'].c.CountryName != 'High income: OECD',

                                        tables['Indicators'].c.CountryName != 'Latin America & Caribbean (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Latin America & Caribbean (developing only)',

                                        tables['Indicators'].c.CountryName != 'Least developed countries: UN classification',

                                        tables['Indicators'].c.CountryName != 'Low & middle income',

                                        tables['Indicators'].c.CountryName != 'Low income',

                                        tables['Indicators'].c.CountryName != 'Lower middle income',

                                        tables['Indicators'].c.CountryName != 'Middle East & North Africa (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Middle East & North Africa (developing only)',

                                        tables['Indicators'].c.CountryName != 'Middle income',

                                        tables['Indicators'].c.CountryName != 'Other small states',

                                        tables['Indicators'].c.CountryName != 'Pacific island small states',

                                        tables['Indicators'].c.CountryName != 'Small states',

                                        tables['Indicators'].c.CountryName != 'South Asia',

                                        tables['Indicators'].c.CountryName != 'Sub-Saharan Africa (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Sub-Saharan Africa (developing only)' ,

                                        tables['Indicators'].c.CountryName != 'Upper middle income' ,

                                        tables['Indicators'].c.CountryName != 'World',

                                        tables['Indicators'].c.CountryName != 'North America',

                                        tables['Indicators'].c.CountryName != 'OECD members'),

                               tables['Indicators'].c.Year >= 2003,

                               tables['Indicators'].c.Year <= 2013))

foreign_direct_investment = connection.execute(query1).fetchall()

foreign_direct_investment = pd.DataFrame(foreign_direct_investment, columns = ['Country','Year','Indicator','Code','Value'])

print(type(foreign_direct_investment))

print(foreign_direct_investment.head(2))



#Export the DataFrame into an Excel File here

writerFDI = pd.ExcelWriter('Foreign_Direct_Investment.xlsx')

foreign_direct_investment.to_excel(writerFDI,'Sheet1')

writerFDI.save()
# DataFrame for Poverty Indicators

poverty = query.where(sqla.and_(sqla.or_(tables['Indicators'].c.IndicatorCode == "SI.POV.2DAY", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.DDAY", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.GAP2", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.GAPS", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.GINI", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.NAGP", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.NAHC", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.RUGP", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.RUHC", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.URGP", 

                                        tables['Indicators'].c.IndicatorCode == "SI.POV.URHC"

                                        ),

                                 sqla.and_(tables['Indicators'].c.CountryName != 'Arab World', 

                                        tables['Indicators'].c.CountryName != 'Caribbean small states',

                                        tables['Indicators'].c.CountryName != 'Central Europe and the Baltics',

                                        tables['Indicators'].c.CountryName != 'East Asia & Pacific (all income levels)',

                                        tables['Indicators'].c.CountryName != 'East Asia & Pacific (developing only)',

                                        tables['Indicators'].c.CountryName != 'Euro area',

                                        tables['Indicators'].c.CountryName != 'Europe & Central Asia (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Europe & Central Asia (developing only)',

                                        tables['Indicators'].c.CountryName != 'European Union',

                                        tables['Indicators'].c.CountryName != 'Fragile and conflict affected situations',

                                        tables['Indicators'].c.CountryName != 'Heavily indebted poor countries (HIPC)',

                                        tables['Indicators'].c.CountryName != 'High income',

                                        tables['Indicators'].c.CountryName != 'High income: nonOECD',

                                        tables['Indicators'].c.CountryName != 'High income: OECD',

                                        tables['Indicators'].c.CountryName != 'Latin America & Caribbean (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Latin America & Caribbean (developing only)',

                                        tables['Indicators'].c.CountryName != 'Least developed countries: UN classification',

                                        tables['Indicators'].c.CountryName != 'Low & middle income',

                                        tables['Indicators'].c.CountryName != 'Low income',

                                        tables['Indicators'].c.CountryName != 'Lower middle income',

                                        tables['Indicators'].c.CountryName != 'Middle East & North Africa (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Middle East & North Africa (developing only)',

                                        tables['Indicators'].c.CountryName != 'Middle income',

                                        tables['Indicators'].c.CountryName != 'Other small states',

                                        tables['Indicators'].c.CountryName != 'Pacific island small states',

                                        tables['Indicators'].c.CountryName != 'Small states',

                                        tables['Indicators'].c.CountryName != 'South Asia',

                                        tables['Indicators'].c.CountryName != 'Sub-Saharan Africa (all income levels)',

                                        tables['Indicators'].c.CountryName != 'Sub-Saharan Africa (developing only)' ,

                                        tables['Indicators'].c.CountryName != 'Upper middle income' ,

                                        tables['Indicators'].c.CountryName != 'World',

                                        tables['Indicators'].c.CountryName != 'North America',

                                        tables['Indicators'].c.CountryName != 'OECD members'),

                                tables['Indicators'].c.Year >= 2003,

                                tables['Indicators'].c.Year <= 2013))

poverty = connection.execute(poverty).fetchall()

poverty = pd.DataFrame(poverty, columns = ['Country','Year','Indicator','Code','Value'])

print(poverty.head(2))



#Export the DataFrame into an Excel File here

writerPoverty = pd.ExcelWriter('Poverty.xlsx')

poverty.to_excel(writerPoverty,'Sheet1')

writerPoverty.save()
#Concatenate Foreign Direct Investment and Poverty

df = pd.concat([foreign_direct_investment, poverty], axis=0)
# Removing Unecessary Column

df.drop(['Indicator'], axis=1, inplace=True)

# print(df.head(5))



# Year column as Integer

df['Year'].apply(int)



# Pivoting

df_pivot = df.pivot_table(index=['Country', 'Year'], columns= 'Code', values = 'Value', aggfunc=sum)



# writer = pd.ExcelWriter('FDIPovPivot.xlsx')

# df_pivot.to_excel(writer,'Sheet1')

# writer.save()
# Selecting only countries with 30% or less misssing GINI data

df_GINI = pd.DataFrame(df_pivot.groupby('Country')['SI.POV.GINI'].count())

# print(df_GINI.head(10))

# print(df_GINI.index)

# Exporting count of entries

# writer = pd.ExcelWriter('GINI counted.xlsx')

# df_GINI10.to_excel(writer,'Sheet1')

# writer.save()



# Dropping countries with only 6 or less entries

GINI_qual = df_GINI[df_GINI.loc[:,'SI.POV.GINI'] >= 7 ]

# print(type(GINI_qual))

# print(GINI_qual.head(10))

GINI_list = list(GINI_qual.index.values)



# Final DataFrame only with countries with enough GINI data

df_FINAL = df_pivot[df_pivot.index.get_level_values('Country').isin(GINI_list)]

df_FINAL = df_FINAL.loc[:,['BX.KLT.DINV.CD.WD','SI.POV.GINI']]

df_FINAL.reset_index(inplace=True)

df_FINAL.columns = ['Country','Year','FDI','GINI']

df_FINAL.FDI = df_FINAL.FDI.astype('float')

df_FINAL.GINI = df_FINAL.GINI.astype('float')

print(df_FINAL.head(2))
# Handling Missing Values: Mean Strategy Imputation by Country over 11 years  

from sklearn.preprocessing import Imputer



# Shaping the dataset for imputation

FDImput = df_FINAL.pivot(index='Year',columns='Country',values='FDI')

GINImput = df_FINAL.pivot(index='Year', columns='Country', values = 'GINI')

print(GINImput)



# Imputation

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputerGINI = imputer.fit(GINImput)

GINImputed = imputerGINI.transform(GINImput)

GINImput = pd.DataFrame(GINImputed, columns = GINImput.columns, index=GINImput.index)

GINImput = GINImput.reset_index()

imputerFDI = imputer.fit(FDImput)

FDImputed = imputerFDI.transform(FDImput)

FDImput = pd.DataFrame(FDImputed, columns = FDImput.columns, index=FDImput.index)

FDImput = FDImput.reset_index()

print(GINImput)



# Putting the DataFrame back together

GINImelt = pd.melt(GINImput, id_vars=['Year'],var_name='Country', value_name='GINI')

FDImelt = pd.melt(FDImput, id_vars=['Year'],var_name='Country',value_name='FDI')

print(GINImelt.shape)

print(GINImelt.head(11))

print(FDImelt.shape)

print(FDImelt.head(11))

df_FINAL = pd.merge(FDImelt,GINImelt, on=['Country','Year'])

df_FINAL = df_FINAL[['Country','Year','FDI','GINI']]

print(df_FINAL.shape)

print(df_FINAL.head(11))
# Exploratory Data Analysis



# getting coordinates for countries

# from geopy.geocoders import Nominatim



# Building a map of GINI coefficients

# from bokeh.models import Plot

from bokeh.plotting import figure



# df_map = figure()

# plot = Plot()

# plot.add_glyph()

# plot.add_layout()

# Slider

# plot.add_tools()

# Bar chart with Slider for years

from bokeh.charts import Bar

from bokeh.plotting import output_file, show

from bokeh.models import ColumnDataSource, Slider

from bokeh.layouts import widgetbox, row

import seaborn as sns



# Only 1 year per plot, default year: 2003

year = 2003

df_YEAR = df_FINAL[df_FINAL.Year == year]

df_YEAR = df_YEAR.groupby('Country')['Year','GINI'].sum()

df_YEAR.reset_index(inplace=True)

df_YEAR = df_YEAR.sort_values(['GINI'])

# print(df_YEAR.head(2))



# Bokeh version

#source = ColumnDataSource(data={

#    'FDI' : df_YEAR.loc[2003].FDI,

#    'GINI' : df_YEAR.loc[2003].GINI,

#    'Country' : df_YEAR.loc[2003].Country

#})

#bar = Bar(source, 'Country', 'GINI', title = "GINI for every country per year")

#output_file('bar.html')

#show(bar)



# Slider widget

# Callback

#def update_plot(attr,old,new):

#    yr = slider.value

    



# Seaborn version

fig, ax = plt.subplots()

fig.set_size_inches(11.7,13.27)

sns.set(color_codes=True)

barr = sns.barplot('GINI','Country', data = df_YEAR, ax=ax,)

plt.show()
# Violin plot to show GINI distribution over the years

GINIlin = sns.violinplot(x="Year", y="GINI", data= df_FINAL,

                         palette="Set2", split=True, scale="count", inner="points",

                       )
# Violin plot to show Foreign Direct Investment distribution over the years

FDIlin = sns.violinplot(x="Year", y="FDI", data= df_FINAL,

                         palette="Set2", split=True, scale="count", inner="points",

                       )
# Line graph with two lines per country: FDI and GINI over the years

# Include a drop down widget for choosing country

# plot = figure(x_axis_label=,y_axis_label=)

# plot.line()

# plot.line()

# plot.circle()
# Feature Scaling

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

sc_X = StandardScaler()

# df_FINAL[['FDI']] = sc_X.fit_transform(df_FINAL[['FDI']])

df_FINAL[['GINI']] = sc_X.fit_transform(df_FINAL[['GINI']])

print(df_FINAL.head(2))



x = df_FINAL[['FDI']]

y = df_FINAL[['GINI']]



regressor = LinearRegression()

regressor.fit(x, y)

plt.scatter(x,y,color='red')

plt.plot(x, regressor.predict(x))

plt.title('Foreign Direct Investment vs. GINI coefficient (Training Set)')

plt.xlabel('Foreign Direct Investment, Inflow')

plt.ylabel('GINI Index')



# Splitting the Data into the Training Set and the Test Set

#from sklearn.cross_validation import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



# Machine Learning Model: Linear Regression

# Fitting Simple Linear Regression to the Training Set

regressor.fit(x_train, y_train)



# Reshape?

# .reshape()



#Predicting the test set results

y_pred = regressor.predict(x_test)



# Visualizing the training test results with matplotplib.pyplot

plt.scatter(x_train, y_train, color='red')

plt.plot(x_train, regressor.predict(x_train))

plt.title('Foreign Direct Investment vs. GINI coefficient (Training Set)')

plt.xlabel('Foreign Direct Investment, Inflow')

plt.ylabel('GINI Index')

plt.show()
