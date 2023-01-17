#1 - Import libraries
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#2 - Load file into dataframe
df = pd.read_csv("../input/nyc-rolling-sales.csv")

#Overview of data
#Find no. of rows & cols
print(df.shape, "\n")
#Show col names and datatypes
print(df.dtypes, "\n")
#Show no. of None values
print(df.isna().sum(), "\n")

#3 - Write functions for data exploration

#Function for plotting histograms:
def PlotHist(col, color='b'):
    col.plot.hist(grid=True, bins=12, rwidth=0.9,
                      color=color)
    plt.title('Value distribution: ' + col.name)
    plt.xlabel(col.name)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show();
    
#Calculate step for y-ticks
def CalcMax(max_val):
    mag = len(str(round(max_val))) - 1
    #power of ten - pot
    PoT = 10 ** mag 
    rounded = math.ceil(max_val/PoT) * PoT
    return rounded;

#Function for selecting categories
def AssignCat(col, num_cats=5):
    if len(col.value_counts().keys()) < num_cats:
        num_cats = len(col.value_counts().keys())
    cats =  pd.DataFrame(col.value_counts().nlargest(num_cats - 1))
    cats['row_num'] = [str(x+1) for x in range(len(cats))]
    cats['name'] = cats.index.values
    col = pd.DataFrame({'val': col.values})
    col = pd.merge(col, cats, how='left', left_on='val', right_on='name')
    col['new'] = col['row_num'].fillna(str(num_cats))
    return col['new'].values;
 
#Function for plotting bar charts:
def PlotBar(col, color='b'):
    means = {}
    for i in col.unique():
        means.update({i : np.exp(df['Price'][df[col.name]== i]).mean()})   
    plt.bar(range(len(means)), list(means.values()), align='center', \
            color=color)
    plt.xticks(range(len(means)), sorted(list(means.keys())))
    #Get largest value
    maximum = CalcMax(max(means.values()))
    s = maximum / 5
    y_labels = ["{:,}".format(x) for x in np.arange(0,maximum, step=s)]
    plt.yticks(np.arange(0,maximum, step=s),y_labels)
    plt.title('Avg. Price per ' + col.name)
    plt.xlabel(col.name)
    plt.ylabel('Price')
    plt.show(); 
    
#Function for creating scatter plots:
def PlotScatter(col, show='none'):
    plt.title('Price - ' + col.name)
    plt.xlabel(col.name)
    plt.ylabel('Price log')
    if show == 'mlp':
        plt.scatter(col,(df['Price']))
        plt.show()
    elif show == 'sns': 
        sns.regplot(x=col.name, y='Price', data=df, x_jitter=.5)
        plt.show()
    else: print(col.name +': No plot shown to reduce processing time');

#4 - Analyse dependent variable

#Check for unusual values
df['Price'] = sorted(df['SALE PRICE'])
#print(df.Price.head(10))
#Some values which represent missing values: '-'
#Convert to int
df['Price'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')
#remove Nan values
df = df[df['Price'].isnull()==0]
df['Price'] = df['Price'].astype(int)
# Show distribution before transformation
PlotHist(df['Price'],'c')

#Remove records with price = 0 - Anomalies
df = df[df['Price'] != 0]

df['Price'] = df['Price'].apply(np.log)

print(df['SALE PRICE'].describe(), "\n")
price_avg = df['Price'].mean()
print('Average Price (log): ' + "{:,}".format(round(price_avg)), "\n")
price_stdev = np.std(df['Price'])
print('Std. Dev. of Price (log): ' + "{:,}".format(round(price_stdev)), "\n")

#Remove outliers: 3 stdevs from mean
df =  df[df['Price'] >= price_avg - price_stdev * 3]
df =  df[df['Price'] <= price_avg + price_stdev * 3]

#Plot again:
PlotHist(df['Price'], 'c')
#5 - Explore independent variables

                                #1. Borough
print('BOROUGH - Value Counts')
print(df.BOROUGH.value_counts(), "\n")
PlotBar(df['BOROUGH'],'g')

#Convert to dummies
borough = pd.get_dummies(df['BOROUGH'], prefix='borough')
df = pd.concat([df, borough], axis=1)

# May use this feature in multiple regression due to significant price difference

                            #2. Neighborhood
#print('NEIGHBORHOOD - Value Counts')
#print(df.NEIGHBORHOOD.value_counts(), "\n")
# Probably not useful: too many distinct, non-numeric values

                        #3. Building Class Category
#Since there are too many categories, we only take the first 4 categories 
#and group the rest into one
df['building_category'] = AssignCat(df['BUILDING CLASS CATEGORY'])

PlotBar(df['building_category'],'r')

#Convert to dummies
build_cat = pd.get_dummies(df['building_category'], prefix='bc')
df = pd.concat([df, build_cat], axis=1)

                        #4.A Tax Class Category
print('Tax Class - Value Counts')
print(df['TAX CLASS AT PRESENT'].value_counts(), "\n")
#Only use 3 most common classes
df['tax_class'] = AssignCat(df['TAX CLASS AT PRESENT'])

PlotBar(df['tax_class'],'gray')

#Convert to dummies
tax_class = pd.get_dummies(df['tax_class'], prefix='tc')
df = pd.concat([df, tax_class], axis=1)

                        #4.B Tax class at time of sale
print('Tax Class at Sale - Value Counts')
print(df['TAX CLASS AT TIME OF SALE'].value_counts(), "\n")
#Only use 3 most common classes
df['tc_at_sale'] = AssignCat(df['TAX CLASS AT TIME OF SALE'])
  

PlotBar(df['tax_class'],'black')

tc_present = pd.get_dummies(df['tc_at_sale'], prefix='tcs')
df = pd.concat([df, tc_present], axis=1)

#Different systems - unclear which is better

                            #5. Block
#print('Block - Value Counts')
#print(df['BLOCK'].value_counts(), "\n")
#Too many distinct values, no category with significant % of counts

                            #6. Lot
#print('Lot - Value Counts')
#print(df['LOT'].value_counts(), "\n")
#Too many distinct values, no category with significant % of counts

                        #7. Ease-ment
print('Ease-ment - Value Counts')
print(df['EASE-MENT'].value_counts(), "\n")
#Same as ID - no use

                #8A - Building class at present
#print('Building Class Present - Value Counts')
#print(df['BUILDING CLASS AT PRESENT'].value_counts(), "\n")
#Only use 3 most common classes
df['bcp'] = AssignCat(df['BUILDING CLASS AT PRESENT'])

PlotBar(df['bcp'],color='y')

                    #8B - Building class at time of sale


df['bcs'] = AssignCat(df['BUILDING CLASS AT TIME OF SALE'])

PlotBar(df['bcs'],color='coral')

#Almost the same as Building Class at Present - use at Sale which is more meaningful

bcs = pd.get_dummies(df['bcs'], prefix='bcs')
df = pd.concat([df, bcs], axis=1)

                        #9. Address
#Too specifig - unlikely to be useful

                    #10. Apartment no.
#print('Apartment No. - Value Counts')
#print(df['APARTMENT NUMBER'].value_counts(), "\n")
#Too many different values 

                    #11. Zip code
#print('ZIP Code - Value Counts')
#print(df['ZIP CODE'].value_counts(), "\n")
#Too many different values 


                    #12. Residential units
#print('RESIDENTIAL UNITS - Value Counts')
#print(df['RESIDENTIAL UNITS'].value_counts(), "\n")
df['res_unit_log'] = df['RESIDENTIAL UNITS'].apply(lambda x: 0 if x == 0 \
                                              else np.log(x)) 
PlotScatter(df['res_unit_log'])

                        #13. Commercial units
df['com_unit_log'] = df['COMMERCIAL UNITS'].apply(lambda x: 0 if x == 0 \
                                              else np.log(x)) 
PlotScatter(df['com_unit_log'])

                        #14. Total units
df['tot_unit_log'] = df['TOTAL UNITS'].apply(lambda x: 0 if x == 0 \
                                              else np.log(x)) 
PlotScatter(df['tot_unit_log'])

                        #15. Land square ft
#Impute mean for missing and values
df['land'] =  pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')
#use median for 0 and nulls
land_median = df['land'][df['land']!=0].median()
df['land'] = df['land'].fillna(land_median)
df['land'] = df['land'].apply(lambda x: x if x > 0 \
                                          else land_median)
df['land_log'] = np.log(df['land'])
PlotScatter(df['land_log'])

                            #16. Gross square ft
df['gross_sqft'] =  pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')
#use median for 0 and nulls
gross_sqft_median = df['gross_sqft'][df['gross_sqft']!=0].median()
df['gross_sqft'] = df['gross_sqft'].fillna(gross_sqft_median)
df['gross_sqft'] = df['gross_sqft'].apply(lambda x: x if x > 0 \
                                          else gross_sqft_median)
df['gross_sqft_log'] = np.log(df['gross_sqft'])
PlotScatter(df['gross_sqft'])

                            #17. Year built
df['year'] = df['YEAR BUILT']
#Impute median where year = 0
year_median = df['year'][df['year'] != 0].median()
#Date cannot before 1764
df['year'] = df['year'].apply(lambda x: x if x >= 1764 else year_median )
PlotHist(df['year'], 'orange')
PlotScatter(df['year'])

#Unclear if relationship between year and price


                        #18. Sale date
df['date'] = pd.to_datetime(df['SALE DATE'], format = '%Y-%m-%d %H:%M:%S') 

sale_dates =  sorted(set(df['date']))

date_means = []
for i in sale_dates:
    date_means.append(np.exp(df['Price'][df['date']== i]).mean())
    
plt.plot(sale_dates,date_means)
plt.title('Price - Sale Date')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
#Apparently no clear relationship

#6 - Split dataset into train and test

#potential features
X= df[['borough_1','borough_2','borough_3','borough_4','borough_5', \
       'bc_1','bc_2','bc_3','bc_4','bc_5','tc_1','tc_3','tc_4',\
       'tcs_3','tcs_1','tcs_2','bcs_5','bcs_1', \
       'bcs_3','bcs_4','res_unit_log','com_unit_log','tot_unit_log',\
       'land_log','gross_sqft_log','year']]
Y = df['Price']
X_train, X_test, Y_train, Y_test = \
   train_test_split(X, Y, test_size=0.3, random_state=False)
#7 - Build regression model

Train = {}
Test = {}
model_no = 1

def RunRegression(model_number, model):
    print('Model no. ' + str([model_number]), "\n" )
    Train[model_number] = X_train[model]
    Test[model_number]= X_test[model]                        
    reg = LinearRegression().fit(Train[model_number], Y_train)
    score = reg.score(Test[model_number], Y_test)
    print('Model using variables: '+ str(list(Train[model_number])))
    print('R^2 Score: ' + str(score), "\n" )
    coefficients = reg.coef_
    print('Coefficients: ' +str(coefficients), "\n" )  
    global model_no
    model_no = model_number + 1
#8 - Test and evaluate different models

#First model: only BOROUGH  
model =  ['borough_1','borough_2','borough_3','borough_4','borough_5']

RunRegression(model_no, model)

#Second model: BOROUGH and land  
model = ['borough_1','borough_2','borough_3','borough_4','borough_5','land_log']

RunRegression(model_no, model)
#Total R^2 increased

#Third model: BOROUGH and Gross_sqft  
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log']
RunRegression(model_no, model)
#Total R^2 increased compared to land

#Fourth model: BOROUGH, Gross_sqft, Res_Unit 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','res_unit_log']
RunRegression(model_no, model)
#Total R^2 not significantly increased

#Fifth model: BOROUGH, Gross_sqft, Com_Unit 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','com_unit_log']
RunRegression(model_no, model)
#Total R^2 reduced

#Sixth model: BOROUGH, Gross_sqft, Tot_Unit 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','tot_unit_log']
RunRegression(model_no, model)

#Total R^2 higher than res_units or com_units

#Seventh model: BOROUGH, Gross_sqft, Tot_Unit, TaxClass 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','tot_unit_log','tc_3','tc_1','tc_3','tc_4']
RunRegression(model_no, model)
#Total R^2 slightly increased

#Next model: BOROUGH, Gross_sqft, Tot_Unit, TaxClass at Sale 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','tot_unit_log','tcs_3','tcs_1','tcs_2']
RunRegression(model_no, model)
#Total R^2 slightly lower

#Next model: BOROUGH, Gross_sqft, Tot_Unit, TaxClass present, TaxClass at Sale 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','tot_unit_log','tc_1','tc_3','tc_4',\
         'tcs_3','tcs_1','tcs_2']
RunRegression(model_no, model)
#Total R^2 only slightly higher than without tax class at sale

"""Next model: BOROUGH, Gross_sqft, Tot_Unit, TaxClass present, 
BuildingCat at Sale""" 
model = ['borough_1','borough_2','borough_3','borough_4','borough_5',\
         'gross_sqft_log','tot_unit_log','tc_1','tc_3','tc_4',\
         'bc_1','bc_2','bc_3','bc_4','bc_5']
RunRegression(model_no, model)
#Total R^2 slightly increased