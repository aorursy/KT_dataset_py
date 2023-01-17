
import pandas as panda

remote_location = '../input/auto_sales.csv'

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


## we had a look at the data placed in the remote location and found that the data is without any headers. 
## hence we pass the arguments, headers = None
data = panda.read_csv(remote_location, header = None) 

data_dimensions = data.shape
print("The dimensions of data downloaded is %d rows * %d columns" %(data_dimensions[0], data_dimensions[1] ))


##dropping the first row and first columns which came automatically once i uploaded non header csv in kaggle
data.drop(labels=[0], inplace=True)
data.drop(labels=[0], inplace=True, axis=1)
data.shape

## we are going to assign headers to the data downloaded.
## we are also going to rename the headers all to lowercase for standardization purposes (this is an optional step)
## and then we are going to print the first 10 rows and last ten rows for simply eyeballing part of the data

data.columns = [ i.lower() for i in headers]

print("************ Top ten rows ************")
print(data.head(10))



print("************ Bottom ten rows ************")
print(data.tail(10))

## we will check for the data types in the data provided and give a count of the unique data types provided

data.info() ## for simple eyeballing of the data provided. gives us overview of datatypes across columns
data_types = data.dtypes ##data_types returned is an instance of pandas.DataSeries class

## value_counts gives us a result of number of unique data types in the entire data. value_counts is an attribute only present in DataSeries and not in DataFrame

print(data_types.value_counts())


## a longer way to achieve the above is given below, where we convert it to data frame and reset indices and run a groupby and count command
## FYI - data.info() would have also given DataFrame instance automatically.
# data_types = panda.DataFrame(data_types, columns = ['column_type'])
# data_types.index.name = 'column_name'

# data_types.reset_index(inplace = True)
# print(data_types.groupby( by = 'column_type', as_index = False).count())




## we will call describe method of pandas data frame to show us data statistics for numeric columns.

## the below two lines could have been combined using include =áll keyword, but i prefer separation
print(data.describe())

print(data.describe(include = 'object'))

import numpy
data.replace('?', numpy.nan, inplace = True) ## inplace marked as true , hence we are modifying original data

temp_null_data_check = data.isnull()
empty_column_list = {}

for col in temp_null_data_check.columns.tolist():
    values = temp_null_data_check[col].values
    
    if True in values :
        empty_column_list[col] = {'missing_count' : list(values).count(True), 'data_type' : data[col].dtype.name} ##dtype is numpy.dtype instance, value is numpy.darray instance
        

print(empty_column_list)


## another way to achieve the above is give below.
# empty_values = data.isnull().sum().to_frame()
# empty_values= empty_values.assign(column_type = data.dtypes)
# empty_values

## we can drop rows where all the values are NaN. THat is empty rows by using the command data.dropna(how='all')

## drop row with empty price cell
 
data.dropna(inplace = True, axis =0, subset = ['price'])

data['price'] = data['price'].astype('float64')

## check to see if the row has been dropped
numpy.nan in data['price'].values

## convert data types for normalized losses, horsepower, bore, stroke, peak -rpm  since we are going to perform mathematical operation

data[['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']] = data[['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']].astype('float64')

## calculate mean of normalized losses, horsepower, bore, stroke, peak -rpm

normalized_losses_mean = data['normalized-losses'].mean()
bore_mean              = data['bore'].mean()
stroke_mean            = data['stroke'].mean()
horsepower_mean        = data['horsepower'].mean()
peak_rpm_mean          = data['peak-rpm'].mean()

## assign missing values to the individual mean value


data['normalized-losses'].replace(to_replace = numpy.nan, value = normalized_losses_mean, inplace =True)
data['bore'].replace(to_replace = numpy.nan, value = bore_mean, inplace =True)
data['stroke'].replace(to_replace = numpy.nan, value = stroke_mean, inplace =True)
data['horsepower'].replace(to_replace = numpy.nan, value = horsepower_mean, inplace =True)
data['peak-rpm'].replace(to_replace = numpy.nan, value = peak_rpm_mean, inplace =True)

## sklearn.impute.SimpleImputer is in latest version and not available yet.
## code in case of SimpleIMputer would have been SimpleImputer(missing = numpy.nan, strategy = 'mean'). and then fit_trasnform

# from sklearn.preprocessing import Imputer

# imputer = Imputer(missing_values = numpy.nan, strategy = 'mean', axis = 0)
# data = imputer.fit_transform(data.values)

print("peak rpm summation after mean calc %s" %data['peak-rpm'].sum())
print("horsepower summation after mean calc %s" %data['horsepower'].sum())
print("stroke summation after mean calc %s" %data['stroke'].sum())
print("bore summation after mean calc %s" %data['bore'].sum())
print("normalized-losses summation after mean calc %s" %data['normalized-losses'].sum())
## we can use the sklearn.impute.SimpleImputer with strategy mean to transform all the above 4 values at one shot

## calculate top value for number of doors and replace empty value with it

number_of_doors_top_value = data['num-of-doors'].value_counts().idxmax()

data['num-of-doors'].replace(to_replace = numpy.nan, value = number_of_doors_top_value, inplace =True)


## run the empty values check again


temp_null_data_check = data.isnull()
empty_column_list = {}

for col in temp_null_data_check.columns.tolist():
    values = temp_null_data_check[col].values
    
    if True in values :
        empty_column_list[col] = {'missing_count' : list(values).count(True), 'data_type' : data[col].dtype.name} ##dtype is numpy.dtype instance, value is numpy.darray instance
        

if not empty_column_list:
    print("Successfully completed all empty values replacement")
# print(empty_column_list) ## empty dict means our empty values replacement is successfully completed




###lets draw a histogram for showing data distribution
%matplotlib inline
import matplotlib.pyplot as plot
import seaborn as sns

plot.hist(data['horsepower'], bins = 3)
plot.title('Horsepower Distribution')
plot.xlabel("Horsepower values")
plot.ylabel("Count")

plot.show()

sns.distplot(data['horsepower'], bins =3, kde=False, rug=True)



## get the range of values for horsepower, the min and the max

min_horse_power = data['horsepower'].min()
max_horse_power = data['horsepower'].max()

## set the number of bins you require and provide names for the same

bin_labels = ['low', 'medium', 'high']
bin_count = 4  ## cut function has default rightmost edge as true.

## lets say we knew which numbers to classify as low, medium and high. say from a given range of [0,20]  we 
## say numbers within 0 - 7 is low, 7 - 14 is medium and 14 - 20 is high. so our bins are essentially [0,7,14,20] i.e one higher
## than the number of labels. 

## we dont want to decide the ranges for bins . we let numpy decide the same
## using linspace, which returns evenly spaced numbers using intervals

bins = numpy.linspace(start = min_horse_power, stop = max_horse_power, num = 4)
# print(bins,min_horse_power,max_horse_power)

## convert data into bins

data['horsepower-binned'] = panda.cut(data['horsepower'], bins, labels = bin_labels ,include_lowest = True )

# print(data[['horsepower','horsepower-binned']].head(20))
print(data['horsepower-binned'].value_counts())



## another way of achieving the below is to use the panda.get_dummies method
## another technique is to use the map function. v = {'gas' :1, 'diesel' : 0}. data[].map(V)

data['fuel-type'] = data['fuel-type'].apply(lambda x: 1 if x == 'gas' else 0 )
data['fuel-type'] = data['fuel-type'].astype('int64')


data['fuel-type'].value_counts()

data['aspiration'] = data['aspiration'].apply(lambda x: 1 if x == 'std' else 0 )
data['aspiration'] = data['aspiration'].astype('int64')


data['aspiration'].value_counts()


## describe the numeric columns which we think are likely to impact price target 
## (we would perform the same for categorical values later). we want to confirm our assumption

print(data[['symboling', 'normalized-losses' , 'length', 'width', 'height', 'curb-weight', 'engine-size']].describe(include = 'all'))

data[['symboling', 'normalized-losses' , 'length', 'width', 'height', 'curb-weight', 'engine-size']].corr()


## plotting boxplots for the above 7 attributes to check for outliers and inter quartile ranges

%matplotlib inline

data[['symboling', 'normalized-losses' , 'length', 'width', 'height', 'curb-weight', 'engine-size']].plot(kind= 'box', figsize =(10,20))
plot.show()

from scipy import stats
from scipy.stats import f_oneway
import matplotlib.patches as mpatches


def calculateCorrelationCoefficientsAndpValues(x_data, y_data, xlabel):
    
    pearson_coef, p_value = stats.pearsonr(x_data, y_data)
    print("The Pearson Correlation Coefficient for %s is %s with a P-value of P = %s" %(xlabel,pearson_coef, p_value))
    
    return (pearson_coef,p_value)

def plotRegressionBetweenTwoVariables(x_label,y_label, x_y_data, pearson_coef, p_value):
    
    plot.figure(figsize=(15,15))
    
    sns.regplot(x = x_label , y = y_label , data = x_y_data)


    # plot.text(x = 1, y = 40000 , s ="Pearson Correlation Coefficient = %s"%pearson_coef, fontsize = 12 )
    # plot.text(x = 1, y = 38000 , s ="P value = %s"%p_value, fontsize = 12 )

    blue_patch = mpatches.Patch(color='blue', label='Pearson Correlation Coefficient = %s, p value is %s '%(pearson_coef, p_value))
    plot.legend(handles=[blue_patch], loc ='best')
    plot.title("Regression Plot %s vs %s"%(x_label, y_label))
    
    

    
from scipy import stats

## checking correlation between symbolizing and target price
data['symboling'] = data['symboling'].astype('float64')

coeff_values = calculateCorrelationCoefficientsAndpValues(data['symboling'], data['price'],'symboling')
plotRegressionBetweenTwoVariables( 'symboling', 'price', data[['symboling', 'price']], coeff_values[0], coeff_values[1])

## we will use the min max technique for normalization of length, width and height
## min-max technique is x-min/max-min. values range betoiwwen 0 and 1
## other technique is z score which is x - mean/std deviation. values range betwwen -inf to inf (however typical is 0 to -3 to +3)
## the otehr is feature scaling which is x/ max . values range between 0 and 1



data['length'] = (data['length'] - data['length'].min())/ (data['length'].max() - data['length'].min())

data['width'] = (data['width'] - data['width'].min())/ (data['width'].max() - data['width'].min())

data['height'] = (data['height'] - data['height'].min())/ (data['height'].max() - data['height'].min())
## drawing plots and calculation of coefficients of each

coeff_values = calculateCorrelationCoefficientsAndpValues(data['length'], data['price'],'length')
plotRegressionBetweenTwoVariables( 'length', 'price', data[['length', 'price']], coeff_values[0], coeff_values[1])







coeff_values = calculateCorrelationCoefficientsAndpValues(data['width'], data['price'],'width')
plotRegressionBetweenTwoVariables( 'width', 'price', data[['width', 'price']], coeff_values[0], coeff_values[1])



coeff_values = calculateCorrelationCoefficientsAndpValues(data['height'], data['price'],'height')
plotRegressionBetweenTwoVariables( 'height', 'price', data[['height', 'price']], coeff_values[0], coeff_values[1])


for item in data.describe().columns.tolist():
    calculateCorrelationCoefficientsAndpValues(data[item], data['price'],item)




## lets deal with the rest of the object or caategorical variables and see if it needs to be included or discarded


data.describe(include = 'object')


## lets draw a bar plot showing relationship between drive-wheels and avegrage price across drive wheels attribute
from scipy.stats import f_oneway

def calculateANOVA(data, xlabel, ylabel):

    f_val, p_val = stats.f_oneway(*data)  
 
    print( "ANOVA results for %s vs %s : F= %s , P = %s " %(xlabel,ylabel,f_val,p_val))
    
    return f_val,p_val


def plotBarChartAcrossCategories(data, xlabel, ylabel, title, anova_results = None):        

    data.plot(kind='bar', figsize= (10,10))

    plot.title(title)
    plot.xlabel(xlabel)
    plot.ylabel(ylabel)
    
    if anova_results:
        
        label = 'F value = %s, p value is %s ' %(anova_results[0], anova_results[1])
        patch = mpatches.Patch(color='orange', label=label)
        plot.legend(handles=[patch], loc ='best')


def prepareANOVAData(data, indices):
    anova_data = []

    for item in indices:
        anova_data.append(data.get_group(item)['price'])
    
    return anova_data
    




grouped_by_fuel_system =  data[['fuel-system','price']].groupby(['fuel-system'], as_index = False) 
mean_price_grouped_by_fuel_system = grouped_by_fuel_system.mean()
mean_price_grouped_by_fuel_system.set_index('fuel-system', inplace= True)
indices = list(mean_price_grouped_by_fuel_system.index)


anova_results = calculateANOVA(data = prepareANOVAData(grouped_by_fuel_system, indices), xlabel='fuel-system', ylabel='price')
plotBarChartAcrossCategories(mean_price_grouped_by_fuel_system, xlabel = 'Fuel System', \
                                 ylabel = 'Average Price', title = 'Fuel System vs Average Price', anova_results = anova_results)



grouped_by_engine_size =  data[['engine-size','price']].groupby(['engine-size'], as_index = False) 
mean_price_grouped_by_engine_size = grouped_by_engine_size.mean()
mean_price_grouped_by_engine_size.set_index('engine-size', inplace= True)
indices = list(mean_price_grouped_by_engine_size.index)


anova_results = calculateANOVA(data = prepareANOVAData(grouped_by_engine_size, indices), xlabel='engine-size', ylabel='price')
plotBarChartAcrossCategories(mean_price_grouped_by_engine_size, xlabel = 'Engine Size', \
                                 ylabel = 'Average Price', title = 'Engine Size vs Average Price', anova_results = anova_results)




grouped_by_body_style =  data[['body-style','price']].groupby(['body-style'], as_index = False) 
mean_price_grouped_by_body_style = grouped_by_body_style.mean()
mean_price_grouped_by_body_style.set_index('body-style', inplace= True)
indices = list(mean_price_grouped_by_body_style.index)


anova_results = calculateANOVA(data = prepareANOVAData(grouped_by_body_style, indices), xlabel='body-style', ylabel='price')
plotBarChartAcrossCategories(mean_price_grouped_by_body_style, xlabel = 'Body Style', \
                                 ylabel = 'Average Price', title = 'Body Style vs Average Price', anova_results = anova_results)



grouped_by_drive_wheels =  data[['drive-wheels','price']].groupby(['drive-wheels'], as_index = False) 
mean_price_groupedby_drive_wheels = grouped_by_drive_wheels.mean()
mean_price_groupedby_drive_wheels.set_index('drive-wheels', inplace= True)
indices = list(mean_price_groupedby_drive_wheels.index)


anova_results = calculateANOVA(data = prepareANOVAData(grouped_by_drive_wheels, indices), xlabel='drive-wheels', ylabel='price')
plotBarChartAcrossCategories(mean_price_groupedby_drive_wheels, xlabel = 'Drive Wheels', \
                                 ylabel = 'Average Price', title = 'Drive Wheels vs Average Price', anova_results = anova_results)




grouped_by_number_of_doors =  data[['num-of-doors','price']].groupby(['num-of-doors'], as_index = False) 
mean_price_grouped_by_number_of_doors = grouped_by_number_of_doors.mean()
mean_price_grouped_by_number_of_doors.set_index('num-of-doors', inplace= True)
indices = list(mean_price_grouped_by_number_of_doors.index)


anova_results = calculateANOVA(data = prepareANOVAData(grouped_by_number_of_doors, indices), xlabel='num-of-doors', ylabel='price')
plotBarChartAcrossCategories(mean_price_grouped_by_number_of_doors, xlabel = 'Number of Doors', \
                                 ylabel = 'Average Price', title = 'Number of Doors vs Average Price', anova_results = anova_results)




grouped_by_number_of_cylinders =  data[['num-of-cylinders','price']].groupby(['num-of-cylinders'], as_index = False) 
mean_price_grouped_by_number_of_cylinders = grouped_by_number_of_cylinders.mean()
mean_price_grouped_by_number_of_cylinders.set_index('num-of-cylinders', inplace= True)
indices = list(mean_price_grouped_by_number_of_cylinders.index)


anova_results = calculateANOVA(data = prepareANOVAData(grouped_by_number_of_cylinders, indices), xlabel='num-of-cylinders', ylabel='price')
plotBarChartAcrossCategories(mean_price_grouped_by_number_of_doors, xlabel = 'Number of Cylinders', \
                                 ylabel = 'Average Price', title = 'Number of Cylinders vs Average Price', anova_results = anova_results)
