# Importing of the librari

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')

sns.set_style(style = 'whitegrid' )

from sklearn.impute import SimpleImputer, KNNImputer

# from impyute.imputation.cs import fast_knn

from fancyimpute import IterativeImputer

import sys

import matplotlib.gridspec as gridspec

import missingno



plt.style.use('seaborn-talk')
# Fetching the data sets and as it can be seen these are the available files in the directrort

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

data.head()



# As it can be seen below

# 1. Application Name

# 2. What is the Application Category

# 3. Whats the rating out of 5

# 4. How many Reviews are being provided

# 5. How many number of installation has been done

# 6. What kind of Subscription this app Holds

# 7. If its paid then whats the price paid

# 8. Whats the Content Rating Age Group

# 9. What Generes this application belongs to

# 10. When was the last update happened

# 11. What's the last updated date of the Application

# 12. What's the current Version Number of the Application

# 13. What's the Android version till which the Application is supported
print(f"Number of Data in the Dataset: {data.size}")

print(f"Dimension of the Dataset: {data.shape}" )

print(f"Columns of the Dataset are: {data.columns.to_list()}")
# To check some of the statistics about the data 

data.info()
# As it can be see above in Installs there are special charaacters

# Like + and , and it should ideally be an Integer, hence the data type conversion

# lets have a look into the data



print("To check the Value counts of different categories")

data.Installs.value_counts()

# As it can be seen below all the integers has the value with + so we can take an assumption 

# All the Installation numbers are more than whats being maintained 
# To have better view on the data as it can be seen above that there is a value which which has 

# a type as Free

print(data.Type.value_counts())

print("=" * 40)

data.head()
# As it can be seen above there is an app where the Installation is free, which is kindaa wierd 

data[data["Installs"] == 'Free']



# As it can be see from above for this record we dont have the Category and thats the reason all the records are shifted by left by 1 record

# Because as it can be seen above the type column can only hold a record with Free type

# So with the name https://play.google.com/store/apps/details?id=com.lifemade.internetPhotoframe&hl=en_IN as the per play store the app type is lifestyle 

# Hence the type will be assigned it
data_index = data[data['App'] == 'Life Made WI-Fi Touchscreen Photo Frame' ].index.tolist()[0]

temp_df = data[data['App'] == 'Life Made WI-Fi Touchscreen Photo Frame' ].shift(periods = 1, axis = 1).copy()



temp_df.at[data_index, 'Category'] = 'LIFESTYLE'

temp_df.App = temp_df.App.apply(str)

temp_df.at[data_index, 'App'] = 'Life Made WI-Fi Touchscreen Photo Frame'

temp_df.Rating = temp_df.Rating.astype('float64')



data[data['App'] == 'Life Made WI-Fi Touchscreen Photo Frame' ] = temp_df

data[data['App'] == 'Life Made WI-Fi Touchscreen Photo Frame' ]



# As it can be seen below now the data looks as expected 
# Hence we can replace + and , in the install feature, as the + and , is a special characters

# For the training only the numeric features is required

data.Installs = data.Installs.apply(lambda x : x.replace('+', '') if '+' in str(x) else x)

data.Installs = data.Installs.apply(lambda x : x.replace(',', '') if ',' in str(x) else x)

data.Installs = data.Installs.astype('int64')
print("Final Look after the data cleaning")

print(data.Installs.value_counts())
# Lets analyze the Size 

# here also as it can be seen below the values looks like an Integer 

print(data.Size.head())



# So lets try to adjust the data 

# 1. Remove the Varies with device with NaN

data.Size = data["Size"].apply(lambda x: float(x.replace('Varies with device', 'NaN')) if 'Varies with device' in str(x) else x)

data.Size = data["Size"].apply(lambda x: ( float((x.replace('M', '')).replace(',', ''))* 10**6)  if 'M' in str(x) else x)

data.Size = data["Size"].apply(lambda x: float(x.replace('k', '')) * 10**3  if 'k' in str(x) else x)



data.Size = data.Size.apply(np.float64)



print("="*20)

print("After applying the transformation on Size column")

print(data.Size.head())



# Lets also change the data type of the Reviews feature

# As that also looks like an integer

data.Reviews = data.Reviews.astype(np.int64)



# Lets have a look into the price 

# It has a special character $ currency, so all the apps prices are in dollar

# Assumption 2: The price is in dollar for all the apps



data.Price = data.Price.apply(lambda x: float(x.replace('$', '')) if '$' in str(x) else float(x))



# And also lets changes the data type of Last Updated to date time



data['Last Updated'] = pd.to_datetime(data['Last Updated'])
# After apply the transformation, lets check the final Data Types

data.info()
def print_the_number_missing_data(data):

    # Lets check the number of duplicated rows

    print("="*30)

    print(f'Number of Duplicate Rows {data[data.duplicated()].shape[0]}')



    #Let's check the percentage of missing values for the application dataset.

    total_missing = data.isnull().sum()

    percent = round((100*(total_missing/data.isnull().count())),2)



    #Making a table for both and get the top 20 Columns



    missing_value_app_data = pd.concat([total_missing, percent], axis =1, keys= ['Total_missing', 'percent'])

    missing_value_app_data.sort_values(by='Total_missing',ascending=False,inplace=True)

    print("="*30)

    print("Number and % of Missing Value")

    print(missing_value_app_data)



print_the_number_missing_data(data)
# So we would drop all the duplicated rows, as duplicated rows is of lowest importance

data.drop_duplicates(inplace = True)

print("After droping the duplicate Record")

print_the_number_missing_data(data)

missingno.matrix(data)

plt.show()
# Lets see how the categorical data will impact the 

print(data.Category.value_counts())

plt.figure(figsize=(15,10))

sns.categorical.countplot(data.Category, order = data.Category.value_counts().index )

plt.xticks(rotation = 60)

mean = np.mean(data.Category.value_counts().values.tolist())

plt.axhline(mean, color='r', linestyle='--', ls='--',label='p=0.05')

plt.legend({'Mean':mean})

plt.title('APPs Category Frequency Distribution')

plt.show()
# Selecting the top 5 category from the above o/p

all_cat_data= data[['Category', 'Installs']].groupby('Category')['Installs'].agg('sum').reset_index(name='Number_of_Installation')



top5_cat = data.Category.value_counts().index.tolist()[:5]

top5_cat_data = all_cat_data[all_cat_data.Category.isin(top5_cat)]

top5_cat_data.sort_values(by = ['Number_of_Installation'], ascending = False, inplace = True)

sns.barplot(y = top5_cat_data['Category'], x = top5_cat_data['Number_of_Installation'], ci = None)

plt.title('Comparing Top 5 App Category with number of installs')

plt.show()



print("="*20)



all_cat_data.sort_values(by = ['Number_of_Installation'], ascending = False, inplace = True)

sns.barplot(y = all_cat_data['Category'], x = all_cat_data['Number_of_Installation'], ci = None)

plt.title('Comparing All App Category with number of installs')

plt.show()
# To have a better insight lets try to predict the correlation

plt.title('Lets see the correlation')

sns.heatmap(data.corr(), annot=True)

plt.show()
most_reviews = data.groupby('App')[['Reviews']].mean().sort_values('Reviews', ascending=False).head(10).reset_index()

sns.barplot(y = most_reviews['App'], x = most_reviews['Reviews'], ci = None)

plt.title('App vs Reviews')

plt.show()
install_type = data.groupby('Type')['Installs'].agg('sum').reset_index(name='Number_of_Installations')

print(install_type)



plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

plt.title('Type of Apps vs Installation')

sns.barplot(x=install_type['Type'],y=install_type['Number_of_Installations'])



plt.subplot(1,2,2)

plt.title('Type of Apps vs Installation in Log Scale')

sns.barplot(x=install_type['Type'],y=install_type['Number_of_Installations'])

plt.yscale('log')

plt.show()
paidAppsData = data[data.Type == 'Paid']

paid_cat_data= paidAppsData[['Category', 'Installs']].groupby('Category')['Installs'].agg('sum').reset_index(name='Number_of_Installation')

paid_cat_data.sort_values(by = ['Number_of_Installation'], ascending = False, inplace = True)

sns.barplot(y = paid_cat_data['Category'], x = paid_cat_data['Number_of_Installation'], ci = None)

plt.title('Comparing All App Category with number of installs for Paid Apps')

plt.show()
app_Count_per_android_vrs = data.groupby("Android Ver")['App'].count().reset_index(name = 'App_Count')

app_Count_per_android_vrs.sort_values(by = ['App_Count'], ascending = False, inplace = True)

plt.title('Android Version vs Number of Apps')

sns.barplot(x=app_Count_per_android_vrs['App_Count'],y=app_Count_per_android_vrs['Android Ver'])

plt.show()
plt.figure(figsize=(10,8))

sns.barplot(x=data['Content Rating'], y=data['Rating'], palette="deep", ci = False)

plt.axhline(0, color="k", clip_on=False)

plt.xlabel('Content Ratings')

plt.xticks(rotation=90)

plt.ylabel("Ratings ")

plt.suptitle('Ratings vs Content Ratings', fontsize=20)

# Adults only 18+ category apps have maximum ratings
sns.barplot(data['Content Rating'].value_counts().index, data['Content Rating'].value_counts())

plt.xticks(rotation=75);

plt.title('Age group the app is targeted at')

plt.show()

# As it can be seen below most of the Apps are targetted for everyone
# Lets check the distribution of rating

plt.figure(figsize=(10, 5))

plt.title('User rating Distribution')

plt.xlabel('Ratings')

plt.ylabel('Number of users')



plt.hist(data.Rating, bins=np.arange(1,6,0.5))

plt.show()



# As it can be seen below most of the apps rating are between 3 - 4.5
plt.figure(figsize=(50,15))

sns.catplot(x='Category',y='Rating',data=data,kind='box',height=10,showmeans=True)

plt.title("Rating of each category", size=18)

plt.xticks(rotation=90)

plt.show()
data['Last_Updated_Year'] = data['Last Updated'].dt.year

most_downloads_by_year = data.groupby(['Last_Updated_Year','Category'])[['Installs']].sum().sort_values('Installs', ascending=False).reset_index()

most_downloads_by_year = most_downloads_by_year.loc[most_downloads_by_year.groupby("Last_Updated_Year")["Installs"].idxmax()]

most_downloads_by_year
data['Diff_Bet_Second_Last_Update'] =  data['Last Updated'].max() - data['Last Updated']

print(data[['Category','Diff_Bet_Second_Last_Update']].sort_values('Diff_Bet_Second_Last_Update', ascending=False).describe())

print("="*20, end = '\n\n\n')

data[['App','Category','Diff_Bet_Second_Last_Update']].sort_values('Diff_Bet_Second_Last_Update', ascending=False).head(10)

# As it can be seen that below are the top 10 Apps and their category and for how long its not been updated

# And it can be seen most of the apps are frequenty updated, only the top 25% app have not been updated for too long
thereshold_val = np.percentile(data['Diff_Bet_Second_Last_Update'].dt.days.values.tolist(), 75)

apps_not_updated_fr_most_Days =  data[data['Diff_Bet_Second_Last_Update'].dt.days > thereshold_val]['Category'].value_counts()

sns.barplot(y = apps_not_updated_fr_most_Days.index, x = apps_not_updated_fr_most_Days.values, ci = None)

plt.title('Apps which are not updated for Most number of Days')

plt.xlabel('Days')

plt.show()



# As it can be seen below the Family, followed by Game and Tools are the one which is not updated for most number of days
data.describe().T

# As it can be seen the max value is 19 and the min is 1, so lets try to see the range 
#  This will plot the numeric features that will be boxplot and distribution plot

def plot_numeric_features(dataset, column):

    plt.figure(figsize=(10, 4))

    plt.suptitle("Distribution of " + column)

    plt.subplot(1,2,1)

    sns.boxplot(y = dataset[column])

    plt.xlabel(column)



    plt.subplot(1,2,2)

    sns.distplot(data[column])

    plt.axvline(np.median(data[column]),color='b', linestyle='--')

    plt.show()



def plot_numeric_features_with_statistic(df, column):

    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)}, figsize=(10,8))

    mean=df[column].mean()

    median=df[column].median()

    mode=df[column].mode().values.tolist()[0]

    sns.boxplot(df[column], ax=ax_box)

    ax_box.axvline(mean, color='r', linestyle='--')

    ax_box.axvline(median, color='g', linestyle='-')

    ax_box.axvline(mode, color='b', linestyle='-')



    sns.distplot(df[column], ax=ax_hist)

    ax_hist.axvline(mean, color='r', linestyle='--')

    ax_hist.axvline(median, color='g', linestyle='-')

    ax_hist.axvline(mode, color='b', linestyle='-')

    

    plt.legend({'Mean':mean,'Median':median,'Mode':mode})



    ax_box.set(xlabel='')

    

# Using the imputation as a strategy

def imputation_using_strategy(data, column, strategy):

    simp_imp = SimpleImputer(strategy= strategy)

    data_in_consider = data[column].values.reshape(-1,1) 

    simp_imp.fit(data_in_consider)

    return simp_imp, simp_imp.transform(data_in_consider).reshape(-1)



# This will do the Imputation using the KNN Approach

# The algorithm uses ‘feature similarity’ to predict the values of any new data points. 

# This means that the new point is assigned a value based on how closely it resembles 

# the points in the training set.

def imputation_using_knn(data, column):

#     sys.setrecursionlimit(100000) 

#     return fast_knn(data[column].values.reshape(-1,1) , k=30).reshape(-1)

    imputer = KNNImputer(n_neighbors=30, weights='uniform', metric='nan_euclidean')

    data_in_consider = data[column].values.reshape(-1,1) 

    imputer.fit(data_in_consider)

    return imputer, imputer.transform(data_in_consider).reshape(-1)



# Imputation with Multivariate Imputation by Chained Equation (MICE)

# This type of imputation works by filling the missing data multiple times.

def imputation_using_mice(data, column):

    data_in_consider = data[column].values.reshape(-1,1)

    mice_imputer = IterativeImputer()

    mice_imputer.fit(data_in_consider)

    return mice_imputer, mice_imputer.transform(data_in_consider).reshape(-1)



# Gets the imputed columns and then print the corresponding values as a single image as a subplots

def plot_multiple_values(imputed_column, original_col_label, images_per_row = 2):

    column_list = imputed_column.columns.tolist()

    images_per_row = min(len(column_list), images_per_row)

    n_rows = (len(column_list) - 1) // images_per_row + 1

    axes = []

    fig = plt.figure(figsize=(20,12))

    outer = gridspec.GridSpec(n_rows, images_per_row, wspace =  0.2, hspace=0.1)

    for i in range(len(column_list)):

        inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[i], wspace=0.1, hspace=0.1, height_ratios = (0.2, 1))

        ax_box = plt.Subplot(fig, inner[0])

        column = column_list[i]

        mean=imputed_column[column].mean()

        median=imputed_column[column].median()

        mode=imputed_column[column].mode().values.tolist()[0]

        sns.boxplot(imputed_column[column], ax=ax_box)

        ax_box.axvline(mean, color='r', linestyle='--')

        ax_box.axvline(median, color='g', linestyle='-')

        ax_box.axvline(mode, color='b', linestyle='-')

        ax_box.set(xlabel='')

        fig.add_subplot(ax_box)

        

        ax_hist = plt.Subplot(fig, inner[1], sharex = ax_box)

        sns.distplot(imputed_column[column], ax=ax_hist)

        ax_hist.axvline(mean, color='r', linestyle='--')

        ax_hist.axvline(median, color='g', linestyle='-')

        ax_hist.axvline(mode, color='b', linestyle='-')

        ax_hist.legend({'Mean':mean,'Median':median,'Mode':mode})

        fig.add_subplot(ax_hist)

    plt.suptitle(f"To see the Data Distribution of {original_col_label} after apply different Imputation Technique")

    for ax in fig.get_axes():

        ax.label_outer()

    plt.show()
# lets start with the Feature Size

plot_numeric_features(data, "Size")

# As it can be see most of the apps have a size between 0-0.6 MB
result_df = pd.DataFrame()

estimator_list = [] # This holds the estimator list that will be used later for the imputation in the test set

# Imputation using median

estimator, imputed_data = imputation_using_strategy(data, "Size", "median")

estimator_list.append(estimator)

result_df['imp_mean'] = pd.Series(imputed_data)

print('Imputation with Mean')





#Imputation using Most Frequent

estimator, imputed_data = imputation_using_strategy(data, "Size", "most_frequent")

estimator_list.append(estimator)

result_df['imp_most_freq'] = pd.Series(imputed_data)

print('Imputation with Most Frequent')



# Imputation using the KNN Strategy

estimator, imputed_data = imputation_using_knn(data, "Size")

estimator_list.append(estimator)

result_df['knn'] = pd.Series(imputed_data)

print('Imputation with KNN')



# Imputation using the MICE Strategy

estimator, imputed_data = imputation_using_mice(data, "Size")

estimator_list.append(estimator)

result_df['mice'] = pd.Series(imputed_data)

print('Imputation with MICE')
plot_multiple_values(result_df, "Size")
final_estimator_list = dict()

plot_numeric_features_with_statistic(data, "Size")

plt.title("Data without Imputation")

plt.show()

data.Size = result_df['imp_most_freq']

plot_numeric_features_with_statistic(data, "Size")

plt.title("Data After Imputation")

final_estimator_list['na_Size'] = estimator_list[1]

plt.show()
print_the_number_missing_data(data)
plot_numeric_features(data, "Rating")
print(data[data["Rating"] > 5].T)

print("="*40)

print("Index where the Anomaly rating is present" , data[data["Rating"] > 5].index.tolist())

print("="*40)

print("Data will be droppped now")

data.drop(index =data[data["Rating"] > 5].index.tolist(), inplace=True)
plot_numeric_features(data, "Rating")

# It can be seen now that the 

plt.figure(figsize=(12, 4))

g = sns.factorplot("Rating", data=data, aspect=1.5, kind="count")

g.set_xticklabels(rotation=80)

plt.show()



# As it can be seen below most of the Rating is highly 