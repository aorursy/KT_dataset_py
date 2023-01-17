import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm;



%matplotlib inline



#gathering the data

df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
#the number of rows and columns

num_rows = df.shape[0]

num_cols = df.shape[1]

print('number of rows equals to {} '.format(num_rows))

print('number of columns equals to {}'.format(num_cols))
#set of columns that have no missing values

no_nulls = set(df.columns[df.isnull().sum() == 0])

no_nulls
#set of columns that have more than 75% if their values missing

most_missing_cols = set(df.columns[df.isnull().mean() > 0.75])

most_missing_cols
null_cols = set(df.columns[df.isnull().sum() != 0])

null_cols
df.describe()
#check the data types and check if any columns are of different data types that you would expect

#check the missing values for every column

df.info()
#discover the number of unique values for categorical columns

print(df.minimum_nights.nunique(), df.id.nunique(), df.host_id.nunique())

print(df.neighbourhood_group.nunique(), df.neighbourhood.nunique())

print(df.room_type.nunique(), df.availability_365.nunique())

print(df.price.nunique(), df.number_of_reviews.nunique())
#visualize the categorical values for the neighbourhood_group

plt.figure(figsize=(6,4))

count_neigh = df.neighbourhood_group.value_counts()

(count_neigh/df.shape[0]).plot(kind='bar');

plt.title('the percent of every neighbourhood group', fontsize = 16)

plt.ylabel('percent', fontsize = 16)

plt.xlabel('neighbourhood group', fontsize = 16)

#visualize the categorical values for the room_type

count_room = df.room_type.value_counts()

(count_room/df.shape[0]).plot(kind='bar');

plt.title('room_type')

plt.ylabel('the percent of every room type')
def clean(df):

    '''

    cleaning the dataframe by converting the data types for the id and host_id to Object,

    filling the reviews per month with 0 value

    filling the missing values in the names cloumns with special characters,

    then removing the last_review column, I'll not use it in my analysis

    

    input -> the dataframe which was generated from the dataset

    output -> a cleaned dataframe

    '''

    

    df[['host_id', 'id']] = df[['host_id', 'id']].astype('str')

    df['reviews_per_month'].fillna(0,inplace = True)

    df['name'].fillna("$$$",inplace=True)

    df['host_name'].fillna("$$$",inplace=True)

    df.drop(['last_review'],axis=1,inplace=True)

    return df

df = clean(df)
df_host_area = df.groupby(['host_id', 'neighbourhood_group']).count()['id'].to_frame().reset_index()

df_host_area.sort_values('id', ascending = False, inplace =True)

df_host_area.rename(columns = {'id':'count'}, inplace = True)

    

#create a list of the names of the neighbourhood groups

neigh_list = df_host_area['neighbourhood_group'].unique().tolist()

    

#create a dictionary to hold the name od the group with the count of unique hosts

count_unique_host = {}



i = 0

for group in neigh_list:

    df_group = df_host_area[df_host_area.neighbourhood_group == group]

    count_unique_host[group] = df_group.shape[0]

        

    #print the first two hosts ids in every group

    print(df_group.head(2))

   

    #plotting the number of unique hosts per group

plt.figure(figsize=(10,6))

    

plt.bar(count_unique_host.keys(), count_unique_host.values())

plt.title('the number of unique hosts per neighbourhood group', fontsize =16)

plt.ylabel('the number of unique hosts', fontsize = 14)

plt.xlabel('the neighbourhood group', fontsize =14)
#Analyzing the number of rooms for every host and neighbourhood group

def unique_hosts(df, neighbour_count = 5, plot = True):

    

    '''

    inputs:

    df -> the dataframe of all the dataset after cleaning

    neihbour_count -> the number of neighbourhood groups, here in this dataset equals to 5

    plot-> True if we needs to making plots

    

    output:

    print(the top 2 host id's)

    the count of unique hosts in every neighbourhood group

    

    plot:

    the top 10 host's ids who have the highest number of rooms in every group

    the number of unique hosts per neighbourhood group

    

    '''

    #create a new dataframe with the number of listings for every host_id in every group

    df_host_area = df.groupby(['host_id', 'neighbourhood_group']).count()['id'].to_frame().reset_index()

    df_host_area.sort_values('id', ascending = False, inplace =True)

    df_host_area.rename(columns = {'id':'count'}, inplace = True)

    

    #create a list of the names of the neighbourhood groups

    neigh_list = df_host_area['neighbourhood_group'].unique().tolist()

    

    #create a dictionary to hold the name od the group with the count of unique hosts

    count_unique_host = {}

    #create a subplots

    fig, axs = plt.subplots(neighbour_count+1,1, figsize = (20,30))

    fig.subplots_adjust(hspace=1)



    i = 0

    for group in neigh_list:

        df_group = df_host_area[df_host_area.neighbourhood_group == group]

        count_unique_host[group] = df_group.shape[0]

        

        #print the first two hosts ids in every group

        print(df_group.head(3))

        #plot the top 10 hosts with the number of their listings

        if plot:

            axs[i].bar(df_group.head(10)['host_id'], df_group.head(10)['count'])

            axs[i].set_title('the top 10 hosts ids who have the highest number of rooms in {} '.format(group))

            axs[i].set_ylabel('the number of rooms for every host')

            axs[i].set_xlabel('host id')

            i += 1

    #plotting the number of unique hosts per group

    if plot:

        axs[neighbour_count].bar(count_unique_host.keys(), count_unique_host.values())

        axs[neighbour_count].set_title('the number of unique hosts per neighbourhood group')

        axs[neighbour_count].set_ylabel('the number of unique hosts')

        axs[neighbour_count].set_xlabel('the neighbourhood group')

    

    return count_unique_host    
unique_hosts(df)
def plot_price_group(df):

    '''

    This function for plotting the price distribution for every neighbourhood group

    

    input -> the dataframe

    output -> plotting the price distribution on average for every  neighbourhood group based on the neighbourhoods

    '''

    neigh_list = df['neighbourhood_group'].unique().tolist()

    fig,axs = plt.subplots(5,1,figsize=(20,20) )

    fig.subplots_adjust(hspace=1)



    i = 0

    for group in neigh_list:

        df_price = df[df.neighbourhood_group==group][["neighbourhood","price"]]

        df_price = df_price.groupby("neighbourhood").mean()

        sns.distplot(df_price, ax = axs[i])

        axs[i].set_xlabel(' The price in {}'.format(group))

        axs[i].set_title('the price distribution in {}'.format(group))

        i += 1

plot_price_group(df)
def plot_price_room(df):

    '''

    This function for plotting the price distribution for every rom type

    input -> the dataframe

    output -> plotting the price distribution for every group

    '''

    room_list = df['room_type'].unique().tolist()

    fig,axs = plt.subplots(3,1,figsize=(10,20) )

    fig.subplots_adjust(hspace=1)



    i = 0

    for room in room_list:

        df_price = df[df.room_type==room][["neighbourhood_group","price"]]

        df_price = df_price.groupby("neighbourhood_group").mean()

        sns.distplot(df_price, ax = axs[i])

        axs[i].set_xlabel(' The price in {}'.format(room))

        axs[i].set_title('the price distribution in {}'.format(room))

        i += 1
plot_price_room(df)
def price_relation(df, disc_var, max_val, min_val, step):

    '''

    This function exploes the relationship between the price and any other numeric feature

    This function restricts the values of price from 0 to 200

    inputs:

    df -> the cleaned dataframe

    disc_var -> the numeric(discrete) feature

    max_val -> the maximum value of the numeric feature used in the plot

    min_val -> the minimum value of the numeric feature used in the plot

    step -> the step between two values in the plot

    

    outputs:

    plotting the heatmap plot to discover many insights from it

    '''

    

    plt.figure(figsize = [12, 5])



    # left plot: scatterplot of discrete data with jitter and transparency

    plt.subplot(1, 2, 1)

    sns.regplot(data = df, x = 'price', y = disc_var, fit_reg = False,

               x_jitter = 0.2, y_jitter = 0.2, scatter_kws = {'alpha' : 1/3})



    # right plot: heat map with bin edges between values

    plt.subplot(1, 2, 2)

    bins_x = np.arange(0, 200+.5,20)

    bins_y = np.arange(min_val, max_val+1, step)

    plt.hist2d(data = df, x = 'price', y = disc_var, cmin = .5,

               bins = [bins_x, bins_y])

    plt.colorbar();
price_relation(df, 'minimum_nights', 10, 0, 1)
price_relation(df, 'availability_365', 200, 1, 5)
price_relation(df, 'number_of_reviews', 25, 0, 1)
def multiexplore(df, disc_var):

    '''

    This function explores the relationship between price and 

    another numeric feature(minimum nights, number of reviews, availability) according to

    every neighbourhood and every room type

    

    inputs:

    df -> the cleaned dataframe

    disc_var -> the numeric(discrete) feature

    

    output:

    plotting the scatterplots

    '''

    g = sns.FacetGrid(data = df, col = 'room_type', row = 'neighbourhood_group', size = 2.5,

                    margin_titles = True)

    g.map(plt.scatter, 'price', disc_var)

    g.fig.set_size_inches(15,20)

multiexplore(df, 'minimum_nights')
multiexplore(df, 'number_of_reviews')
def explan_price(df):

    '''

    This function shows a visualization for the relationship between price, neighbourhood group, and room type

    

    inputs:

    df -> the cleaned dataframe

    outputs:

    the line visualization

    '''

    plt.figure(figsize = [8,6])



    ax = sns.pointplot(data = df, x = 'room_type', y = 'price', hue = 'neighbourhood_group')

    ax.set_yticklabels([0,400,50], minor = True)

    plt.title('The price range in every neighbourhood group for every room type', fontsize = 16)

    plt.xlabel('Room Type',fontsize = 14)

    plt.ylabel('Price', fontsize = 14)

    ax.set_ylim(0,400)

    ax.set_yticks(range(0,450, 50))

    plt.show();
explan_price(df)
def create_dummy_df(df, dummy_na):

    '''

    INPUT:

    df - pandas dataframe with categorical variables you want to dummy

    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not

    

    OUTPUT:

    df - a new dataframe that has the following characteristics:

            1. contains all columns that were not specified as categorical

            2. removes all the original columns in cat_cols and unwanted columns in the prediction

            3. dummy columns for each of the categorical columns in cat_cols

            4. if dummy_na is True - it also contains dummy columns for the NaN values

            5. Use a prefix of the column name with an underscore (_) for separating 

    '''

    df.drop(['host_name', 'id', 'name', 'host_id', 'neighbourhood', 'latitude', 'longitude',

         'reviews_per_month', 'calculated_host_listings_count'], axis =1, inplace = True)

    cat_cols = df.select_dtypes(include=['object']) # Subset to a dataframe only holding the categorical columns

    for col in  cat_cols:

        try:

            # for each cat add dummy var, drop original column

            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)

        except:

            continue

    return df
def price_pred(df):

    '''

    This function for predicting the price using these features:

    neighbourhood_group, room_type, minimum_nights, number_of_reviews, availability_356

    INPUT:

    df -> the cleaned dataframe

    OUTPUT:

    printing the r-squared score for our model

    '''

    df = create_dummy_df(df, dummy_na=False)



    X = df.drop('price', axis =1)

    y = df['price']



    scaler = StandardScaler() #scaling all the features using the normalization

    scaled_X = scaler.fit_transform(X)



    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.25)#splitting the data to train and test data



    model = LinearRegression() #instantiate the model

    model.fit(X_train,y_train) #fit the model



    y_test_preds = model.predict(X_test) #predict the test values

    print("The r-squared score for the model was {} on {} values.".format(r2_score(y_test, y_test_preds), len(y_test)))    
price_pred(df)
#reading the dataframe again and cleaning it (because of the changes in the dataframe happened for prediction)

df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df = clean(df)
def highest_reviews_rooms(df):

    '''

    INPUT:

    df -> the cleaned dataframe

    

    OUTPUT:

    df_max_reviews -> a dataframe holds the names of neighbourhood_group,neighbourhoods that

                      have the highest number of reviews in every group

    df_max_rooms ->   a dataframe holds the names of neighbourhood_group,neighbourhoods that

                      have the highest number of rooms in every group 

                      

    '''

    df_group_reviews = df.groupby(['neighbourhood_group', 'neighbourhood']).sum()['number_of_reviews'].to_frame().reset_index()

    df_max_reviews = df_group_reviews.groupby('neighbourhood_group').max()[['neighbourhood','number_of_reviews']].reset_index()

        

    df_group_count = df.groupby(['neighbourhood_group', 'neighbourhood']).count()['id'].to_frame().reset_index()

    df_max_rooms = df_group_count.groupby('neighbourhood_group').max()[['neighbourhood','id']].reset_index().rename(columns = {'id':'number of listings'})

        

    return df_max_reviews, df_max_rooms
#print the two tables which have the names of neighbourhoods 

#that have the highest number of reviews, rooms in every group

print(highest_reviews_rooms(df)[0], '\n')

print(highest_reviews_rooms(df)[1])


