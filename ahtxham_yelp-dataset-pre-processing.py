import pandas as pd
yelp = pd.read_csv("../input/yelp.csv")

Metadata = pd.read_csv("../input/Yelp NYC Metadata.csv")



dataset= pd.DataFrame()



dataset['User_id'] = yelp['Review_id']

dataset['Product_id'] = yelp['Product_id']

dataset['Rating'] = Metadata['Rating']

dataset['Date'] = Metadata['Date']

dataset['Review'] = yelp['Review']

dataset['Label'] = Metadata['Label']

dataset.head()
# Pre-Processing

unique_users = set(dataset['User_id'])

unique_products = set(dataset['Product_id'])

unique_rating = set(dataset['Rating'])

unique_dates = set(dataset['Date'])

# Products Wise Review Count

product_count = []

for prod in unique_products:

    product_count.append(len(dataset.loc[dataset['Product_id'] == prod]))
# Rating Count List For Products

for rating in unique_rating:

    globals()['rating_list_%s' % rating] =[]

for prod in unique_products:

    for rating in unique_rating:

        globals()['rating_list_%s' % rating].append(len(dataset.loc[(dataset['Product_id'] == prod) & (dataset['Rating']==rating)]))
#Label Count List For Products

positive_label=[]

for prod in unique_products:

    positive_label.append(len(dataset.loc[(dataset['Product_id'] == prod) & (dataset['Label']==1)]))

    

negative_label=[]

for prod in unique_products:

    negative_label.append(len(dataset.loc[(dataset['Product_id'] == prod) & (dataset['Label']==-1)]))
# Product Wise Data Frame

products_dataset = pd.DataFrame()

products_dataset['Products_id'] = list(unique_products)

products_dataset['Total_Reviews'] = product_count

products_dataset['1_Ranking'] = rating_list_1

products_dataset['2_Ranking'] = rating_list_2

products_dataset['3_Ranking'] = rating_list_3

products_dataset['4_Ranking'] = rating_list_4

products_dataset['5_Ranking'] = rating_list_5

products_dataset['Positive_Label'] = positive_label

products_dataset['Negative_Label'] = negative_label



products_dataset.head()
# User Wise Review Count

user_count = []

for user in unique_users:

    user_count.append(len(dataset.loc[dataset['User_id'] == user]))
# Rating Count List For User

for rating in unique_rating:

    globals()['User_rating_list_%s' % rating] =[]

for user in unique_users:

    for rating in unique_rating:

        globals()['User_rating_list_%s' % rating].append(len(dataset.loc[(dataset['User_id'] == user) & (dataset['Rating']==rating)]))
#Label Count List For User

positive_label_user=[]

for user in unique_users:

    positive_label_user.append(len(dataset.loc[(dataset['User_id'] == user) & (dataset['Label']==1)]))

    

negative_label_user=[]

for user in unique_users:

    negative_label_user.append(len(dataset.loc[(dataset['User_id'] == user) & (dataset['Label']==-1)]))
# Product Wise Data Frame

user_dataset = pd.DataFrame()

user_dataset['user_id'] = list(unique_users)

user_dataset['Total_Reviews'] = user_count

user_dataset['1_Ranking'] = User_rating_list_1

user_dataset['2_Ranking'] = User_rating_list_2

user_dataset['3_Ranking'] = User_rating_list_3

user_dataset['4_Ranking'] = User_rating_list_4

user_dataset['5_Ranking'] = User_rating_list_5

user_dataset['Positive_Label'] = positive_label_user

user_dataset['Negative_Label'] = negative_label_user



user_dataset.head()