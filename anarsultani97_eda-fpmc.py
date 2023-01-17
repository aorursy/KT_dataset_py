# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
from tqdm import tqdm_notebook
import matplotlib.dates as dates
from datetime import datetime
import matplotlib.pyplot as plt

%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
row_data = pd.read_csv('/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv') #import the dataset


if (row_data.isnull().values.any() == True):  # checking missing values
    print(row_data.isnull().sum())
else: 
    print("There is not any null number")
#Total visitors number

df = row_data
visitors  = df['user_id'].nunique()
print("Number of visitors : {}".format(visitors))
# x = pd.Series(visitor_by_date.index.values).apply(lambda s: datetime.strptime(s, '%Y-%m-%d').date())
# y = visitor_by_date[:]

# print(type(visitor_by_date))
# plt.rcParams['figure.figsize'] = (25,12)
# plt.plot(x,y)
# plt.show()
# purchase  = df.loc[df['event_type'] == 'purchase'] #getting only purchase event type 
# purchase = purchase.dropna(axis='rows') # dropping rows that have  at least one missing value

# top_brands = purchase.groupby(['brand'])['brand'].agg(['count']).sort_values(by=['count'],ascending=False)
# top_brands.head(25) # [samsung, apple, xiaomi, huawei, ...]
    

index = df[df['event_type'] == 'remove_from_cart'].index
df.drop(index=index,inplace=True)
df['user_session'] = df['user_session'].astype('category').cat.codes
def take_column_products(multiarr,a):
    array1 = []
    for i in range(len(multiarr)):
        array1.append(multiarr[i][a])
    return array1

def create_product_cols(sessions,df):
    products_for_each_user = []
    
    
    for i in sessions:
        products_for_each_user.append(df.loc[df['user_session'] == i, 'product_id'].values)
     
    
    print(products_for_each_user)
        

        
    column1 = take_column_products(products_for_each_user,0) #product_id1
    column2 = take_column_products(products_for_each_user,1) #product_id2
    column3  = take_column_products(products_for_each_user,2) #product_id3
    column4  = take_column_products(products_for_each_user,3) #product_id3
    column5  = take_column_products(products_for_each_user,4) #product_id3
    column6  = take_column_products(products_for_each_user,5) #product_id3       
    column7  = take_column_products(products_for_each_user,6) #product_id3
    column8  = take_column_products(products_for_each_user,7) #product_id3
    column9  = take_column_products(products_for_each_user,8) #product_id3
    column10  = take_column_products(products_for_each_user,9) #product_id3
    column11  = take_column_products(products_for_each_user,10) #product_id3
    column12  = take_column_products(products_for_each_user,11) #product_id3
   # column13  = take_column_products(products_for_each_user,12) #product_id3

    
    
    
    
    data = {'product_id1': column1, 
        'product_id2': column2, 
        'product_id3': column3,
        'product_id4': column4,
        'product_id5': column5,
        'product_id6': column6,
        'product_id7': column7,
        'product_id8': column8,
        'product_id9': column9,
        'product_id10': column10,
        'product_id11': column11,
        'product_id12': column12,
        } 
    
    new_df = pd.DataFrame(data)
    
    
    
    
#     df.insert(3,'product_id1',column1 )
#     df.insert(4,'product_id2', column2)
#     df.insert(5, 'product_id3' , column3)
#     df.insert(6, 'product_id4' , column4)
#     df.insert(7, 'product_id5' , column5)
#     df.insert(8, 'product_id6' , column6)
#     df.insert(9, 'product_id7' , column7)
#     df.insert(10, 'product_id8' , column8)
#     df.insert(11, 'product_id9' , column9)
#     df.insert(12, 'product_id10' , column10)
#     df.insert(13, 'product_id11' , column11)
#     df.insert(14, 'product_id12' , column12)
   # df.insert(15, 'product_id13' , column13)
   
    
        
     
    return new_df


import numpy as np

target_values = []

def generate_target(df , frequent_products):
        for index, row in df.iterrows():
            for i in range(1,13):
                 if row['product_id'+  str(i)] in frequent_products:
                        target_values.append(row['product_id'+  str(i)])
        return target_values


def convert_product_to_Nan(df):
    for index, row in df.iterrows():
        for i in range(1,13):
            if row['product_id'+  str(i)] == row['target']:
               df.loc[index,'product_id' + str(i)] = np.nan
            
    return df

           
           
        
df = df.drop_duplicates(['user_session','product_id'])

users_interactions_count_df = df.groupby(['user_id', 'product_id' , 'user_session']).size().groupby('user_session').size()  #Ask question about event_time or product_id
print("Number of users: %d" % len(users_interactions_count_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df == 12].reset_index()[['user_session']]
users_with_enough_interactions_df
print(users_with_enough_interactions_df)
print('amount of  users with  13 interactions: %d' % len(users_with_enough_interactions_df))
print(round(len(users_with_enough_interactions_df) * 100 / len(users_interactions_count_df), 2) , "%")
interactions_from_selected_users_df = df.merge(users_with_enough_interactions_df, 
                                           how = 'right',
                                           left_on = 'user_session',
                                           right_on = 'user_session'
                                          )
print(interactions_from_selected_users_df)

print('# of interactions: %d' % len(df))
print('# of interactions from users with at least 12 interactions: %d' % len(interactions_from_selected_users_df))
interactions_from_selected_users_df = interactions_from_selected_users_df.drop(['event_type','event_time','category_id','brand', 'price'],axis=1)
interactions_from_selected_users_df = interactions_from_selected_users_df.drop(['category_code'],axis=1)
interactions_from_selected_users_df['user_session'] = interactions_from_selected_users_df['user_session'].astype('category').cat.codes
interactions_from_selected_users_df[-13: -1]

interactions_from_selected_users_df.drop(interactions_from_selected_users_df[interactions_from_selected_users_df['user_session'] == 0].index,inplace=True)
interactions_from_selected_users_df.drop(interactions_from_selected_users_df[interactions_from_selected_users_df['user_session'] ==-1].index,inplace=True)

print(interactions_from_selected_users_df[-13 : -1])

interactions_from_selected_users_df = interactions_from_selected_users_df.sort_values(by=['user_session'])
print(interactions_from_selected_users_df)
user_session_values = interactions_from_selected_users_df['user_session'].values
user_session_values[-13:-1]

df_new =interactions_from_selected_users_df.groupby(['product_id','user_session'],sort=True)['product_id'].count()
df_new=interactions_from_selected_users_df[['product_id']].apply(pd.Series.value_counts)
df_new = df_new.loc[df_new['product_id'] >=10].reset_index()
df_new.columns = ("product_id", 'count_of_products')
df_new
from collections import Counter

frequent_products =df_new.product_id.values
print(len(frequent_products))
print(frequent_products)
[item for item, count in Counter(frequent_products).items() if count > 1]
frequent_products_sessions_df = interactions_from_selected_users_df.merge(df_new, 
                                           how = 'right',
                                           left_on = 'product_id',
                                           right_on = 'product_id'
                                          )



frequent_sessions = frequent_products_sessions_df['user_session']
frequent_sessions = frequent_sessions.sort_values(ascending=True)

frequent_sessions

print(len(frequent_sessions))
prod_for_basket_whole_df = create_product_cols(frequent_sessions,interactions_from_selected_users_df)
prod_for_basket_whole_df[:50]
df_for_target_df = prod_for_basket_whole_df.drop_duplicates()

target_values = generate_target(df_for_target_df,frequent_products)
print(len(target_values))
print(target_values)
def justify(a, invalid_val=0, axis=1, side='left'):    
    """
    Justifies a 2D array

    Parameters
    ----------
    A : ndarray
        Input array to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """

    if invalid_val is np.nan:
        #change to notnull
        mask = pd.notnull(a)
    else:
        mask = a!=invalid_val
    justified_mask = np.sort(mask,axis=axis)
    if (side=='up') | (side=='left'):
        justified_mask = np.flip(justified_mask,axis=axis)
    #change dtype to object
    out = np.full(a.shape, invalid_val, dtype=object)  
    if axis==1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out 

prod_for_basket_whole_df['target'] = target_values
prod_for_basket_whole_df
prod_for_basket_whole_df = convert_product_to_Nan(prod_for_basket_whole_df)
prod_for_basket_whole_df

new_df = prod_for_basket_whole_df.iloc[:, :12]


df = pd.DataFrame(justify(new_df.values, invalid_val=np.nan, side='left', axis=1), 
                  columns=new_df.columns)


df[:50]
df.dropna(axis=1 , inplace=True) 

df = df.astype('int')
df
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



X = df.values
y = prod_for_basket_whole_df.target.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5 )
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
print(number_of_train, number_of_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import validation_curve 
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import catboost as cb
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# scores = []
# neighbors= []

# for n in range(1 , 400, 50):
#     knn = KNeighborsClassifier(n_neighbors=n)
#     neighbors.append(n)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     score = accuracy_score(y_test,y_pred)
#     scores.append(score)
#     print("Accuracy in neighbor {0} : ".format(n) ,score)

    
# from matplotlib.legend_handler import HandlerLine2D

# line1, = plt.plot(neighbors, scores, 'r', label="Test AUC")

# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel("Accuracy score")
# plt.xlabel('n_neighbor')
# plt.show()
# scores = []
# params= []

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# score = accuracy_score(y_test,y_pred)
# scores.append(score)
# print("Accuracy in max_depth {0} :  " , score*100)

    
# # from matplotlib.legend_handler import HandlerLine2D

# # line1, = plt.plot(params, scores, 'r', label="Test accuracy")

# # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# # plt.ylabel("Accuracy score")
# # plt.xlabel('n_neighbor')
# # plt.show()
