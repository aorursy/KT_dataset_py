from IPython.display import Image,display

Image("../input/fm-diag/FM-diag.PNG")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from scipy.sparse.linalg import svds

%matplotlib inline

import matplotlib.pyplot as plt



import seaborn as sns

import matplotlib.image as mpimgimport 

import sys

if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

from statistics import mean

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from random import shuffle  

from zipfile import ZipFile

##Deep Learning specific stuff

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense , Concatenate,Add

from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import model_to_dot

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.layers import Dropout, Flatten,Activation,Input,Embedding

from tensorflow.keras.layers import  BatchNormalization

from tensorflow.keras.layers import dot

from tensorflow.keras.models import Model







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
book_rating = pd.DataFrame()

for file in glob.glob("../input/goodreads-book-datasets-10m/book*.csv"):

    df = pd.read_csv(file)

    if book_rating.empty:

        book_rating = df

    else:

        book_rating.append(df, ignore_index=True)



        

user_rating = pd.DataFrame()

for file in glob.glob("../input/goodreads-book-datasets-10m/user_rating*.csv"):

    df = pd.read_csv(file)

    if user_rating.empty:

        user_rating = df

    else:

        user_rating.append(df, ignore_index=True)
book_rating.shape,user_rating.shape
book_rating.head(3)
book_rating['Num_1star_rating']=book_rating['RatingDist1'].str.split('\:').str[-1].str.strip()

book_rating['Num_2star_rating']=book_rating['RatingDist2'].str.split('\:').str[-1].str.strip()

book_rating['Num_3star_rating']=book_rating['RatingDist3'].str.split('\:').str[-1].str.strip()

book_rating['Num_4star_rating']=book_rating['RatingDist4'].str.split('\:').str[-1].str.strip()

book_rating['Num_5star_rating']=book_rating['RatingDist5'].str.split('\:').str[-1].str.strip()

book_rating['Num_1star_rating'] = book_rating['Num_1star_rating'].astype(int)

book_rating['Num_2star_rating'] = book_rating['Num_2star_rating'].astype(int)

book_rating['Num_3star_rating'] = book_rating['Num_3star_rating'].astype(int)

book_rating['Num_4star_rating'] = book_rating['Num_4star_rating'].astype(int)

book_rating['Num_5star_rating'] = book_rating['Num_5star_rating'].astype(int)

book_rating['Total_rating_count']= book_rating['Num_1star_rating']+book_rating['Num_2star_rating']+book_rating['Num_3star_rating']+book_rating['Num_4star_rating']+book_rating['Num_5star_rating']

book_rating['Pct_1Star']=book_rating['Num_1star_rating']/book_rating['Total_rating_count']

book_rating['Pct_2Star']=book_rating['Num_2star_rating']/book_rating['Total_rating_count']

book_rating['Pct_3Star']=book_rating['Num_3star_rating']/book_rating['Total_rating_count']

book_rating['Pct_4Star']=book_rating['Num_4star_rating']/book_rating['Total_rating_count']

book_rating['Pct_5Star']=book_rating['Num_5star_rating']/book_rating['Total_rating_count']

book_rating.head()
book_rating_df = book_rating[['Name','pagesNumber','PublishMonth','PublishDay',\

                              'CountsOfReview','Num_1star_rating','Num_2star_rating',\

                              'Num_3star_rating','Num_4star_rating','Num_5star_rating']]

scaling_cols = ['pagesNumber','PublishMonth','PublishDay',\

                              'CountsOfReview','Num_1star_rating','Num_2star_rating',\

                              'Num_3star_rating','Num_4star_rating','Num_5star_rating']

def mix_max_scaler(df):

    result = df.copy()

    for feature_name in scaling_cols:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result

book_rating_scaled = mix_max_scaler(book_rating)

book_rating_df = book_rating_scaled[['Name','pagesNumber','PublishMonth','PublishDay',\

                              'CountsOfReview','Num_1star_rating','Num_2star_rating',\

                              'Num_3star_rating','Num_4star_rating','Num_5star_rating']]
##Let's create Book_id that we can use

book_id_0 = book_rating_df[['Name']]

book_id_1 = user_rating[['Name']]

book_id = pd.concat([book_id_0,book_id_1],axis=0,ignore_index=True)

book_id.rename(columns={ book_id.columns[0]: "Name" }, inplace = True)

book_id.drop_duplicates(inplace=True)

book_id['Book_Id']=book_id.index.values

book_id.head()
user_rating.head(3)
user_rating = pd.merge(user_rating,book_id, on='Name', how='left')

book_rating_df = pd.merge(book_rating_df,book_id, on='Name', how='left')

user_rating.head()
user_rating['Rating'].unique()
le = preprocessing.LabelEncoder()

user_rating['Rating_numeric'] = le.fit_transform(user_rating.Rating.values)

book_rating_numeric = book_rating_df[['Book_Id','pagesNumber','PublishMonth','PublishDay',\

                              'CountsOfReview','Num_1star_rating','Num_2star_rating',\

                              'Num_3star_rating','Num_4star_rating','Num_5star_rating']]

user_rating = pd.merge(user_rating,book_rating_numeric, on='Book_Id', how='left')

user_rating.head()
user_rating.fillna(0, inplace=True)
users = user_rating.ID.unique()

books = user_rating.Book_Id.unique()



userid2idx = {o:i for i,o in enumerate(users)}

bookid2idx = {o:i for i,o in enumerate(books)}

user_rating['ID'] = user_rating['ID'].apply(lambda x: userid2idx[x])

user_rating['Book_Id'] = user_rating['Book_Id'].apply(lambda x: bookid2idx[x])

y=user_rating['Rating_numeric']

X=user_rating.drop(['Rating_numeric'],axis=1)

####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#split = np.random.rand(len(user_rating)) < 0.8

#train = user_rating[split]

#valid = user_rating[~split]

print(X_train.shape , X_test.shape)
exogenous_train = np.array(X_train[['pagesNumber','PublishMonth','PublishDay',\

                              'CountsOfReview','Num_1star_rating','Num_2star_rating',\

                              'Num_3star_rating','Num_4star_rating','Num_5star_rating']])

exogenous_valid = np.array(X_test[['pagesNumber','PublishMonth','PublishDay',\

                              'CountsOfReview','Num_1star_rating','Num_2star_rating',\

                              'Num_3star_rating','Num_4star_rating','Num_5star_rating']])
model = Sequential()

embedding_layer = Embedding(input_dim=10, output_dim=4, input_length=2)

model.add(embedding_layer)

model.compile('adam', 'mse')

model.summary()
input_data = np.array([

    [1,2]

])



pred = model.predict(input_data)



print(input_data.shape)

print(pred)
embedding_layer.get_weights()
Image("../input/lightfm-pred/lightfm_rating_pred.PNG")
def build_model(dropout,latent_factors):

    n_books=len(user_rating['Book_Id'].unique())

    n_users=len(user_rating['ID'].unique())

    n_latent_factors=latent_factors  # hyperparamter to deal with. 

    user_input=Input(shape=(1,),name='user_input',dtype='int64')

    user_embedding=Embedding(n_users,n_latent_factors,name='user_embedding')(user_input)

    user_vec =Flatten(name='FlattenUsers')(user_embedding)

    user_vec=Dropout(dropout)(user_vec)

    book_input=Input(shape=(1,),name='book_input',dtype='int64')

    book_embedding=Embedding(n_books,n_latent_factors,name='book_embedding')(book_input)

    book_vec=Flatten(name='FlattenBooks')(book_embedding)

    book_vec=Dropout(dropout)(book_vec)

    sim=dot([user_vec,book_vec],name='Similarity-Dot-Product',axes=1)

    ###Exogenous Features input

    exog_input = Input(shape=(9,),name='exogenous_input',dtype='float64')

    exog_embedding = Embedding(9,20,name='exog_embedding')(exog_input)

    #exog_embedding = Dense(65,activation='relu',name='exog_Dense')(exog_input)

    exog_vec =Flatten(name='FlattenExog')(exog_embedding)   

    ##############

    nn_inp = Add(dtype='float64',name='Combine_inputs')([sim,exog_vec])

    nn_inp=Dense(128,activation='relu')(nn_inp)

    nn_inp=Dropout(dropout)(nn_inp)

    nn_inp=Dense(64,activation='relu')(nn_inp)

    nn_inp=BatchNormalization()(nn_inp)

    nn_output=Dense(1,activation='relu')(nn_inp)

    nn_model =Model([user_input, book_input,exog_input],nn_output)

    return nn_model

    
nn_model = build_model(0.4,65)

nn_model.summary()
nn_model.compile(optimizer=Adam(lr=1e-4),loss='mse')

batch_size=128

epochs=15

History = nn_model.fit([X_train.ID,X_train.Book_Id,exogenous_train],y_train, batch_size=batch_size,

                              epochs =epochs, validation_data = ([X_test.ID,X_test.Book_Id,exogenous_valid],y_test),

                              verbose = 1)
plt.plot(History.history['loss'] , 'g')

plt.plot(History.history['val_loss'] , 'b')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.grid(True)

plt.show()
preds = nn_model.predict([X_test.ID,X_test.Book_Id,exogenous_valid])

avp = (preds,y_test)

df_id = pd.DataFrame(np.array(X_test.ID))

df_Book_id = pd.DataFrame(np.array(X_test.Book_Id))

df_actual_rating = pd.DataFrame(np.array(y_test))

df_preds = pd.DataFrame(preds)

dfList = [df_id, df_Book_id,df_actual_rating,df_preds]  # List of your dataframes

avp = pd.concat(dfList,ignore_index=True,axis=1)

#new_df = pd.concat([new_df,df_preds],ignore_index=True,axis=1)

avp.rename(columns={ avp.columns[0]: "ID" }, inplace = True)

avp.rename(columns={ avp.columns[1]: "Book_Id" }, inplace = True)

avp.rename(columns={ avp.columns[2]: "Rating_numeric" }, inplace = True)

avp.rename(columns={ avp.columns[3]: "Pred_Rating" }, inplace = True)

avp
avp['Pred_Rating'].max(),avp['Pred_Rating'].min()
def check_overlap(UserId,top_recos_to_check):

    samp_cust = avp[avp['ID']==UserId][['ID','Rating_numeric','Book_Id']]

    samp_cust.sort_values(by='Rating_numeric', ascending=False, inplace=True)

    available_actual_ratings = samp_cust.shape[0]

    rows_to_fetch = min(available_actual_ratings,top_recos_to_check)

    preds_df_sampcust = avp[avp['ID']==UserId][['ID','Pred_Rating','Book_Id']]

    preds_df_sampcust.sort_values(by='Pred_Rating', ascending=False, inplace=True)

    actual_rating = samp_cust.iloc[0:rows_to_fetch,:]

    pred_rating = preds_df_sampcust.iloc[0:rows_to_fetch,:]

    overlap = pd.Series(list(set(actual_rating.Book_Id).intersection(set(pred_rating.Book_Id))))

    pct_overlap = (len(overlap)/rows_to_fetch)*100

    #print("Percentage of overlap in top"+str(top_recos_to_check)+" for User ID - "+str(UserId)+" : "+str(pct_overlap))

    return pct_overlap
test_user_list = avp.ID.unique().tolist()

overlap_summary={}

top_recos_to_check =10

for users in test_user_list:

    overlap_summary[users]= check_overlap(users,top_recos_to_check)
sorted_summary = sorted(overlap_summary.items(), key=lambda x: x[1], reverse=True)

max_overlap = np.array(list(overlap_summary.values())).max()

min_overlap = np.array(list(overlap_summary.values())).min()

mean_overlap = np.array(list(overlap_summary.values())).mean()

print("Max overlap in top" +str(top_recos_to_check)+ " books "+str(max_overlap))

print("Min overlap in top "+str(top_recos_to_check)+ " books "+str(min_overlap))

print("Average overlap in top "+str(top_recos_to_check)+ " books "+str(mean_overlap))