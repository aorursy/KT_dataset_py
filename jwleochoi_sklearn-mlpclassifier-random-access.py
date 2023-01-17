from __future__ import print_function



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import io

import bson

import matplotlib.pyplot as plt

from skimage.data import imread   # or, whatever image library you prefer

from sklearn.cross_validation import train_test_split

import random

from sklearn.externals import joblib

from sklearn import neural_network
def extract_categories_df(num_images):

    img_category = list()

    item_locs_list = list()

    items_len_list = list()

    pic_ind_list = list()

    prod_id_list = list()



    with open('../cdisount/train.bson', 'rb') as f:

        data = bson.decode_file_iter(f)

        last_item_loc = 0

        item_len = 0

        for c, d in enumerate(data):

            loc = f.tell()

            item_len = loc - last_item_loc

            category_id = d['category_id']

            prod_id = d["_id"]



            for e, pic in enumerate(d['imgs']):

                prod_id_list.append(prod_id)

                img_category.append(category_id)

                item_locs_list.append(last_item_loc)

                items_len_list.append(item_len)

                pic_ind_list.append(e)

                

                if num_images is not None:

                    if len(img_category) >= num_images:

                        break

            

            last_item_loc = loc

            

            if num_images is not None:

                if len(img_category) >= num_images:

                    break

    

    f.close()

    df_dict = {

        'category': img_category,

        "prod_id": prod_id_list,

        "img_id": range(len(img_category)),

        "item_loc": item_locs_list,

        "item_len": items_len_list,

        "pic_ind": pic_ind_list

    }

    df = pd.DataFrame(df_dict)

    df.to_csv("all_images_categories.csv", index=False)

        

    return df



def get_image(image_id,data_df,fh):

    img_info = data_df[data_df["img_id"] == image_id]

    item_loc = img_info["item_loc"].values[0]

    item_len = img_info["item_len"].values[0]

    pic_ind = img_info["pic_ind"].values[0]

    fh.seek(item_loc)

    item_data = fh.read(item_len)

    d = bson.BSON.decode(item_data)

    

    picture = imread(io.BytesIO(d["imgs"][pic_ind]['picture']))

    return picture
cat_df = extract_categories_df(None)
train_fh = open('../cdisount/train.bson', 'rb')
X_images = np.zeros((10000,180,180,3)) #m images are 180 by 180 by 3

X_ids = []

Y_prod_ids = []
# extract 10000 images for training.

num = 0

for i in random.sample(range(len(cat_df)),10000):

    pic = get_image(i,cat_df,train_fh)

    X_images[num] = pic

    X_ids.append(cat_df['prod_id'][i])

    Y_prod_ids.append(cat_df['category'][i])

    num = num + 1

    

    #show update every 1,000 images

    if num > 0 and num % 1000 == 0:

        print("[INFO] processed {}/{}".format(num, 10000))
# get number of unique categories

Y_df = pd.DataFrame({'product_id':Y_prod_ids})

len(Y_df["product_id"].unique())
# flatten images

X_images = X_images.reshape(X_images.shape[0], -1)

X_images = X_images/255
# partition the data into training and testing splits, using 80%

# of the data for training and the remaining 20% for testing

(X_train, X_test, Y_train, Y_test) = train_test_split(X_images, Y_prod_ids, test_size=0.20, random_state=42)
# Now, your classifier is 'Artificial Neural Network of 10 layers'

# solver: one of lbfgs(quasi-Newton method), sgd (stochastic gradient descent), adam (a special stochasitc gradient decent); default is adam

# alpha: L2 penalty(regularization term) parameter

# hidden_layer_sizes: a tuple; the ith element represents the number of neurons in the ith hidden layer

# activation: one of identity, logistic, tanh, relu; relu is default

clf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1).fit(X_train, Y_train)

print("Artificial Neural Network(ANN)-10")

print(clf.score(X_test, Y_test))
# save the model to disk

joblib.dump(clf, '/ANN_model.sav')

 

# some time later...

 

# load the model from disk

# loaded_model = joblib.load(filename)

# result = loaded_model.score(X_test, Y_test)

# print(result)
trial = 1



while trial < 11:

  

  print('Starting trial', trial)

  X_images = []

  X_images = np.zeros((10000,180,180,3)) #m images are 180 by 180 by 3

  X_ids = []

  Y_prod_ids = []



  num = 0

  for i in random.sample(range(len(cat_df)),10000):

    pic = get_image(i,cat_df,train_fh)

    X_images[num] = pic

    X_ids.append(cat_df['prod_id'][i])

    Y_prod_ids.append(cat_df['category'][i])

    num = num + 1

    

    #show update every 2,500 images

    if num > 0 and num % 2500 == 0:

      print("[INFO] processed {}/{}".format(num, 10000))



  # flatten images

  X_images = X_images.reshape(X_images.shape[0], -1)

  X_images = X_images/255

  

  # partition the data into training and testing splits, using 90%

  # of the data for training and the remaining 10% for testing

  (X_train, X_test, Y_train, Y_test) = train_test_split(X_images, Y_prod_ids, test_size=0.10)

 



  loaded_model = joblib.load('../cdiscount/ANN_model.sav')

  clf = loaded_model.fit(X_train, Y_train)

  print("Artificial Neural Network(ANN)-10")

  print(clf.score(X_test, Y_test))

  joblib.dump(clf, '../cdiscount/ANN_model.sav')

  

  print('trial', trial, 'completed')

  trial = trial + 1
def extract_test_df(num_images):

    prod_id_list = list()

    item_locs_list = list()

    items_len_list = list()

    pic_ind_list = list()



    with open('../cdiscount/test.bson', 'rb') as f:

        data = bson.decode_file_iter(f)

        last_item_loc = 0

        item_len = 0

        for c, d in enumerate(data):

            loc = f.tell()

            item_len = loc - last_item_loc

            prod_id = d["_id"]



            for e, pic in enumerate(d['imgs']):

                prod_id_list.append(prod_id)

                item_locs_list.append(last_item_loc)

                items_len_list.append(item_len)

                pic_ind_list.append(e)

                

                if num_images is not None:

                    if len(prod_id) >= num_images:

                        break

            

            last_item_loc = loc

            

            if num_images is not None:

                if len(prod_id) >= num_images:

                    break

    

    f.close()

    df_dict = {

        'prod_id': prod_id_list,

        "img_id": range(len(prod_id_list)),

        "item_loc": item_locs_list,

        "item_len": items_len_list,

        "pic_ind": pic_ind_list

    }

    df = pd.DataFrame(df_dict)

    df.to_csv("../cdiscount/all_test_images_categories.csv", index=False)

        

    return df
test_cat_df = extract_test_df(None) # pd.read_csv("../cdiscount/all_test_images_categories.csv")
test_fh = open('../cdiscount/test.bson', 'rb')
test_cat_df.head()
len(test_cat_df)
test_cat_df.drop_duplicates(subset='prod_id', inplace=True)

len(test_cat_df)
test_cat_df.head()
test_cat_df = test_cat_df.drop('img_id', 1)
test_cat_df.head()
new_entry = list(range(1768182)) 

test_cat_df = test_cat_df.assign(img_id = new_entry)
test_cat_df = test_cat_df[['img_id', 'item_len', 'item_loc', 'pic_ind', 'prod_id']]
test_cat_df.head()
test_cat_df.to_csv("../cdiscount/all_test_images_categories.csv", index=False)
mul = 20000

phase = 1



category_id = []

loaded_model = joblib.load('../cdiscount/ANN_model.sav')



while mul < 1768182:



  print('Starting prediction', phase, '/ 89')

  X_images = []

  clf = []

  X_images = np.zeros((20000,180,180,3)) #m images are 180 by 180 by 3

  

  i = 0

  num = 0

  for i in range(mul-20000, mul):

    pic = get_image(i,test_cat_df,test_fh)

    X_images[num] = pic

    num = num + 1

    

  # flatten images

  X_images = X_images.reshape(X_images.shape[0], -1)

  X_images = X_images/255

  clf = loaded_model.predict(X_images)

  

  numb = 0

  while numb < len(clf):

    category_id.append(clf[numb])

    numb = numb + 1

  

  phase = phase + 1

  mul = mul + 20000



  

print('Starting prediction', phase, '/ 89')

X_images = []

clf = []

X_images = np.zeros((8182,180,180,3)) #m images are 180 by 180 by 3



i = 0

num = 0

for i in range(mul-20000, 1768182):

  pic = get_image(i,test_cat_df,test_fh)

  X_images[num] = pic

  num = num + 1

  

# flatten images

X_images = X_images.reshape(X_images.shape[0], -1)

X_images = X_images/255

  

clf = loaded_model.predict(X_images)

  

numb = 0

while numb < len(clf):

  category_id.append(clf[numb])

  numb = numb + 1
_id = test_cat_df['prod_id']
submission_df = test_cat_df.assign(category_id = category_id)
submission_df = submission_df.drop('img_id', 1)

submission_df = submission_df.drop('item_len', 1)

submission_df = submission_df.drop('item_loc', 1)

submission_df = submission_df.drop('pic_ind', 1)

submission_df.columns = ['_id', 'category_id']

submission_df.head()
submission_df.to_csv("../cdiscount/my_submission.csv.gz", compression="gzip", index=False)