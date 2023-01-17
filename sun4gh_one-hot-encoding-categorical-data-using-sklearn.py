# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')





# Any results you write to the current directory are saved as output.
breeds_df = pd.read_csv('../input/breed_labels.csv')



orig_train_df = pd.read_csv('../input/train/train.csv')

test_df = pd.read_csv('../input/test/test.csv')



orig_train_df['dataset_type'] = 'train'

test_df['dataset_type'] = 'test'

all_data_df = pd.concat([orig_train_df, test_df])



print ("all_data_df shape: ", all_data_df.shape)

print ("orig_train_df shape: ", orig_train_df.shape)

print ("test_df shape: ", test_df.shape)

print (list(test_df.columns))

the_test_df_pet_ids = test_df["PetID"]

print(the_test_df_pet_ids.head())
list_of_num_fields = [ 'Age', 'MaturitySize', 'Quantity', 'Fee', 'VideoAmt', 'PhotoAmt', ]

list_of_categ_fields = ['Breed1']

list_to_drop = [ "RescuerID", "Name", "Description", "PetID", 'dataset_type']

print("What the head() of Breeds1 column looks like in the all_data dataframe :",all_data_df["Breed1"].head())



print("\nLength of breeds_df.BreedID: ", len(breeds_df.BreedID))

print("\nWhat the head() Breeds1 lookup table looks like:\n", breeds_df.head())

print("\nWhat the tail() Breeds1 lookup table looks like:\n", breeds_df.tail())

print("\nThe numerical values are of type: ", type(breeds_df.BreedID[3]))



if 307 in list(breeds_df.BreedID):

    print ("Category 307 (Mixed Breed) is in the dictionary, at position: " , 

           list(breeds_df.BreedID).index(307), 

           ". It is wedged between all the dog breeds and all the cat breeds:") 

    print (breeds_df[238:243])
the_one_hot_encodings = pd.get_dummies(all_data_df.Breed1.head(5), prefix = 'Breed1')

print (the_one_hot_encodings.head(5))

print("\nThe columns are: ", the_one_hot_encodings.columns)
all_data_df["Breed1"] = all_data_df["Breed1"].astype('category', categories = list(breeds_df.BreedID) )

the_one_hot_encodings = pd.get_dummies(all_data_df.Breed1.head(), prefix = 'Breed1')

print(the_one_hot_encodings.head(5))

print(the_one_hot_encodings.columns)
all_data_df = pd.concat( [all_data_df, pd.get_dummies(all_data_df.Breed1, prefix = 'Breed1') ] , axis = 1)

print("all_data_df shape: ", all_data_df.shape)

print("Now extract from that the following:")

#Split them back into the original train vs. test data

orig_train_df = all_data_df[all_data_df.dataset_type == "train"]

test_df = all_data_df[all_data_df.dataset_type == "test"]

print("orig_train_df shape:", orig_train_df.shape)

print("test_df shape:", test_df.shape)
from sklearn.model_selection import train_test_split



train_df, val_df = train_test_split(orig_train_df, test_size = .2)



print ("train_df shape:", train_df.shape)

print ("val_df shape:", val_df.shape)
train_targets = train_df["AdoptionSpeed"]

train_df.drop( ["AdoptionSpeed"] + list_to_drop + ["Breed1"], axis =1, inplace = True)

val_targets = val_df["AdoptionSpeed"]

val_df.drop(["AdoptionSpeed"] + list_to_drop + ["Breed1"], axis = 1, inplace = True)



test_df.drop(["AdoptionSpeed"] + list_to_drop + ["Breed1"], axis = 1, inplace = True)



train_df.isna().sum()
print ("\tall_data_df has been: ", all_data_df.shape)

print ("From which we extracted :")

print ("\torig_train_df", orig_train_df.shape)

print ("\ttest_df", test_df.shape)

print ("\nAnd furthermore, we split the orig_train_df into:")

print ("\ttrain_df", train_df.shape)

print ("\tval_df", val_df.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import cohen_kappa_score



clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=22)
clf.fit(train_df, train_targets)
clf.predict(test_df.iloc[3:4])
val_preds = clf.predict(val_df)

cohen_kappa_score(val_preds, val_targets, weights = "quadratic")
test_preds = clf.predict(test_df)
sub_df = pd.read_csv('../input/test/sample_submission.csv')

print(sub_df.PetID.head())

print()

print(the_test_df_pet_ids.head())



print("Are the two series types comparable:", type(sub_df.PetID) == type(the_test_df_pet_ids))

print("The two series are identical: ", sub_df["PetID"].equals(the_test_df_pet_ids))



sub_df['AdoptionSpeed'] = test_preds.astype(int)
sub_df.to_csv('submission.csv', index=False)
from sklearn import tree

import graphviz



outfile = 'ourtree.dot'

dot_data = tree.export_graphviz(clf, out_file=outfile,

                                feature_names=train_df.columns,  

                                class_names=['0','1','2','3','4'],  

                                filled=True, rounded=True, max_depth=4, 

                                special_characters=True) 
#Efforts to create a png

#!ls

#!rm ourtree.png

#!ls

#!dot -Tpng ourtree.dot -o ourtree.png

#!ls

#%matplotlib inline

#import matplotlib.pyplot as plt

#import matplotlib.image as mpimg

#img = mpimg.imread('ourtree.png')

#plt.imshow(img)
dot_data = tree.export_graphviz(clf, out_file=None,

                                feature_names=train_df.columns,  

                                class_names=['0','1','2','3','4'],  

                                filled=True, rounded=True, max_depth=4, 

                                special_characters=True) 



graph = graphviz.Source(dot_data)  # you can't have these two lines when trying out_file (for some reason)

graph 