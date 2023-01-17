#################################### AllState Claims Severity ############################################################



# Below is function to encode categorical variables with high cardinality into numeric values such that they can 

# used in modeling exercises. The technique has been inspired from Owen Zhang's method of dealing with categorical variables

# with high cardinality





# Reading in training and test data



import pandas as pd

import numpy as np

import matplotlib as plt

%pylab inline

df_train = pd.read_csv("../input/train.csv", index_col='id')

df_test = pd.read_csv("../input/test.csv", index_col='id')
# Getting all continuous features into a separate dataset



contfeatures = df_train.select_dtypes(include=["float64"])
# Getting all categorical features into a separate dataset

catfeatures = df_train.select_dtypes(include=["object"])
catfeatures_list = list(catfeatures)
# We can possibly feed categorical variables with less or eq 10 levels direclty into our model.

# But, cat variables with >10 levels have to be feature engineered so that their effects can be included into the model

catvarbs_10 = list((df_train[catfeatures_list].apply(pd.Series.nunique)>10))



catvarlist = []

for (i, v) in zip(catfeatures_list, catvarbs_10):

    if(v):

        catvarlist.append(i)
print(catvarlist)
# WE append 'loss' variable to the cat varb dataset to compute means and variance



catvarlist.append('loss')

df_cat_encod = df_train[catvarlist]

df_cat_encod.head(5)
#before running our function to encode, we need to ensure that the list of char variables which we pass to the function

#does not the 'loss' variable in it



catvarlist.remove('loss')

catvarlist

target=['loss']
df_cat_encod.head(5)
# We define a function which will flatten a multi index column names which are created after aggregation of data

# This will be useful after creating mean & standard dev of categorical variable levels





def flattenHierarchicalCol(col,sep = ','):

    if not type(col) is tuple:

        return col

    else:

        new_col = ''

        for leveli,level in enumerate(col):

            if not level == '':

                if not leveli == 0:

                    new_col += sep

                new_col += level

        return new_col
# The function below computes the mean and std dev of the target variable across each level of each categorical variable

# identified and creates two separate features. This can instead be used as a continuous feature in any models we build

# We add the std dev too so as to introduce some random variation/noise into the data

def cat_encoding(list, dataframe, target):

    for i in range(len(list)):

        group_df = dataframe.groupby([list[i]], as_index=False).agg({target:{"mean"+list[i]:'mean', 

                                                                    "stdev"+list[i]:'std'}})

        dataframe = pd.merge(dataframe, group_df, on=list[i], how='left')

    

    dataframe.columns = dataframe.columns.map(flattenHierarchicalCol)

    return dataframe
cat_encoded = cat_encoding(catvarlist,df_cat_encod,target[0])
cat_encoded.head(5)



# Mean and std dev of all categorical variables identified have been computed and returned as a separate dataset which can be joined

# to our original training set. The same mean & std dev values can be used to transform the same variables in the test set
names = cat_encoded.columns

names
del cat_encoded['loss']
# Removing the word 'loss' from the left of the newly created columns



cat_encoded.rename(columns = lambda x: x.replace('loss,',''), inplace=True)
cat_encoded.columns
# Taking the same categorical variables we encoded in train set from test set



cat_encod_test = df_test[catvarlist]

cat_encod_test.head(5)
cat_encod_test = cat_encod_test.reset_index()
del cat_encod_test['id']
cat_encoded.head(5)
cat_encoded2 = cat_encoded
cat_encoded2 = cat_encoded2.drop(cat_encoded2[catvarlist],axis=1)
cat_encoded2.head(5)
onlystdev = cat_encoded2.filter(like='stdev', axis=1)

onlystdev.head(5)
stdev_names  = onlystdev.columns
onlymean = cat_encoded2.filter(like='mean', axis=1)

mean_names = onlymean.columns

mean_names
stdev_names.sort

mean_names.sort
# Getting a dictionary based on training set encoding and mapping the same encoding to our test dataset





for i in range(len(catvarlist)):

    mydict = dict(zip(cat_encoded[catvarlist[i]], cat_encoded[mean_names[i]]))

    cat_encod_test[mean_names[i]] = cat_encod_test[catvarlist[i]].map(mydict)

    mydict2 = dict(zip(cat_encoded[catvarlist[i]], cat_encoded[stdev_names[i]]))

    cat_encod_test[stdev_names[i]] = cat_encod_test[catvarlist[i]].map(mydict2)
cat_encod_test.head(5)