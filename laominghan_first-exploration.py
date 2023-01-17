# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv("../input/train/train.csv")

# Any results you write to the current directory are saved as output.
#Pictures are not considered



#Size and Type of our DF. Seems like only the Name column has some missing values

train.info()
#Lets start having a look at the training data

train.head(5)
train.tail(5)

# Name does not only have Null values but also "NaN" string and sometimes multiple names.

#Age of 60 seems very odd for a cat
#What are the value counts for the numeric columns 

for column in train.select_dtypes(exclude=["object"]):

    print(column)

    print(train[column].value_counts())

    #Slightly more dogs then cats

    #Age column contains many unexpected values - further investigation is needed    

    #a couple of Breeds are dominant in the dataset and some are very rare

    # ~70 percent contain only one animal

    # ~70 percent of the animal are fee free, some are expensive (up to 2000/3000)

    # nearly no videos are provided

    #allmost every add has at least one picture (0: 341)
train.pivot_table(values='AdoptionSpeed',columns='Type').plot.bar()
train["Description"].fillna("", inplace = True)

train["Desc_Len"] = train.apply(lambda x: len(x["Description"]), axis = 1)

train["No_Name"] = np.where(train['Name']=='No Name', 1, 0)

train.drop("AdoptionSpeed", axis=1).select_dtypes(exclude=["object"]).apply(lambda x: x.corr(train.AdoptionSpeed)).sort_values(ascending = False)