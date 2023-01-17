# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#This kernel is an attempt to study womens,s shoes with respect to rating/reviews.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



pd.set_option('display.max_colwidth', -1)



data = pd.read_csv("../input/7210_1.csv")







import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Lets have a look at data

data.head()
#  want to select labels that i am interested only



labels = ['id','brand','dateAdded','dateUpdated' , 'dateUpdated' , 'features' , 'merchants' , 'prices.amountMin' , 'prices.amountMax', 'reviews' , 'prices.color' , 'prices.currency' , 'prices.dateAdded' , 'prices.condition']



data = data[labels]



data.head()
#The idea is to take review column and get rating from it.



pd.set_option('display.max_colwidth', -1)

#Lets take new dataset data_review for analysis of review purpose



data_review = data.copy()



#Also lets remove the NaN values from review 



data_review = data_review.dropna(subset=['reviews'])



data_review = data_review.sort_values(by=['reviews'])



data_review.head(10)

# Now lets split the reviewand take out the rating 



data_review['reviews'] = data_review['reviews'].str.split('"rating":',expand=True)[1].str.split(',"',expand=True, n=1)[0]



data_review['reviews'].unique()



# thus na values are nicely removed
# Now i am interested in brand , price and reviews

Data_analysis = data_review[['reviews','brand' , 'prices.amountMin']]

Data_analysis.head()
# Convert 





Data_analysis = Data_analysis.dropna(subset=['reviews'])



Data_analysis['reviews']

# None alseo has been removed , convert to float type



Data_analysis['reviews'] =Data_analysis['reviews'].astype(float)



Data_analysis['reviews'].dtype
sns.barplot(x='brand',y='reviews',data=Data_analysis)

plt.show()
# This plot doesnt show much. Lets plot a joint plot



sns.jointplot(x='prices.amountMin',y='reviews',data=Data_analysis)

plt.show()



# This plot shows that rating is highest at mid pricess , at very high prices rating is low



sum_df = Data_analysis.groupby(['reviews','brand']).agg({'prices.amountMin': 'sum'})



sum_df
# Lets plot prices with respect to rating



Data_analysis.reset_index().plot(x="reviews",y="prices.amountMin")



Data_analysis.reset_index().plot(x="brand",y="prices.amountMin")
# Conclusion



# At high prices rating is low , mid prices tend to have good rating.



#These graphs are not good for review to price relationship so lets try something else



Data_new = data_review[['reviews','brand' , 'prices.amountMin']]

Data_new = Data_new.dropna(subset=['reviews'])

Data_new['reviews'] =Data_new['reviews'].astype(float)



Data_new.reset_index().plot(x="prices.amountMin",y="reviews")
# Hence conclusion mid prices tend to have good rating low prices and very high prices do not have good rating!



#Thankyou!