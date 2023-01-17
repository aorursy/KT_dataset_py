import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train_users_2.csv')
train.info()
train.head(20)
train.gender.value_counts()
train.language.value_counts()
# There is an outlier that is disrupting the graph
plt.hist(train.age.dropna(), bins=60)
plt.show()
# A guess is that the Year of when people are born are being entered as "age"
# I will run an anomaly classifier and an age regression 
train.age.value_counts().tail(15)
train.affiliate_provider.value_counts()
for aff in train.affiliate_provider.value_counts().index:
    print(aff)
    print(train.country_destination[train.affiliate_provider == aff].value_counts())
    
# It is not easy to tell what the connection is for affiliate proviers and the country destination, 
# so I guess we will be very blindly feeding the classification algorithm for this kaggle competition . 
train.affiliate_channel.value_counts()
train.signup_flow.value_counts()
dem = pd.read_csv('../input/age_gender_bkts.csv')
coun = pd.read_csv('../input/countries.csv')
sess = pd.read_csv('../input/sessions.csv')
dem.info()
dem.head(20)
dem.year.value_counts()
# Year is not useful for t
coun.info()
coun
sess.info()
