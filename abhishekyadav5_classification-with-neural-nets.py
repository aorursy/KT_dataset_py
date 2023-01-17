import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, MinMaxScaler
train = pd.read_csv("../input/train_LZdllcl.csv")
train.head()
train.shape
train.info()
train.isnull().sum()
# First, let's look at the distribution of is_promoted



train["is_promoted"].value_counts()
# employee_id

# This doesn't need to have any direct effect on promotion of employees so we ae going to simply drop this column.



train = train.drop(["employee_id"], axis=1)
# department



train["department"].value_counts()
# There are only 9 values in this column. We can easily label encode it later but first, we have to take a look on the distribution of is_promoted in different departments.



department_df = train.groupby(["department", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
department_df
department_df["1_per"] = (department_df[1]/(department_df[0] + department_df[1]))*100
department_df
# We will keep this column for future use.
# region



train["region"].value_counts()
# In total, there are 34 regions of employment, from region_1 to region_34. Let's see if any region is getting preference in promotion or not.



region_df = train.groupby(["region", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
region_df
region_df["1_per"] = (region_df[1]/(region_df[0] + region_df[1]))*100
region_df
# It has much bigger 1_per window than department column, the lowest being 1.9% for region_9 and highest being 14.5% for region_4. This tells us that this column has a significant impact on the target variable.
train["education"].isnull().sum()
train["education"].value_counts()
# There are only 3 values in this column but the important thing is how are we going to handle the NaN values.

# There are about 55000 total observations out of which 2400 have missing values. This is about 4.5% of total observations. Therefore, it is not wise to just throw the Nan rows away. We have to find another way.

# Other thing we can do is that we can treat the Nan values as other categorical values but this method is detrimental for final model because the value of Nan rows is too high.

# The most complex but most sensible solution is to make a model and train it on non-Nan rows and then use that model to find the value of the Nan rows. The effect of this method on final model cannot be determined.
# gender



train["gender"].value_counts()
gender_df = train.groupby(["gender", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
gender_df
gender_df["1_per"] = (gender_df[1]/(gender_df[0] + gender_df[1]))*100
gender_df
# The percentage of promotion in both the gender is almost same.
# recruitment_channel



train["recruitment_channel"].value_counts()
# There are only three values. Let's see if any of them has given any priority in promotion.



recruitment_channel_df = train.groupby(["recruitment_channel", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
recruitment_channel_df["1_per"] = (recruitment_channel_df[1]/(recruitment_channel_df[0] + recruitment_channel_df[1]))*100
recruitment_channel_df
# There is very small difference is all three values.
# no_of_trainings



train["no_of_trainings"].value_counts()
training_df = train.groupby(["no_of_trainings", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
training_df["1_per"] = (training_df[1]/(training_df[0] + training_df[1]))*100
training_df
# This distribution shows us a very intersting trend. The more the no. of training completed by employee, the less there chance of getting promoted.

# All of the employee who completed 7 or more training does not got promoted and the employees who does only one training has most percent of promotions.
# age



train["age"].value_counts()
len(train["age"].value_counts())
# There are a total of 41 values in age column.

# Let's make a bar graph to see which age group gets preference, if any.



age_df = train.groupby(["age","is_promoted"])["is_promoted"].count().unstack("is_promoted")
age_df["1_percent"] = (age_df[1]/(age_df[0] + age_df[1]))*100
age_df
age_df["1_percent"].plot(kind="barh", figsize=[7,9])
print(age_df["1_percent"].mean())  #mean of 1_percent column

print(age_df["1_percent"].std())  #standard deviation of 1_percent column
# As we can see from the graph, there is no clear pattern that shows that a specific age group is prioritised.

# We can braodly divide the graph in 3 parts.

# 20-26 = This age group has low chance of promotion

# 27-44 = This is where the chance of promotion is high.

# 45-60 = The chance of promotion is slightly less for this agegroup.
# length_of_service



train["length_of_service"].value_counts()
len(train["length_of_service"].value_counts())
length_df = train.groupby(["length_of_service", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
length_df["1_percent"] = (length_df[1]/(length_df[0] + length_df[1]))*100
length_df
length_df["1_percent"].plot(kind="barh", figsize=[7,10])
# We can divide this graph in 3 parts.

# 1-11 = This part got maximum promotion.

# 12-20 = This part got a little bit less promotion.

# 21-37 = This part shows no clear trend.
# KPI

# First, we will change this column name for ease in analysis.



train = train.rename(columns={"KPIs_met >80%":"KPI"})
train["KPI"].value_counts()
KPI_df = train.groupby(["KPI", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
KPI_df["1_percent"] = (KPI_df[1]/(KPI_df[0] + KPI_df[1])) * 100
KPI_df
# There is a massive difference in this column. This shows that KPI is one of the most important factor in getting promotion. 
# awards_won?



train["awards_won?"].value_counts()
awards_df = train.groupby(["awards_won?", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
awards_df
awards_df["1_percent"] = (awards_df[1]/(awards_df[0] + awards_df[1]))*100
awards_df
# This column is also very influencial in deciding the target variable
# avg_training_score



train["avg_training_score"].value_counts()
print(min(train["avg_training_score"]))  # minimum training score

print(max(train["avg_training_score"]))  # maximum training score
score_df = train.groupby(["avg_training_score", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
score_df["1_percent"] =(score_df[1]/(score_df[0] + score_df[1]))*100
score_df["1_percent"].plot(kind="barh", figsize=[7,15])
score_df
# as you can see, the score is directly proportional to the percent of promotions. This is also one of the most influential criteria.
# previous_year_rating



train["previous_year_rating"].value_counts()
rating_df = train.groupby(["previous_year_rating", "is_promoted"])["is_promoted"].count().unstack("is_promoted")
rating_df["1_percent"] = (rating_df[1]/(rating_df[0] + rating_df[1]))*100
rating_df
# The more the rating, the more the chances of promotion.
train.head()
# let's remove all nan values row and see how many are left.



train_new = train[(train["education"].isnull() == False) & (train["previous_year_rating"].isnull() == False)]
train_new.shape
# label encoder for department and region

# dummy variables for education and recruitment channel

# mapping for gender

# rest all will be mapped between 0 and 1.
# department



le_dep = LabelEncoder()

le_dep.fit(train_new["department"])
encoded_department = le_dep.transform(train_new["department"])
train_new["encoded_department"] = encoded_department
train_new = train_new.drop(["department"], axis=1)
train_new.head()
# region



le_region = LabelEncoder()

le_region.fit(train_new["region"])
encoded_region = le_region.transform(train_new["region"])
train_new["encoded_region"] = encoded_region
train_new = train_new.drop(["region"], axis=1)
train_new.head()
# education



onehot_education = pd.get_dummies(train_new["education"])

train_new = train_new.join(onehot_education)

train_new = train_new.drop(["education"], axis=1)
train_new.head()
# recruitment_channel



train_new["recruitment_channel"].value_counts()
onehot_recruitment = pd.get_dummies(train["recruitment_channel"])

train_new = train_new.join(onehot_recruitment)

train_new = train_new.drop(["recruitment_channel"], axis=1)
train_new.head()
gender_map = {"f":0, "m":1}

train_new["gender"] = train_new["gender"].map(gender_map)
train_new.head()
# no_of_trainings



train_new["no_of_trainings"] = train_new["no_of_trainings"]/10
# age



train_new["age"] = (train_new["age"] - 19)/42
# previous_year_rating



train_new["previous_year_rating"] = train_new["previous_year_rating"]/5
# length_of_service



train_new["length_of_service"] = train_new["length_of_service"]/37
# avg_training_score



train_new["avg_training_score"] = train_new["avg_training_score"]/100
# encoded_department



train_new["encoded_department"] = train_new["encoded_department"]/8
# encoded_region



train_new["encoded_region"] = train_new["encoded_region"]/33
train_new.describe()
train_new.head()
from keras import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.callbacks import ReduceLROnPlateau
x_train = train_new.drop(["is_promoted"], axis=1)

y_train = list(train_new["is_promoted"])
model = Sequential()

model.add(Dense(8, input_dim=16, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(Dense(64, activation="relu"))

model.add(Dense(128, activation="relu"))

model.add(Dense(32, activation="relu"))

model.add(Dense(16, activation="relu"))

model.add(Dense(8, activation="relu"))

model.add(Dense(4, activation="relu"))

model.add(Dense(2, activation="relu"))

model.add(Dense(1, activation="sigmoid"))



model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
model.summary()
reducelr = ReduceLROnPlateau(monitor="acc", factor=0.2, patience=3, min_lr=0.0005, verbose=1)
model.fit(x_train, y_train, batch_size=10, epochs=32, callbacks=[reducelr], validation_split=0.1)