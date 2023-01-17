import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import math



from sklearn.linear_model import SGDClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVR
# reading in dataframes

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



# reordering survived column to the end in train_df

reordered_columns = [x for x in train_df.columns if x != 'Survived']+['Survived']

train_df = train_df.reindex_axis(reordered_columns, axis=1)



# creating empty survived columns for test_df

test_df["Survived"] = [np.nan for x in test_df.PassengerId]



# merging the 2 dataframes

df = pd.concat([train_df,test_df], ignore_index = True)

df.head()



# storing train_df length

train_df_length = len(train_df)

# Understanding where are the nulls

column_name = []

null_count = []



for x in df.columns:

    null_count_val = df[x].isnull().values.sum()

    column_name.append(x)

    null_count.append(null_count_val)



np.column_stack([column_name, null_count])



null_df = pd.DataFrame(np.column_stack([column_name, null_count]), columns = ["column_name","null_count"])

null_df.sort_values(by = "null_count",axis = 0,ascending = False, inplace = True)

null_df["null_perc_of_total"] = [round(int(x)*100/len(df),2) for x in null_df.null_count]

null_df.head(15)
# Dealing with the null in Fare

print("The entry with missing Fare:\n")

print(df[df["Fare"].isnull().values == 1])

print("\n")



# storing the index of the missing_fare_row

missing_fare_index = df[df["Fare"].isnull().values == 1].index.values

missing_fare_index = missing_fare_index[0]



#   hypothesis --> class has a big impact on fare

g = plt.figure(figsize = (5,4))

g = sns.boxplot(x="Pclass", y="Fare", data=df)

g.set_ylim([0,300])

g.set_title("Fares by Pclass")



# inside Pclass = 3, there is still variation in fares

#    noticing that those with high numbers of SibSp + Parch tend to have higher Fares in Pclass = 3

p_class_3 = df[df["Pclass"]==3].copy()



# fam_size = SibSp + Parch

p_class_3["fam_size"] = p_class_3["SibSp"]+ p_class_3["Parch"]



# graphing fares in Pclass = 3 by fam_size

a = plt.figure(figsize = (15,4))

a = sns.boxplot(x="fam_size", y="Fare", data=p_class_3)

a.set_ylim([0,100])

a.set_title("Fares by fam_size in Pclass = 3")
# finding the median Fare at Pclass = 3 and fam_size = 0

fam_size_0 = p_class_3[(p_class_3["fam_size"]==0) &(p_class_3["Fare"].isnull().values != 1)]

missing_fare = fam_size_0["Fare"].quantile(0.5)





#inserting the value in df

df.set_value(missing_fare_index,"Fare",missing_fare)

print("{}{}".format('The fare we will be inserting in is:',missing_fare))
# Dealing with nulls in Embarked

embarked_null = df[df["Embarked"].isnull().values == 1]



# the index of the entries with missing Embarked

embarked_null_index = embarked_null.index.values



# boxplot of fares by embarked

g = sns.boxplot(x = "Embarked",y = "Fare", data = df)

g.set_ylim([0,200])

g.set_title("Fares by Embarked")



# the missing values have Fare = 80

#    this is far above embarked in (S,Q)

#    hence we will allocate their Embarked value as 'C'

for x in embarked_null_index:

    df.set_value(x, "Embarked", 'C')

     
common_titles = ["Mr","Miss","Mrs","Master"]



# adding the title

title = []



for x in df.Name:

    splitting_by_comma = x.split(',',1)

    post_comma_field = splitting_by_comma[1]

    splitting_by_fullstop = post_comma_field.split('.',1)

    untrimmed_title =splitting_by_fullstop[0]

    title_val = untrimmed_title.strip()

    

    if title_val in common_titles:

        title_val_modified = title_val

    elif title_val == 'Mlle':

        title_val_modified = 'Miss'

    elif title_val == 'Mme':

        title_val_modified = 'Mrs'

    else:

        title_val_modified = 'Rare'

    title.append(title_val_modified)



df["title"] = title



# adding has_brackets

has_brackets = []



for x in df.Name:

    left_bracket_count = x.count('(')

    right_bracket_count = x.count(')')

    bracket_count = left_bracket_count + right_bracket_count

    if bracket_count > 0:

        val = True

    else:

        val = False

    has_brackets.append(val)



df["has_brackets"] = has_brackets



# adding has_quotes

has_quotes = []



for x in df.Name:

    quote_count = 0

    quote_count = x.count('"')

    if quote_count > 0:

        val = True

    else:

        val = False

    has_quotes.append(val)



df["has_quotes"] = has_quotes



df["fam_size"] = df["SibSp"] + df["Parch"]

# investigating how useful the added dimensions have with predicting survival



survival_by_brackets = pd.crosstab(index = df.has_brackets,columns = df.Survived)

survival_by_brackets.rename(columns = {0.0: 'did_not_survive'

                                      ,1.0: 'survived'}, inplace = True)

survival_by_brackets["survival_rate"] = survival_by_brackets.survived/(survival_by_brackets.survived + survival_by_brackets.did_not_survive)



print("survival_by_brackets\n")

print(survival_by_brackets)

print("\n")



survival_by_quotes = pd.crosstab(index = df.has_quotes, columns = df.Survived)

survival_by_quotes.rename(columns = {0.0: 'did_not_survive'

                                    ,1.0: 'survived'}, inplace = True)



survival_by_quotes["survival_rate"] = survival_by_quotes.survived/(survival_by_quotes.survived + survival_by_quotes.did_not_survive)

print("survival_by_quotes\n")

print(survival_by_quotes)





# Note

# those with brackets and quotes in their names are about twice as much likely to survive
# Deriving information from tickets



aggregations = {'PassengerId':'count'

                , 'Fare':['max','min',np.mean]

                , 'Pclass': 'max'}

non_distinct_tickets = df[["PassengerId","Ticket", "Fare", "Pclass"]].groupby(["Ticket"]).agg(aggregations)

non_distinct_tickets.reset_index(inplace = True)

non_distinct_tickets.columns = non_distinct_tickets.columns.droplevel()

non_distinct_tickets.columns = ['ticket','max_fare','min_fare','mean_fare','Pclass','ticket_group_size']

non_distinct_tickets.sort_values(by = 'ticket_group_size', ascending = False, inplace = True)





non_distinct_tickets['actual_fare'] = non_distinct_tickets['mean_fare']/non_distinct_tickets['ticket_group_size']





# tack onto the df dataframe the count_passengers and actual fare

df_w_actual_fare = df.copy()

df_w_actual_fare = df_w_actual_fare.merge(non_distinct_tickets[['ticket','ticket_group_size','actual_fare']]

                                         , how = 'outer'

                                         , left_on = 'Ticket'

                                         , right_on = 'ticket')



df_w_actual_fare = df_w_actual_fare.drop('ticket',axis = 1)

df_w_actual_fare.head()





# what is the distribution of ticket_group_size

ticket_group_size_distribution = df_w_actual_fare[['ticket_group_size','PassengerId']].groupby(['ticket_group_size']).agg('count')

ticket_group_size_distribution.reset_index(inplace = True)

ticket_group_size_distribution.rename(columns = {'PassengerId':'count_of_passengers'}, inplace = True)

print('Count of passengers by ticket group size')

print(ticket_group_size_distribution.head(10))





# does ticket_group_size affect survival rate



ticket_group_size_no_nulls = df_w_actual_fare[df_w_actual_fare['Survived'].isnull().values == 0]

aggregations = {'PassengerId':'count'

                ,'Survived':'sum'}

ticket_group_size_survival_rate = ticket_group_size_no_nulls[['ticket_group_size','PassengerId','Survived']].groupby(['ticket_group_size']).agg(aggregations)

ticket_group_size_survival_rate.reset_index(inplace = True)

ticket_group_size_survival_rate.rename(columns = {'Survived':'count_survived'

                                                 ,'PassengerId':'count_passengers'}, inplace = True)

ticket_group_size_survival_rate['survival_rate'] = ticket_group_size_survival_rate['count_survived']/ticket_group_size_survival_rate['count_passengers']



print('\n')

print('Survival rate by ticket group size')

print(ticket_group_size_survival_rate.head(10))



# -------------------------------

# ------------NOTE------------

# # -------------------------------

# Split into buckets

#     ticket_group_size in (1,2,3,4) --> split into individual buckets (due to decent group sizes and distinct survival rates)

#     ticket_group_size > 5 --> put into one bucket (due to small group sizes and dramatic drop in survival rate)

# -------------------------------
# creating the ticket_group_size buckets

df_w_actual_fare.head()



ticket_group_bucket = []

for x in df_w_actual_fare.ticket_group_size:

    if x <=4:

        val = x

    else:

        val = 5

    ticket_group_bucket.append(val)

df_w_actual_fare["ticket_group_bucket"] = ticket_group_bucket



# how does the distribution of actual fare look?

ax = plt.figure(figsize = (3,3))

ax = sns.distplot(df_w_actual_fare.actual_fare, kde=False)
# group fare into buckets of 10 and calculate survival rate

fare_df = df_w_actual_fare.copy()



bucket_size = 10



start_fare_bucket = []

for x in fare_df.actual_fare:

    val_start = math.floor(x/bucket_size)

    if val_start >= 5:

        val = 5

    else:

        val = val_start

    start_fare_bucket.append(val)

    

fare_df['start_fare_bucket'] = start_fare_bucket

fare_df_no_nulls = fare_df[fare_df['Survived'].isnull().values == 0]

aggregations = {'PassengerId':'count'

               ,'Survived':'sum'}

fare_survival_rate = fare_df[['start_fare_bucket','PassengerId','Survived']].groupby(['start_fare_bucket']).agg(aggregations)

fare_survival_rate.reset_index(inplace = True)

fare_survival_rate.rename(columns = {'Survived':'count_survived'

                                    ,'PassengerId':'count_passengers'}, inplace = True)

fare_survival_rate['survival_rate'] = fare_survival_rate['count_survived']/fare_survival_rate['count_passengers']



print('\n')

print('Survival rate by start_fare_bucket')

print(fare_survival_rate.head(10))



# -------------------------------

# ------------NOTE------------

# # -------------------------------

# Split into buckets

#     if start_fare_bucket in (0,1) --> split into individual buckets

#     if start_fare_bucket > 1 --> put into one_bucket

# -------------------------------



fare_bucket = []

for x in fare_df.start_fare_bucket:

    if x <= 1:

        val = x

    else:

        val = 2

    fare_bucket.append(val)

fare_df['fare_bucket'] = fare_bucket



# attach fare_bucket to df_w_actual_fare



df_v2 = df_w_actual_fare.copy()

df_v2 = df_v2.merge(fare_df[['PassengerId','fare_bucket']]

                   , how = 'left'

                   , on = 'PassengerId')
# having cabin number vs not having

cabin_df = df_v2.copy()

cabin_df['has_cabin'] = [1 if pd.isnull(x) == False else 0 for x in cabin_df.Cabin]



cabin_df_count = cabin_df[['has_cabin','PassengerId']].groupby(['has_cabin']).agg('count')

cabin_df_count.reset_index(inplace = True)



print('count of passengers by has_cabin')

print(cabin_df_count.head())



cabin_df_no_nulls = cabin_df[cabin_df['Survived'].isnull().values == 0]



aggregations = {'PassengerId':'count'

               ,'Survived':'sum'}

cabin_survival_rate = cabin_df_no_nulls[['has_cabin','Survived','PassengerId']].groupby(['has_cabin']).agg(aggregations)

cabin_survival_rate.reset_index(inplace = True)

cabin_survival_rate.rename(columns = {'Survived':'count_survived'

                                     ,'PassengerId':'count_passengers'}, inplace = True)

cabin_survival_rate['survival_rate'] = cabin_survival_rate['count_survived']/cabin_survival_rate['count_passengers']





print('\n')

print('survival rate by has_cabin')

print(cabin_survival_rate.head())





# -------------------------------

# ------------NOTE------------

# -------------------------------

# looks like having a none null field in Cabin more than doubles your chances of survival

# -------------------------------

# CREATING THE CABIN FEATURE SET 





# grab all rows with values in Cabin column

cabin_not_null_df = df[df["Cabin"].isnull().values == 0].copy()



# count survived in each cabin excluding the current person

# join by cabin_id where passenger_id is not equal

# count survived group (total, males, females)





# joining the rows with not null cabin values back onto itself

cabin_not_null_df_v1 = cabin_not_null_df.copy()

    # indicator for survived is null

cabin_not_null_df_v1["survived_is_null"] = [1 if math.isnan(x) == 1 else 0 for x in cabin_not_null_df_v1.Survived]

    # indicator for survived is 1

cabin_not_null_df_v1["survived_true"] = [1 if x == 1 else 0 for x in cabin_not_null_df_v1.Survived]    

    # indicator for survived is 0

cabin_not_null_df_v1["survived_false"] = [1 if x == 0 else 0 for x in cabin_not_null_df_v1.Survived]



cabin_all = cabin_not_null_df.merge(cabin_not_null_df_v1, how = 'left', on = 'Cabin')



# removing rows where it is the same person

cabin_all = cabin_all[cabin_all["PassengerId_x"] != cabin_all["PassengerId_y"]]



# group by and counting

aggregations = {"PassengerId_y":'count',

                "survived_is_null":'sum',

                "survived_true":'sum',

                "survived_false":'sum'}

cabin_all_grouped = cabin_all[["PassengerId_x",

                               "Sex_y",

                               "Survived_y",

                               "PassengerId_y",

                               "survived_is_null",

                               "survived_true",

                               "survived_false"]].groupby(["PassengerId_x",

                                                           "Sex_y"]).agg(aggregations)



cabin_all_grouped.reset_index(inplace = True)

cabin_all_grouped.columns = ["PassengerId","Sex","count_other_passengers","survived_true","survived_is_null","survived_false"]

cabin_females =  cabin_all_grouped[cabin_all_grouped["Sex"]== 'female']

cabin_males = cabin_all_grouped[cabin_all_grouped["Sex"]== 'male']





cabin_passengers = pd.DataFrame(cabin_all_grouped["PassengerId"].unique(), columns = ["PassengerId"])



cabin_with_female = cabin_passengers.merge(cabin_females, how = 'left',on = 'PassengerId')

cabin_all = cabin_with_female.merge(cabin_males, how = 'left', on = "PassengerId")

cabin_all.fillna(value = 0, inplace = True)



cabin_all.drop(labels = ["Sex_x","Sex_y"], axis = 1, inplace = True)

cabin_all.columns = ["PassengerId",

                     "count_female_cabinmates",

                     "count_female_survived",

                     "count_female_unknown",

                     "count_female_died",

                     "count_male_cabinmates",

                     "count_male_survived",

                     "count_male_unknown",

                     "count_male_died"]



cabin_all["cabin_mate_count"] = cabin_all["count_female_cabinmates"] + cabin_all["count_male_cabinmates"]

cabin_all["cabin_total_survived"] = cabin_all["count_female_survived"] + cabin_all["count_male_survived"]

cabin_all["cabin_total_died"] = cabin_all["count_female_died"] + cabin_all["count_male_died"]

cabin_all["cabin_total_unknown"] = cabin_all["count_female_unknown"] + cabin_all["count_male_unknown"]



cabin_all.head()
# adding in has_cabin and cabin_all to df_v2



df_v2.head()



df_v2['has_cabin'] = [1 if pd.isnull(x)==False else 0 for x in df_v2.Cabin]



# attaching cabin_all

df_v3 = df_v2.copy()

df_v3 = df_v3.merge(cabin_all

                   ,how = 'left'

                   ,on = 'PassengerId')



# fillna in the cabin_all columns

columns_to_fill_na = np.array(cabin_all.columns[1:])

df_v3[columns_to_fill_na] = df_v3[columns_to_fill_na].fillna(value = 0)

# INVESTIGATING AGE



adding_age = df.copy()

adding_age_has_age = adding_age[adding_age["Age"].isnull().values!=1]

adding_age_has_age.head()



adding_age_has_age["age_rounded"] = [round(x) for x in adding_age_has_age.Age]



buckets_of = 10

adding_age_has_age["bucket"] = [math.floor(x/buckets_of) for x in adding_age_has_age.age_rounded]



aggregations = {'PassengerId':'count','Survived':'sum'}

grouped_by_bucket = adding_age_has_age[["bucket","Sex","PassengerId","Survived"]].groupby(["bucket","Sex"]).agg(aggregations)

grouped_by_bucket.reset_index(inplace = True)

grouped_by_bucket.rename(columns = {'bucket':'age_bucket',

                                    'PassengerId':'total_passengers'}, inplace = True)



grouped_by_bucket["survival_perc"] = [round(grouped_by_bucket["Survived"][x]/grouped_by_bucket["total_passengers"][x],2) 

                                      for x in grouped_by_bucket.index.values]

ax = sns.barplot(x="age_bucket", y="survival_perc", hue="Sex", data=grouped_by_bucket)

ax.set_xlabel("{}{}".format("age_grouped_into_buckets_of_",buckets_of))
adding_age_has_age["age_rounded"] = [round(x) for x in adding_age_has_age.Age]



buckets_of = 5

adding_age_has_age["bucket"] = [math.floor(x/buckets_of) for x in adding_age_has_age.age_rounded]



aggregations = {'PassengerId':'count','Survived':'sum'}

grouped_by_bucket = adding_age_has_age[["bucket","Sex","PassengerId","Survived"]].groupby(["bucket","Sex"]).agg(aggregations)

grouped_by_bucket.reset_index(inplace = True)

grouped_by_bucket.rename(columns = {'bucket':'age_bucket',

                                    'PassengerId':'total_passengers'}, inplace = True)



grouped_by_bucket["survival_perc"] = [round(grouped_by_bucket["Survived"][x]/grouped_by_bucket["total_passengers"][x],2) 

                                      for x in grouped_by_bucket.index.values]

                            



# plt.figure(figsize = (10,8))

# plt.bar(grouped_by_bucket.index.values, grouped_by_bucket.survival_perc)

# plt.show()

ax = sns.barplot(x="age_bucket", y="survival_perc", hue="Sex", data=grouped_by_bucket)

ax.set_xlabel("{}{}".format("age_grouped_into_buckets_of_",buckets_of))

total = sns.barplot(x="age_bucket", y="total_passengers", hue="Sex", data=grouped_by_bucket)

total.set_xlabel("{}{}".format("age_grouped_into_buckets_of_",buckets_of))
# PUTTING AGE INTO BUCKETS



# creating the predictor variables ____________________________________________________________



# putting age into buckets of 10 (we can see significant variations in buckets of 10), group those >= 50 into separate bucket



age_bucket_10 = []



for x in range(0, len(df_v3)):

    

    age_is_null = math.isnan(df_v3["Age"][x])

    

    if age_is_null == False and df_v3["Age"][x] < 50:

        val = math.floor(df_v3["Age"][x]/10)

    elif age_is_null == False:

        val = math.floor(50/10)

    else:

        val = None

        

    age_bucket_10.append(val)

    

df_v3["age_bucket_10"] = age_bucket_10

    

    

# putting males under 25 into buckets of 5, with the rest bucketed into another group

#  (as we can see significant variation in survival rates for males under 25)   

   

age_bucket_m5 = []



for x in range(0, len(df_v3)):

    

    age_is_null = math.isnan(df_v3["Age"][x])

    

    if age_is_null == False and df_v3["Sex"][x] == "male" and df_v3["Age"][x] < 25:

        value = math.floor(df_v3["Age"][x]/5)

    elif age_is_null == False:

        value = 5

    else:

        value = None

        

    age_bucket_m5.append(value)

    

df_v3["age_bucket_m5"] = age_bucket_m5
# what are the most important factors for determining age_bucket_10 & age_bucket_m5



# cleaning up the dataset

columns_to_keep = ['Pclass','Sex','SibSp','Parch','Embarked','title','has_brackets','has_quotes'

                  ,'fam_size','ticket_group_bucket','fare_bucket','has_cabin'

                  ,'count_female_cabinmates','count_female_survived','count_female_unknown','count_female_died'

                  ,'count_male_cabinmates','count_male_survived','count_male_unknown','count_male_died'

                  ,'cabin_mate_count','cabin_total_survived','cabin_total_died','cabin_total_unknown'

                  ,'age_bucket_10','age_bucket_m5', 'Survived','PassengerId']



filtered_df = df_v3[columns_to_keep].copy()



# transforming string to int

filtered_df['Sex'] = filtered_df['Sex'].astype('category')

filtered_df['Sex'].cat.categories = [0,1]

filtered_df['Sex'] = filtered_df['Sex'].astype('int')



embarked_vals = len(filtered_df.Embarked.unique())

filtered_df['Embarked'] = filtered_df['Embarked'].astype('category')

filtered_df['Embarked'].cat.categories = [x for x in range(0, embarked_vals)]

filtered_df['Embarked'] = filtered_df['Embarked'].astype('int')



title_vals = len(filtered_df.title.unique())

filtered_df['title'] = filtered_df['title'].astype('category')

filtered_df['title'].cat.categories = [x for x in range(0, title_vals)]

filtered_df['title'] = filtered_df['title'].astype('int')



filtered_df['has_brackets'] = filtered_df['has_brackets'].astype('category')

filtered_df['has_brackets'].cat.categories = [0,1]

filtered_df['has_brackets'] = filtered_df['has_brackets'].astype('int')



filtered_df['has_quotes'] = filtered_df['has_quotes'].astype('category')

filtered_df['has_quotes'].cat.categories = [0,1]

filtered_df['has_quotes'] = filtered_df['has_quotes'].astype('int')



filtered_df.head()



ax = plt.subplots( figsize =( 12 , 10 ) )

foo = sns.heatmap(filtered_df.corr(), vmax=1.0, square=True, annot=True)
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier



# creating testing dataset for age bucket 10 prediction



# removing where age is null

age_prediction = filtered_df[filtered_df['age_bucket_10'].isnull().values == 0]

age_prediction.head()





# training

non_features = ['age_bucket_10','age_bucket_m5','Survived','PassengerId']

features = [x for x in age_prediction.columns if x not in non_features]



X = np.array(age_prediction[features])

y = np.array(age_prediction.age_bucket_10)



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=44)



# best hyperparameters as discovered by gridsearch

# Best score: 0.419

# Best parameters set:

# 	learning_rate: 0.08

# 	max_depth: 2

# 	max_features: 'sqrt'

# 	min_samples_split: 4



model = GradientBoostingClassifier( learning_rate = 0.08, max_depth = 2, max_features = 'sqrt',min_samples_split = 4)

model.fit(X_train, y_train)



y_predict = model.predict(X_test)



# % correct

# magnitude of incorrect

correct_count = 0

total_difference = 0

for x in range(0, len(y_test)):

    total_difference = total_difference + abs(y_test[x] - y_predict[x])

    

    if y_test[x] == y_predict[x]:

        correct_count = correct_count + 1



print("{}{}".format('Percentage correct = ', correct_count/len(y_test)))



print("{}{}".format('Total differences = ', total_difference))





#plot

# ax = sns.stripplot(x=y_test, y=y_predict, jitter=True)

ax = sns.violinplot(x=y_test, y=y_predict)

ax.set_xlabel('actual_age_bucket')

ax.set_ylabel('predicted_age_bucket')

ax.set_title('actual vs predicted age_buckets')
# creating testing dataset for age bucket m5 prediction



#filtering for males

age_bucket_m5_prediction = filtered_df[filtered_df['Sex'] == 1].copy()





# filtering for non-null values in age_bucket_m5

age_bucket_m5_prediction = age_bucket_m5_prediction[age_bucket_m5_prediction['age_bucket_m5'].isnull().values == False]



age_bucket_m5_prediction.head()



non_features = ['age_bucket_10','age_bucket_m5','Survived','PassengerId']

features = [x for x in age_bucket_m5_prediction.columns if x not in non_features]



X = np.array(age_bucket_m5_prediction[features])

y = np.array(age_bucket_m5_prediction.age_bucket_m5)



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33, random_state=44)

        

model = LogisticRegression()

model.fit(X_train, y_train)



y_predict = model.predict(X_test)



# % correct

# magnitude of incorrect

correct_count = 0

total_difference = 0

for x in range(0, len(y_test)):

    total_difference = total_difference + abs(y_test[x] - y_predict[x])

    

    if y_test[x] == y_predict[x]:

        correct_count = correct_count + 1



print("{}{}".format('Percentage correct = ', correct_count/len(y_test)))



print("{}{}".format('Total differences = ', total_difference))





#plot

# ax = sns.stripplot(x=y_test, y=y_predict, jitter=True)

ax = sns.violinplot(x=y_test, y=y_predict)

ax.set_xlabel('actual_age_bucket_m5')

ax.set_ylabel('predicted_age_bucket_m5')

ax.set_title('actual vs predicted age_buckets for age_bucket_m5')

# predict age_bucket_10



null_age_bucket_10 = filtered_df[filtered_df['age_bucket_10'].isnull().values == True]

null_age_bucket_10.reset_index(inplace = True)

not_null_age_bucket_10 = filtered_df[filtered_df['age_bucket_10'].isnull().values == False]

not_null_age_bucket_10.reset_index(inplace = True)

# training

X_train = np.array(not_null_age_bucket_10[features])

y_train = np.array(not_null_age_bucket_10.age_bucket_10)



# features for prediction

X_predict = np.array(null_age_bucket_10[features]) 



# model

model = GradientBoostingClassifier( learning_rate = 0.08, max_depth = 2, max_features = 'sqrt',min_samples_split = 4)

model.fit(X_train, y_train)



# predicting

y_predict = model.predict(X_predict)



null_age_bucket_10['age_bucket_10'] = y_predict
# predict age_bucket_m5



null_age_bucket_m5_male_only = filtered_df[(filtered_df['Sex']==1)

                                           &(filtered_df['age_bucket_m5'].isnull().values == True)]

null_age_bucket_m5_male_only.reset_index(inplace = True)

not_null_age_bucket_m5_male_only = filtered_df[(filtered_df['Sex']==1)

                                           &(filtered_df['age_bucket_m5'].isnull().values == False)]

not_null_age_bucket_m5_male_only.reset_index(inplace = True)

# train and predict data

X_train = np.array(not_null_age_bucket_m5_male_only[features])

y_train = np.array(not_null_age_bucket_m5_male_only.age_bucket_m5)



X_predict = np.array(null_age_bucket_m5_male_only[features])



#model

model = LogisticRegression()

model.fit(X_train, y_train)



y_predict = model.predict(X_predict)



#adding predictions into dataset

null_age_bucket_m5_male_only['age_bucket_m5']= y_predict

# creating the complete data_set for survival prediction



# inserting the values for age_bucket_10

for x in range(0, len(null_age_bucket_10)):

    

    passenger_id = null_age_bucket_10['PassengerId'][x]

    

    the_value_to_insert = null_age_bucket_10['age_bucket_10'][x]

    

    the_index_of_filtered_df = filtered_df[filtered_df['PassengerId']==passenger_id].index.values



    filtered_df.set_value(the_index_of_filtered_df,'age_bucket_10', the_value_to_insert)





filtered_df.tail()



# inserting the values for age_bucket_m5 for males

for x in range(0, len(null_age_bucket_m5_male_only)):

    

    passenger_id = null_age_bucket_m5_male_only['PassengerId'][x]

    

    the_value_to_insert = null_age_bucket_m5_male_only['age_bucket_m5'][x]

    

    the_index_of_filtered_df = filtered_df[filtered_df['PassengerId']==passenger_id].index.values

    

    filtered_df.set_value(the_index_of_filtered_df,'age_bucket_m5', the_value_to_insert)







# inserting the remaining values (for females)

filtered_df['age_bucket_m5'] = filtered_df['age_bucket_m5'].fillna(value = 5)

# lets take another look at correlation (with survival)



filtered_df_with_not_null_survival = filtered_df[filtered_df['Survived'].isnull().values == False]



a = plt.figure(figsize = (12,10))

a = sns.heatmap(filtered_df_with_not_null_survival.corr(), vmax = 1.0, square = True, annot = True)

# features for prediction

not_features = ['Survived'

                ,'PassengerId'

                #,'Sex'

                ]

features = [x for x in filtered_df_with_not_null_survival.columns if x not in not_features]



X = np.array(filtered_df_with_not_null_survival[features])

y = np.array(filtered_df_with_not_null_survival.Survived)



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.30, random_state=44)



model = GradientBoostingClassifier()







model.fit(X_train, y_train)



y_predict = model.predict(X_test)



# % correct

correct_count = 0

for x in range(0, len(y_test)):

    

    if y_test[x] == y_predict[x]:

        correct_count = correct_count + 1



print("{}{}".format('Percentage correct = ', correct_count/len(y_test)))
# FINAL

# predicting survival



# datasets

filtered_df_with_not_null_survival = filtered_df[filtered_df['Survived'].isnull().values == False]

filtered_df_with_null_survival = filtered_df[filtered_df['Survived'].isnull().values == True]



# features for prediction

not_features = ['Survived'

                ,'PassengerId'

                ,'Sex'

                ]

features = [x for x in filtered_df_with_not_null_survival.columns if x not in not_features]



X_train = np.array(filtered_df_with_not_null_survival[features])

y_train = np.array(filtered_df_with_not_null_survival.Survived)



X_predict = np.array(filtered_df_with_null_survival[features])



model = GradientBoostingClassifier()

model.fit(X_train, y_train)



y_predict = model.predict(X_predict)



filtered_df_with_null_survival['Survived'] = y_predict



# results to export



results = filtered_df_with_null_survival[['PassengerId','Survived']]



results.sort_values(by = 'PassengerId', ascending = True, inplace = True)

results.reset_index(drop = True, inplace = True)



results.to_csv('titanic_survival_results.csv')


# FOR GRIDSEARCH 



from __future__ import print_function



from pprint import pprint

from time import time

import logging



# MODELS (comment out the ones not being tested)

# pipeline = tree.DecisionTreeRegressor()

# pipeline = Ridge()

# pipeline = GradientBoostingRegressor()

# pipeline = SVR()

# pipeline = DecisionTreeClassifier()

pipeline = GradientBoostingClassifier()

# pipeline = LogisticRegression()

# pipeline = RandomForestClassifier()

# pipeline = ExtraTreeClassifier()



decision_tree_parameters = {

    'splitter': ('best','random'),

    'max_features': (None, 'auto','sqrt','log2') ,

    'max_depth': (None,10,14,18,22,28,35,40,45,50),

    'min_samples_split':(2,3,4,5,6,7,8,9,10),

    'min_samples_leaf':(1,2,3,4,5),

    'min_weight_fraction_leaf':(0., 0.02, 0.05, 0.1,0.12,0.13,0.15),

    'max_leaf_nodes':(None, 10,20,30,40,50)

}



ridge_parameters = {

    'alpha': (1,0.1,0.2,0.3,0.4,0.5),

    'fit_intercept': (True,False) ,

    'max_iter': (None,10,15,20,25),

    'normalize':(True,False),

    'solver':('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg')

}



gradient_boosting_parameters = {

    'n_estimators': (100,150),

    'max_depth':(None, 3, 5),

    'min_samples_split':(2,3,4,5,6),  

    'min_samples_leaf':(2,3,4),

    'min_weight_fraction_leaf':(0., 0.02),

    'max_features': (None,'log2'),

    'max_leaf_nodes':(None, 10,20),

    'warm_start':(True, False)

}



class_decision_tree = {

    'max_features':(None, 'auto','sqrt','log2'),

    'max_depth': (None,10,14,18,25),

    'min_samples_split':(2,3,4,5,6,7,8,9,10),

    'min_samples_leaf':(1,2,3,4,5),

    'min_weight_fraction_leaf':(0., 0.02, 0.05, 0.1,0.12,0.13,0.15),

    'max_leaf_nodes':(None, 10,20,30,40,50)

}



class_gradient_boosting = { 

    'learning_rate' : (0.08,0.1,1.2,1.5),

    'max_depth' : (2,3,4,5),

    'min_samples_split':(2,3,4,5,6,7,8,9,10),

    'max_features':(None, 'auto','sqrt','log2')

}



class_random_forest = {

    'n_estimators': (10,20,30),

    'criterion': ('gini','entropy'),

    'max_features':(None, 'auto','sqrt','log2'),

    'max_depth': (None,10,15,20),

    'min_samples_split':(2,3,4,5),  

    'min_samples_leaf':(1,2,3),

    'min_weight_fraction_leaf':(0., 0.1,0.12),

    'max_leaf_nodes':(None, 10,20,30)

}



logistic_regression_parameters = {

    'penalty': ('l1','l2'),

    'C':(0.95,1,1.05,1.1),

    'fit_intercept' : (True, False)

}



parameters = class_gradient_boosting



if __name__ == "__main__":

    # multiprocessing requires the fork to happen in a __main__ protected

    # block



    # find the best parameters for both the feature extraction and the

    # classifier

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)



    print("Performing grid search...")

    print("parameters:")

    pprint(parameters)

    t0 = time()

    grid_search.fit(X, y)

    print("done in %0.3fs" % (time() - t0))

    print()



    print("Best score: %0.3f" % grid_search.best_score_)

    print("Best parameters set:")

    best_parameters = grid_search.best_estimator_.get_params()

    for param_name in sorted(parameters.keys()):

        print("\t%s: %r" % (param_name, best_parameters[param_name]))