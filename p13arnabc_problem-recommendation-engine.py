#Necessary imports
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
user_df = pd.read_csv(r'../input/recommendation-engine/user_data.csv')
problem_df = pd.read_csv(r'../input/recommendation-engine/problem_data.csv')
train_submussion_df = pd.read_csv(r'../input/recommendation-engine/train_submissions.csv')
test_submussion_df = pd.read_csv(r'../input/recommendation-engine/test_submissions_NeDLEvX.csv')
#lets look at the sample data for each of the data frame. Sample data for user data
user_df.head()
#description of user data
user_df.describe()
#count of null values
user_df.isna().sum()
#percentage of null values
user_df.isnull().mean()
#plotting submision counts
sns.distplot(user_df["submission_count"])
#plotting problem solved counts
user_df['submission_count'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])
#creating bins for submission counts
submission_count_bins = pd.qcut(user_df["submission_count"], 4,labels = False)

#creating the new column for quantiled submission count
user_df["submission_count_bins"] = submission_count_bins.values
#Let's have a look at the distribution for submission count bins
sns.distplot(user_df["submission_count_bins"])
#plotting problem solved counts
sns.distplot(user_df["problem_solved"])
#plotting problem solved counts
user_df['problem_solved'].quantile([.2, .4,.6, .8])
#quantiling the problem solved counts
problem_solved_bins = pd.qcut(user_df["problem_solved"], 5,labels = False)

#creating bins for problem solved counts
user_df["problem_solved_bins"] = problem_solved_bins.values
#let's look at the distribution of the problem solved bins
sns.distplot(user_df["problem_solved_bins"])
#let's have a look at the new column for problem solved bins
user_df.head()
#looks like problem solved count bins and submission count bins are identical. So I will check them once if they are idential
user_df['submission_count_bins'].equals(user_df['problem_solved_bins'])
#define success rate as a column
user_df['success_rate'] = user_df['problem_solved']/user_df['submission_count']*100
#Let's look at the distribution of the contribution
sns.distplot(user_df["contribution"], kde=False, rug=True)
#this is quite skewed with 2530 values as 0
user_df["contribution"].value_counts(normalize = True).head(10)
#Now let's look at the number of countries or values
user_df["country"].unique().shape
#Getting all the ratios
country_data = (user_df["country"].value_counts()/user_df["country"].count())
#imputing missing values
user_df["country"]= user_df["country"].fillna(pd.Series(np.random.choice(country_data.index,p=country_data.values, size=len(user_df))))
country_list = user_df['country'].value_counts().index[:9]
user_df['country_new'] = np.where(user_df['country'].isin(country_list), user_df['country'], 'Other')
#Now let's look at the countries distribution
sns.countplot(user_df["country_new"])
#value counts of new field
user_df["country_new"].value_counts()
#plotting follower_count
sns.distplot(user_df["follower_count"])
user_df.loc[user_df["follower_count"]==0].shape
#quantiling the follower_count
user_df['follower_count'].quantile([.2, .4, .6, .8])
#creating bins for submission counts
follower_count_bins = pd.qcut(user_df["follower_count"], 5,labels = False)
#creating the new column for quantiled submission count
user_df["follower_count_bins"] = follower_count_bins.values
#Let's have a look at the new distribution
sns.distplot(user_df["follower_count_bins"])
#let's find the age of the user in the platform in months
user_df["age_in_platform"] = (user_df["last_online_time_seconds"] - user_df["registration_time_seconds"])/(24*3600*30)
sns.distplot(user_df["age_in_platform"])
#plotting max_rating
sns.distplot(user_df["max_rating"])
#creating bins for max_rating counts
max_rating_bins = pd.qcut(user_df["max_rating"], 4,labels = False)
#creating the new column for quantiled max_rating count
user_df["max_rating_bins"] = max_rating_bins.values
#plotting max_rating counts
sns.distplot(user_df["max_rating_bins"])
#plotting submision counts
sns.distplot(user_df["rating"])
#Now let's look at the unique ranks
user_df["rank"].unique()
#Now let's look at the rank distribution
sns.countplot(user_df["rank"])
#Percentage distribution of rank
sns.barplot(user_df["rank"].value_counts(normalize = True).index, user_df["rank"].value_counts(normalize = True).values)
#percentage distribution of rank. It looks good to go
user_df["rank"].value_counts(normalize = True)
user_df.columns
#70% values are 0, so we can drop this field
user_df.drop(columns = ["contribution"],axis = 1, inplace = True)
#drop country as we have a new field for country with 'Other'
user_df.drop(columns = ["country"],axis = 1, inplace = True)
#registration time in years
user_df["registration_time"] = (time.time()-user_df["registration_time_seconds"])/(3600*24*365)
#last online time in years
user_df["last_online_time"] = (time.time()-user_df["last_online_time_seconds"])/(3600*24*365)
#drop last_online_time_seconds and registration_time_seconds as we have new fields for them
user_df.drop(columns = ["last_online_time_seconds","registration_time_seconds"],axis = 1, inplace = True)
#change values of country_new using a label encoder
labelencoder = LabelEncoder()
user_df['country_new'] = labelencoder.fit_transform(user_df['country_new'])
#change values of rank to numeric
rank_dict = {'beginner':0, 'intermediate':1, 'advanced':2, 'expert':3}
user_df["rank"] = user_df["rank"].apply(lambda x: rank_dict[x])
user_df.head()
#lets look at the sample data for problem data.
problem_df.head()
#let's look at the null values and the shape of the problem data
print(problem_df.shape)
print(problem_df.isna().sum())
print(problem_df.isna().mean())
#let's look at the distribution of the level type
problem_df.level_type.value_counts()
#let's look at the distribution of the level type
problem_df.level_type.value_counts(normalize = True)
#I will fill up the values based on the ratio of distribution
#Getting all the ratios
level_type_data = (problem_df["level_type"].value_counts()/problem_df["level_type"].count())

#imputing missing values
problem_df["level_type_new"]= problem_df["level_type"].fillna(pd.Series(np.random.choice(level_type_data.index,p=level_type_data.values, size=len(problem_df))))
#Now I will have to label the level_type_new field
level_type_dict = {'A':0, 'B':1, 'C':2, 'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13}
problem_df["level_type_new"] = problem_df["level_type_new"].apply(lambda x: level_type_dict[x])
print(problem_df["points"].mean())
print(problem_df["points"].mode())
print(problem_df["points"].median())
#imputing missing points values
problem_df["points"]= problem_df["points"].fillna(problem_df["points"].mean())
#I will fill up the values based on the ratio of distribution
#Getting all the ratios
#points_data = (problem_df["points"].value_counts()/problem_df["points"].count())

#imputing missing values for points
#problem_df["points"]= problem_df["points"].fillna(pd.Series(np.random.choice(points_data.index,p=points_data.values, size=len(problem_df))))
#I will remove level_type as there is a new field for that. tags should be removed as they have more than 50% null values
problem_df.drop(columns = ["level_type","tags"],axis = 1, inplace = True)
problem_df.head()
train_submussion_df.head()
#let's look at the distribution of the attempts range
sns.distplot(train_submussion_df["attempts_range"])
train_submussion_df["attempts_range"].value_counts(normalize = True)
#merge train submission and user data
train_df = pd.merge(train_submussion_df,user_df,how = 'left',on = "user_id")
test_df = pd.merge(test_submussion_df,user_df,how = 'left',on = "user_id")
#merge train data and problem data
train_df = pd.merge(train_df,problem_df,how = 'left',on = "problem_id")
test_df = pd.merge(test_df,problem_df,how = 'left',on = "problem_id")
#create ID field for train data, ID already there for test data
train_df["ID"] = train_df["user_id"] + train_df["problem_id"]
train_df.head()
test_df.head()
#user_id count - number of times user is appearing
train_df['user_id_count'] = train_df.groupby('user_id')['user_id'].transform('count')
test_df['user_id_count'] = train_df.groupby('user_id')['user_id'].transform('count')
#problem_id count - number of times problem is appearing
train_df['problem_id_count'] = train_df.groupby('problem_id')['problem_id'].transform('count')
test_df['problem_id_count'] = train_df.groupby('problem_id')['problem_id'].transform('count')
#user id min attempts
train_df['user_id_min_attempts'] = train_df.groupby('user_id')['attempts_range'].transform('min')
test_df['user_id_min_attempts'] = train_df.groupby('user_id')['attempts_range'].transform('min')
#user id max attempts
train_df['user_id_max_attempts'] = train_df.groupby('user_id')['attempts_range'].transform('max')
test_df['user_id_max_attempts'] = train_df.groupby('user_id')['attempts_range'].transform('max')
#user id mean attempts
train_df['user_id_mean_attempts'] = train_df.groupby('user_id')['attempts_range'].transform('mean')
test_df['user_id_mean_attempts'] = train_df.groupby('user_id')['attempts_range'].transform('mean')
#problem id min attempts
train_df['problem_id_min_attempts'] = train_df.groupby('problem_id')['attempts_range'].transform('min')
test_df['problem_id_min_attempts'] = train_df.groupby('problem_id')['attempts_range'].transform('min')
#problem id max attempts
train_df['problem_id_max_attempts'] = train_df.groupby('problem_id')['attempts_range'].transform('max')
test_df['problem_id_max_attempts'] = train_df.groupby('problem_id')['attempts_range'].transform('max')
#problem id mean attempts
train_df['problem_id_mean_attempts'] = train_df.groupby('problem_id')['attempts_range'].transform('mean')
test_df['problem_id_mean_attempts'] = train_df.groupby('problem_id')['attempts_range'].transform('mean')
#user id min level
train_df['user_id_min_level'] = train_df.groupby('user_id')['level_type_new'].transform('min')
test_df['user_id_min_level'] = train_df.groupby('user_id')['level_type_new'].transform('min')
#user id max level
train_df['user_id_max_level'] = train_df.groupby('user_id')['level_type_new'].transform('max')
test_df['user_id_max_level'] = train_df.groupby('user_id')['level_type_new'].transform('max')
#user id mean level
train_df['user_id_mean_level'] = train_df.groupby('user_id')['level_type_new'].transform('mean')
test_df['user_id_mean_level'] = train_df.groupby('user_id')['level_type_new'].transform('mean')
train_df['country_percent'] = train_df.groupby('country_new')['country_new'].transform('count')/len(train_df)
test_df['country_percent'] = train_df.groupby('country_new')['country_new'].transform('count')/len(train_df)
print(train_df.columns.shape)
print(test_df.columns.shape)
#define X
X = train_df.drop(columns=['user_id','problem_id','ID','attempts_range'],axis=1)
#define y
y = train_df["attempts_range"]
#split training data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#xgb bseline model
xgbC = XGBClassifier(n_estimators= 300)
xgbC.fit(X_train,y_train)
y_test_pred = xgbC.predict(X_test)
accuracy_score(y_test, y_test_pred)
f1_score(y_test, y_test_pred, average='weighted')
#use random forest regressor
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_depth=5, min_samples_leaf=100)
#fit the model
RF.fit(X_train,y_train)
y_test_pred = xgbC.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)
from sklearn.metrics import f1_score
f1_score(y_test, y_test_pred, average='weighted')
