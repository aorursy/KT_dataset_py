import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv("../input/train.csv")
train_df.head()
pred_df = pd.read_csv("../input/test.csv")

# append the two datasets for feature engineering
train_df["dataset"] = "train"
pred_df["dataset"] = "pred"
data_df = train_df.append(pred_df, sort=True)
# show missing values
data_df.isnull().sum()
# clean name and extracting Title

data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

# replace rare titles with more common ones
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)
titles = list(data_df.Title.value_counts().index)

# for each title, impute missing age by the median of the persons with the same title
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[title]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute
# compute family size
data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
# get the last name (family name): the string part before the ,
data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

# remove null fare values
data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)
# get information about family survival using Last_Name and Fare

# 0.5: default value for no information
# 1: someone of the same family survived
# 0: we don't know if somebody survived but we know that somebody died

default_survival_value = 0.5
data_df['Family_Survival'] = default_survival_value

for grp, grp_df in data_df.groupby(['Last_Name', 'Fare']):
    # if a family group is found
    if (len(grp_df) != 1):
        # for every person, look for the other people from the same family
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            
            if (smax == 1.0): # if anyone in the family survived, assign 1
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0): # else if we saw someone dead, assign 0
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      data_df.loc[data_df['Family_Survival']!=0.5].shape[0])
# get information about family survival using the Ticket number

for _, grp_df in data_df.groupby('Ticket'):
    # if a family group is found
    if (len(grp_df) != 1):
        # for every person, look for the other people from the same family
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0): # if anyone in the family survived, assign 1
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0): # else if we saw someone dead, assign 0
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))
# fare bins
data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)
label = LabelEncoder()
data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])
data_df.drop(['Fare'], 1, inplace=True)
# age bins
data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)
label = LabelEncoder()
data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])
data_df.drop(['Age'], 1, inplace=True)
# encode sex variable
data_df['Sex'].replace(['male','female'],[0,1],inplace=True)
# choose features and labels
label = "Survived"
features = ["Pclass", "Sex", "Family_Size", "Family_Survival", "FareBin_Code", "AgeBin_Code"]

# split back data_df into train and prediction sets
train_df = data_df[data_df["dataset"] == "train"][features + [label]]
pred_df = data_df[data_df["dataset"] == "pred"][features]

# convert Survived variable to int for train dataset
train_df["Survived"] = train_df["Survived"].astype(np.int64)
train_df.head()
# setup dataframes
X = train_df[features]
y = train_df['Survived']
X_pred = pred_df

# scale data for KNN classifier
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
X_pred = std_scaler.transform(X_pred)
# setup parameters values for grid search
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'weights': weights, 'leaf_size': leaf_size, 'n_neighbors': n_neighbors}


gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True, 
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
print(gd.best_score_)
print(gd.best_estimator_)
# make predictions
gd.best_estimator_.fit(X, y)
y_pred = gd.best_estimator_.predict(X_pred)

# output predictions dataframe
temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = y_pred
temp.to_csv("../working/submission.csv", index = False)
