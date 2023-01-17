###Import imports for analysis and visualization



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style("whitegrid")
###Reading and checking the data



df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")

df.head()
###Description of the dataset



df.info()
###Checking for missing values



missing = df.isnull().sum()

missing = missing[missing>0]

missing.sort_values(ascending=False, inplace=True)

missing
###Replacing the missing Version assuming it's 1.0



df["Current Ver"].replace(np.nan, "1.0", inplace=True)
###Checking Type null index



df[df["Type"].isnull()]
###Since the price is 0, it is a Free app



df["Type"].replace(np.nan, "Free", inplace=True)
##Checking for null Android Ver indices



df[df["Android Ver"].isnull()]
###This index is entered incorrectly, better to drop it 



df.drop(index=10472, inplace=True)
###The mode of the Android Ver is 4.1 and up 



df["Android Ver"].replace(np.nan, "4.1 and up", inplace=True)
###Removing the incorrect index fixed this null value problem



df[df["Content Rating"].isnull()]
###Checking the head of the null Rating indices



df[df["Rating"].isnull()].head()
###Plotting a graph to see how many of those apps received reviews

# around 80% of the null Rating apps received less than 10 reviews



plt.figure(figsize=(20,8))

sns.countplot(x=df[df["Rating"].isnull()]["Reviews"].astype(int))

plt.tight_layout()
###Plotting a graph to see the ratings of apps which received less than 10 reviews (80-85% of our null cases)

#less than 100 reviews (for the rest of the missing values)

#and all the ratings



fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

sns.boxplot(y=df[df["Reviews"] < '10']["Rating"], ax=ax1)

sns.boxplot(y=df[df["Reviews"] < '100']["Rating"], ax=ax2)

sns.boxplot(y=df["Rating"], ax=ax3)



plt.tight_layout()
###Getting the mean of those Ratings



print(df[df["Reviews"]<'10']["Rating"].mean())

print(df[df["Reviews"]<'100']["Rating"].mean())
###Impute function to fill in the missing Rating values



def FillRating(row):

    Rating = row[0]

    Reviews = row[1]

    

    if pd.isnull(Rating):

        if Reviews == '0':

            return 0

        elif Reviews < '10':

            return 4.1

        else:

            return 4.2

    else:

        return Rating



df["Rating"] = df[["Rating","Reviews"]].apply(FillRating, axis=1)
###All null values have been accounted for



df.isnull().sum().sort_values(ascending=False)
df.info()
###Impute functions to fix the data types of some of the columns



def FixSize(Size):

    if Size == "Varies with device":

        return 11

    byte = Size[-1]

    if byte == 'k':

        byte_size = 10**-3

    else:

        byte_size = 1

    

    split = Size[:-1]

    return float(split) * byte_size



def FixInstalls(Installs):

    if Installs == '0':

        return 0

    Installs = Installs[:-1]

    split = Installs.split(',')

    Installs = ''.join(x for x in split)

    return int(Installs)



def FixPrice(Price):

    if Price == '0':

        return 0.0

    else:

        Price = Price.split('$')[1:]

        Price = ''.join(x for x in Price)

        return float(Price)

    

###Applying the impute functions



df["Reviews"] = df["Reviews"].apply(lambda x: int(x))

df["Size"] = df["Size"].apply(FixSize)

df["Installs"] = df["Installs"].apply(FixInstalls)

df["Price"] = df["Price"].apply(FixPrice)



#Renaming Size --> Size(MBs) to reflect the new size

df.rename(columns={"Size":"Size(MBs)"}, inplace=True)
###All numerical columns have been assigned the proper data type



df.info()
###Stumbled upon the fact that there are several duplicates in the dataset



df["App"].nunique()
###Dropping the exact duplicate rows



exact_duplicates = df[df.duplicated()].index

df.drop(index=exact_duplicates, inplace=True)
###Checking the remaining duplicates it seems they only differ by a rating or so, I assume this happened during the scraping

#I decided to remove all the occurences except the last



duplicates = df[df.duplicated("App", keep='last')].index

df.drop(index=duplicates, inplace=True)
###There are only 9659 unique apps in the dataset



df.info()
###Content Rating can be converted to age



df["Content Rating"].value_counts()
###Mapping new values into Content Rating and renaming it to Age Rating



age = {"Everyone":0,"Teen":13,"Mature 17+":17,"Everyone 10+":10,"Adults only 18+":18,"Unrated":0}

df["Content Rating"] = df["Content Rating"].map(age)

df.rename(columns={"Content Rating":"Age Rating"}, inplace=True)
###Categorizing Android Versions as the minimum version needed to run the app, assigning varies with models as 4 as it is the most common

#renaming Android Ver --> Min. Android Ver to better reflect the new data



def FixAndroidVer(Ver):

    if Ver[0] == "V":

        return 4

    else:

        return int(Ver[0])



df["Android Ver"] = df["Android Ver"].apply(FixAndroidVer)

df.rename(columns={"Android Ver":"Min. Android Ver"}, inplace=True)
df.info()
df[df["Current Ver"].str.isalpha()]["Current Ver"].value_counts()
###Categorizing Current Ver and assigning 1 in it's place as it is the second most common entry

#There is one App which has ³ in it's version, special test case for it included



def FixCurrentVer(Ver):

    for i in range(len(Ver)):

        if Ver[i].isdigit():

            if Ver[i] == "³":

                return 3

            else:

                return int(Ver[i])

        else:

            continue

    return 1



df["Current Ver"] = df["Current Ver"].apply(FixCurrentVer)
###Changing Last Updated to Updated based on if it's Current Version is greater than 1, as well as renaming the column



df["Last Updated"] = df["Current Ver"].apply(lambda x: 1 if x>1 else 0)

df.rename(columns = {"Last Updated":"Updated"}, inplace=True)
###Dropping the Apps with 0 Reviews



df = df.drop(df[df["Reviews"] == 0].index)
df.info()
###Checking the Correlation between the attributes



plt.figure(figsize=(16,8))

sns.heatmap(df.corr(), annot=True, cmap="Blues", linewidth=0.5)
###Checking the spread of the data



df.describe()
###Having some outliers for Price



df[df["Price"] > 100]
###Dropping the garbage apps



df = df.drop(df[df["Price"] > 100].index)
###Dropping App and Genres as they are too big to categorize, Genres had over 100 unique values



df.drop(["App","Genres"], axis=1, inplace=True)
###Extracting all the columns for Visualization



num_cols = df[["Rating","Reviews","Size(MBs)","Installs","Price"]]
###Visualizing the distribution of the data



fig = plt.figure(figsize=(20,8))



for i in range(len(num_cols.columns)):

    fig.add_subplot(2,3,i+1)

    sns.distplot(num_cols.iloc[:,i], hist=False, rug=True, kde_kws={"bw":0.01, "shade":True}, label="Dist")

    

plt.tight_layout()
###Comparing against Rating



fig = plt.figure(figsize=(20,12))



for i in range(len(num_cols.columns)):

    fig.add_subplot(2,3,i+1)

    sns.scatterplot(x=num_cols.iloc[:,i], y=df["Rating"])

    

plt.tight_layout()
plt.figure(figsize=(20,6))

temp = df["Category"]

temp.value_counts().plot.bar()

plt.xticks(rotation=90)

plt.title("Popularity of App Categories")
pop = pd.DataFrame(data={"No. of Apps":df["Category"].value_counts()})

pop["Percent of Total Share"] = pop["No. of Apps"].apply(lambda x: (x/df["Category"].count())*100)

pop
plt.figure(figsize=(12,4))

sns.countplot(df["Rating"])

plt.tight_layout()
plt.figure(figsize=(20,6))

sns.violinplot(x=df["Category"], y=df["Rating"])

plt.xticks(rotation=90)
plt.figure(figsize=(20,6))

sns.distplot(df["Size(MBs)"])
df.info()
###Contverting Type and Category into numerical columns and scaling the dataset



fixType = {"Free":0 ,"Paid":1}

df["Type"] = df["Type"].map(fixType)



df = pd.get_dummies(df)



from sklearn.preprocessing import StandardScaler



scaled = StandardScaler().fit_transform(df)

df = pd.DataFrame(scaled, columns=df.columns)
###Imports for Modelling



from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
###Setting up all the models



models = [["Logistic Regression",LogisticRegression()],

         ["Decision Tree",DecisionTreeClassifier()],

         ["Random Forest",RandomForestClassifier(n_estimators=100)],

         ["Boost",XGBClassifier(n_estimators=100, n_jobs=-1, learning_rate=0.05)],

         ["KNN",KNeighborsClassifier(n_neighbors=1)]]
from sklearn.preprocessing import LabelEncoder



df["Installs"] = LabelEncoder().fit_transform(df["Installs"])
###Setting up the training and testing data



X = df.drop("Installs", axis=1)

y = df["Installs"]



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
###Training and Testing the models



for name, model in models:

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    print("{} MAE: ".format(name), mean_absolute_error(y_test,pred))

    print("{} RMSE: ".format(name), np.sqrt(mean_squared_error(y_test,pred)))

    print("{} Accuracy : ".format(name), accuracy_score(y_test,pred), end='\n\n')
###XGBRegressor gave the best results

#Setting up the final model with better parameters



final = XGBClassifier(n_estimators=1000, n_jobs=-1, learning_rate=0.05)

final.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test,y_test)], verbose=False)

predictions = final.predict(X_test)

print("XGBClassifier MAE:", mean_absolute_error(y_test, predictions))

print("XGBClassifier RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

print("XGBClassifier Accuracy:", accuracy_score(y_test,predictions))
sns.regplot(y=y_test, x=predictions)

plt.ylabel("Actual")

plt.xlabel("Predicted")