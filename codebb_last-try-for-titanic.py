# We can use the pandas library in python to read in the csv file.

import pandas as pd

#for numerical computaions we can use numpy library

import numpy as np
# This creates a pandas dataframe and assigns it to the titanic variable.

titanic = pd.read_csv("../input/train.csv")

# Print the first 5 rows of the dataframe.

titanic.head()
titanic_test = pd.read_csv("../input/test.csv")

#transpose

titanic_test.head().T

#note their is no Survived column here which is our target varible we are trying to predict
#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset

#(rows,columns)

titanic.shape
#Describe gives statistical information about numerical columns in the dataset

titanic.describe()

#you can check from count if there are missing vales in columns, here age has got missing values
#info method provides information about dataset like 

#total values in each column, null/not null, datatype, memory occupied etc

titanic.info()
#lets see if there are any more columns with missing values 

null_columns=titanic.columns[titanic.isnull().any()]

titanic.isnull().sum()
#how about test set??

titanic_test.isnull().sum()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



pd.options.display.mpl_style = 'default'

labels = []

values = []

for col in null_columns:

    labels.append(col)

    values.append(titanic[col].isnull().sum())

ind = np.arange(len(labels))

width=0.6

fig, ax = plt.subplots(figsize=(6,5))

rects = ax.barh(ind, np.array(values), color='purple')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_ylabel("Column Names")

ax.set_title("Variables with missing values");
titanic.hist(bins=10,figsize=(9,7),grid=False);
g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple");
g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare');
titanic.Embarked.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers per boarding location");
sns.factorplot(x = 'Embarked',y="Survived", data = titanic,color="r");
sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=titanic, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Passenger Class');
ax = sns.boxplot(x="Survived", y="Age", 

                data=titanic)

ax = sns.stripplot(x="Survived", y="Age",

                   data=titanic, jitter=True,

                   edgecolor="gray")

sns.plt.title("Survival by Age",fontsize=12);
titanic.Age[titanic.Pclass == 1].plot(kind='kde')    

titanic.Age[titanic.Pclass == 2].plot(kind='kde')

titanic.Age[titanic.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;
corr=titanic.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
#correlation of features with target variable

titanic.corr()["Survived"]
g = sns.factorplot(x="Age", y="Embarked",

                    hue="Sex", row="Pclass",

                    data=titanic[titanic.Embarked.notnull()],

                    orient="h", size=2, aspect=3.5, 

                   palette={'male':"purple", 'female':"blue"},

                    kind="violin", split=True, cut=0, bw=.2);
#Lets check which rows have null Embarked column

titanic[titanic['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic);
titanic["Embarked"] = titanic["Embarked"].fillna('C')
#there is an empty fare column in test set

titanic_test.describe()
titanic_test[titanic_test['Fare'].isnull()]
#we can replace missing value in fare by taking median of all fares of those passengers 

#who share 3rd Passenger class and Embarked from 'S' 

def fill_missing_fare(df):

    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

#'S'

       #print(median_fare)

    df["Fare"] = df["Fare"].fillna(median_fare)

    return df



titanic_test=fill_missing_fare(titanic_test)
titanic["Deck"]=titanic.Cabin.str[0]

titanic_test["Deck"]=titanic_test.Cabin.str[0]

titanic["Deck"].unique() # 0 is for null values
g = sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=titanic[titanic.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8);
titanic = titanic.assign(Deck=titanic.Deck.astype(object)).sort("Deck")

g = sns.FacetGrid(titanic, col="Pclass", sharex=False,

                  gridspec_kws={"width_ratios": [5, 3, 3]})

g.map(sns.boxplot, "Deck", "Age");
titanic.Deck.fillna('Z', inplace=True)

titanic_test.Deck.fillna('Z', inplace=True)

titanic["Deck"].unique() # Z is for null values
# Create a family size variable including the passenger themselves

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]+1

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1

print(titanic["FamilySize"].value_counts())
# Discretize family size

titanic.loc[titanic["FamilySize"] == 1, "FsizeD"] = 'singleton'

titanic.loc[(titanic["FamilySize"] > 1)  &  (titanic["FamilySize"] < 5) , "FsizeD"] = 'small'

titanic.loc[titanic["FamilySize"] >4, "FsizeD"] = 'large'



titanic_test.loc[titanic_test["FamilySize"] == 1, "FsizeD"] = 'singleton'

titanic_test.loc[(titanic_test["FamilySize"] >1) & (titanic_test["FamilySize"] <5) , "FsizeD"] = 'small'

titanic_test.loc[titanic_test["FamilySize"] >4, "FsizeD"] = 'large'

print(titanic["FsizeD"].unique())

print(titanic["FsizeD"].value_counts())
sns.factorplot(x="FsizeD", y="Survived", data=titanic);
#Create feture for length of name 

# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))



titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#print(titanic["NameLength"].value_counts())



bins = [0, 20, 40, 57, 85]

group_names = ['short', 'okay', 'good', 'long']

titanic['NlengthD'] = pd.cut(titanic['NameLength'], bins, labels=group_names)

titanic_test['NlengthD'] = pd.cut(titanic_test['NameLength'], bins, labels=group_names)



sns.factorplot(x="NlengthD", y="Survived", data=titanic)

print(titanic["NlengthD"].unique())
import re



#A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



#Get all the titles and print how often each one occurs.

titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))





#Add in the title column.

titanic["Title"] = titles



# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

titanic.loc[titanic["Title"] == "Mlle", "Title"] = 'Miss'

titanic.loc[titanic["Title"] == "Ms", "Title"] = 'Miss'

titanic.loc[titanic["Title"] == "Mme", "Title"] = 'Mrs'

titanic.loc[titanic["Title"] == "Dona", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Lady", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Countess", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Capt", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Col", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Don", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Major", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Rev", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Sir", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Jonkheer", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Dr", "Title"] = 'Rare Title'



#titanic.loc[titanic["Title"].isin(['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

#                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']), "Title"] = 'Rare Title'



#titanic[titanic['Title'].isin(['Dona', 'Lady', 'Countess'])]

#titanic.query("Title in ('Dona', 'Lady', 'Countess')")



titanic["Title"].value_counts()





titles = titanic_test["Name"].apply(get_title)

print(pd.value_counts(titles))



#Add in the title column.

titanic_test["Title"] = titles



# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

titanic_test.loc[titanic_test["Title"] == "Mlle", "Title"] = 'Miss'

titanic_test.loc[titanic_test["Title"] == "Ms", "Title"] = 'Miss'

titanic_test.loc[titanic_test["Title"] == "Mme", "Title"] = 'Mrs'

titanic_test.loc[titanic_test["Title"] == "Dona", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Lady", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Countess", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Capt", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Col", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Don", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Major", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Rev", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Sir", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Jonkheer", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Dr", "Title"] = 'Rare Title'



titanic_test["Title"].value_counts()
titanic["Ticket"].tail()
titanic["TicketNumber"] = titanic["Ticket"].str.extract('(\d{2,})', expand=True)

titanic["TicketNumber"] = titanic["TicketNumber"].apply(pd.to_numeric)





titanic_test["TicketNumber"] = titanic_test["Ticket"].str.extract('(\d{2,})', expand=True)

titanic_test["TicketNumber"] = titanic_test["TicketNumber"].apply(pd.to_numeric)
#some rows in ticket column dont have numeric value so we got NaN there

titanic[titanic["TicketNumber"].isnull()]
titanic.TicketNumber.fillna(titanic["TicketNumber"].median(), inplace=True)

titanic_test.TicketNumber.fillna(titanic_test["TicketNumber"].median(), inplace=True)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']

for col in cat_vars:

    titanic[col]=labelEnc.fit_transform(titanic[col])

    titanic_test[col]=labelEnc.fit_transform(titanic_test[col])



titanic.head()
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(titanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="red")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count");
from sklearn.ensemble import RandomForestRegressor

#predicting missing values in age using Random Forest

def fill_missing_age(df):

    

    #Feature set

    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',

                 'TicketNumber', 'Title','Pclass','FamilySize',

                 'FsizeD','NameLength',"NlengthD",'Deck']]

    # Split sets into train and test

    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values

    test = age_df.loc[ (df.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df
titanic=fill_missing_age(titanic)

titanic_test=fill_missing_age(titanic_test)
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(titanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="tomato")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")

    plt.xlim((15,100));
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])

titanic[['Age', 'Fare']] = std_scale.transform(titanic[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(titanic_test[['Age', 'Fare']])

titanic_test[['Age', 'Fare']] = std_scale.transform(titanic_test[['Age', 'Fare']])
titanic.corr()["Survived"]


from sklearn.cross_validation import KFold



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
titanic.head()
titanic_test.head()
X_train = titanic.drop(['PassengerId','Name','Ticket','Cabin','TicketNumber'],axis=1)

X_train.head()

X_test = titanic_test.drop(['PassengerId','Name','Ticket','Cabin','TicketNumber'],axis=1).copy()

X_test.head()
X_train = titanic.drop(['Survived','PassengerId','Name','Ticket','Cabin','TicketNumber'],axis=1)

Y_train = titanic['Survived']

X_test = titanic_test.drop(['PassengerId','Name','Ticket','Cabin','TicketNumber'],axis=1).copy()



y_train = titanic['Survived'].ravel()

ttrain = titanic.drop(['Survived','PassengerId','Name','Ticket','Cabin','TicketNumber'],axis=1)

x_train = ttrain.values 

x_test = titanic_test.drop(['PassengerId','Name','Ticket','Cabin','TicketNumber'],axis=1).values 
ntrain = X_train.shape[0]

ntest = X_test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)



def get_oof(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1).ravel(), oof_test.reshape(-1, 1).ravel()
logreg = LogisticRegression()



Y_logreg_train_pred, Y_logreg_pred_ensemble = get_oof(logreg, x_train, y_train, x_test)



Y_logreg_pred=np.where(Y_logreg_pred_ensemble>0.5,Y_logreg_pred_ensemble,0)

Y_logreg_pred=np.where(Y_logreg_pred<=0.5,Y_logreg_pred,1).astype(int)



print(np.equal(Y_logreg_train_pred,Y_train).astype(float).mean())
from sklearn.svm import SVC, LinearSVC



svc = SVC(probability=True)



Y_svc_train_pred, Y_svc_pred_ensemble = get_oof(svc, x_train, y_train, x_test)



Y_svc_pred=np.where(Y_svc_pred_ensemble>0.5,Y_svc_pred_ensemble,0)

Y_svc_pred=np.where(Y_svc_pred<=0.5,Y_svc_pred,1).astype(int)



print(np.equal(Y_svc_train_pred,Y_train).astype(float).mean())
## Random Forests
rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}



random_forest = RandomForestClassifier(**rf_params)





Y_random_forest_train_pred, Y_random_forest_pred_ensemble = get_oof(random_forest, x_train, y_train, x_test)



Y_random_forest_pred=np.where(Y_random_forest_pred_ensemble>0.5,Y_random_forest_pred_ensemble,0)

Y_random_forest_pred=np.where(Y_random_forest_pred<=0.5,Y_random_forest_pred,1).astype(int)



print(np.equal(Y_random_forest_train_pred,Y_train).astype(float).mean())
knn = KNeighborsClassifier(n_neighbors = 3)



Y_knn_train_pred, Y_knn_pred_ensemble = get_oof(knn, x_train, y_train, x_test)



Y_knn_pred=np.where(Y_knn_pred_ensemble>0.5,Y_knn_pred_ensemble,0)

Y_knn_pred=np.where(Y_knn_pred<=0.5,Y_knn_pred,1).astype(int)



print(np.equal(Y_knn_train_pred,Y_train).astype(float).mean())
gaussian = GaussianNB()



Y_gaussian_train_pred, Y_gaussian_pred_ensemble = get_oof(gaussian, x_train, y_train, x_test)



Y_gaussian_pred=np.where(Y_gaussian_pred_ensemble>0.5,Y_gaussian_pred_ensemble,0)

Y_gaussian_pred=np.where(Y_gaussian_pred<=0.5,Y_gaussian_pred,1).astype(int)



print(np.equal(Y_gaussian_train_pred,Y_train).astype(float).mean())
decisionTree = DecisionTreeClassifier(splitter='best')



Y_decisionTree_train_pred, Y_decisionTree_pred_ensemble = get_oof(decisionTree, x_train, y_train, x_test)



Y_decisionTree_pred=np.where(Y_decisionTree_pred_ensemble>0.5,Y_decisionTree_pred_ensemble,0)

Y_decisionTree_pred=np.where(Y_decisionTree_pred<=0.5,Y_decisionTree_pred,1).astype(int)



print(np.equal(Y_decisionTree_train_pred,Y_train).astype(float).mean())
xgbClassifier = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)



Y_xgbClassifier_train_pred, Y_xgbClassifier_pred_ensemble = get_oof(xgbClassifier, x_train, y_train, x_test)



Y_xgbClassifier_pred=np.where(Y_xgbClassifier_pred_ensemble>0.5,Y_xgbClassifier_pred_ensemble,0)

Y_xgbClassifier_pred=np.where(Y_xgbClassifier_pred<=0.5,Y_xgbClassifier_pred,1).astype(int)



print(np.equal(Y_xgbClassifier_train_pred,Y_train).astype(float).mean())
et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}



extraTreesClassifier = ExtraTreesClassifier(**et_params)



Y_extraTreesClassifier_train_pred, Y_extraTreesClassifier_pred_ensemble = get_oof(extraTreesClassifier, x_train, y_train, x_test)



Y_extraTreesClassifier_pred=np.where(Y_extraTreesClassifier_pred_ensemble>0.5,Y_extraTreesClassifier_pred_ensemble,0)

Y_extraTreesClassifier_pred=np.where(Y_extraTreesClassifier_pred<=0.5,Y_extraTreesClassifier_pred,1).astype(int)



print(np.equal(Y_extraTreesClassifier_train_pred,Y_train).astype(float).mean())
ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}

adaBoostClassifier = AdaBoostClassifier(**ada_params)



Y_adaBoostClassifier_train_pred, Y_adaBoostClassifier_pred_ensemble = get_oof(adaBoostClassifier, x_train, y_train, x_test)



#Y_adaBoostClassifier_pred = Y_adaBoostClassifier_pred_ensemble.copy()



#Y_adaBoostClassifier_pred[Y_adaBoostClassifier_pred > 0.5] = 1

#Y_adaBoostClassifier_pred[Y_adaBoostClassifier_pred <= 0.5] = 0

Y_adaBoostClassifier_pred=np.where(Y_adaBoostClassifier_pred_ensemble>0.5,Y_adaBoostClassifier_pred_ensemble,0)

Y_adaBoostClassifier_pred=np.where(Y_adaBoostClassifier_pred<=0.5,Y_adaBoostClassifier_pred,1).astype(int)



print(np.equal(Y_adaBoostClassifier_train_pred,Y_train).astype(float).mean())
gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}



gradientBoostingClassifier = GradientBoostingClassifier(**gb_params)



Y_gradientBoostingClassifier_train_pred, Y_gradientBoostingClassifier_pred_ensemble = get_oof(gradientBoostingClassifier, x_train, y_train, x_test)



Y_gradientBoostingClassifier_pred=np.where(Y_gradientBoostingClassifier_pred_ensemble>0.5,Y_gradientBoostingClassifier_pred_ensemble,0)

Y_gradientBoostingClassifier_pred=np.where(Y_gradientBoostingClassifier_pred<=0.5,Y_gradientBoostingClassifier_pred,1).astype(int)



print(np.equal(Y_gradientBoostingClassifier_train_pred,Y_train).astype(float).mean())
X_ensemble_train = pd.DataFrame({

    "logreg_pred":Y_logreg_train_pred,

    "svc_pred":Y_svc_train_pred,

    "random_forest_pred":Y_random_forest_train_pred,

    #"knn_pred":Y_knn_train_pred,

    #"gaussian_pred":Y_gaussian_train_pred,

    "decisionTree_pred":Y_decisionTree_train_pred,

    #"xgbClassifier_pred":Y_xgbClassifier_train_pred,

    "Y_extraTreesClassifier_pred":Y_extraTreesClassifier_train_pred,

    "Y_adaBoostClassifier_pred":Y_adaBoostClassifier_train_pred,

    "Y_gradientBoostingClassifier_pred":Y_gradientBoostingClassifier_train_pred

}

)



X_ensemble_test = pd.DataFrame({

    "logreg_pred":Y_logreg_pred_ensemble,

    "svc_pred":Y_svc_pred_ensemble,

    "random_forest_pred":Y_random_forest_pred_ensemble,

    #"knn_pred":Y_knn_pred_ensemble,

    #"gaussian_pred":Y_gaussian_pred_ensemble,

    "decisionTree_pred":Y_decisionTree_pred_ensemble,

    #"xgbClassifier_pred":Y_xgbClassifier_pred_ensemble,

    "Y_extraTreesClassifier_pred":Y_extraTreesClassifier_pred_ensemble,

    "Y_adaBoostClassifier_pred":Y_adaBoostClassifier_pred_ensemble,

    "Y_gradientBoostingClassifier_pred":Y_gradientBoostingClassifier_pred_ensemble

}

)



#ensemble

xgbClassifier = xgb.XGBClassifier(

 n_estimators= 1000,

 max_depth= 3,

 min_child_weight= 2,

 #gamma=1,

 gamma=0.9,                        

 subsample=0.5,

 colsample_bytree=0.5,

 objective= 'binary:logistic',

 nthread= -1,

 scale_pos_weight=1)



xgbClassifier.fit(X_ensemble_train, Y_train)



Y_ensemble_pred = xgbClassifier.predict(X_ensemble_test)



xgbClassifier.score(X_ensemble_train, Y_train)
from sklearn.ensemble import VotingClassifier

ensemble_vote = VotingClassifier(estimators=[

        ('lr', logreg), ('svc', svc), ('random_forest', random_forest),

        #('knn',knn),('gaussian',gaussian),('decisionTree',decisionTree),

        ('xgbClassifier',xgbClassifier),('extraTreesClassifier',extraTreesClassifier),

        ('adaBoostClassifier',adaBoostClassifier),('gradientBoostingClassifier',gradientBoostingClassifier)], voting='soft')



Y_ensemble_vote_train_pred, Y_ensemble_vote_pred_ensemble = get_oof(ensemble_vote, x_train, y_train, x_test)



Y_ensemble_vote_pred=np.where(Y_ensemble_vote_pred_ensemble>0.5,Y_ensemble_vote_pred_ensemble,0)

Y_ensemble_vote_pred=np.where(Y_ensemble_vote_pred<=0.5,Y_knn_pred,1).astype(int)



print(np.equal(Y_ensemble_vote_train_pred,Y_train).astype(float).mean())
submission1 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_logreg_pred

    })

submission1.to_csv('logreg_pred_titanic.csv', index=False)



submission2 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_svc_pred

    })

submission2.to_csv('svc_pred_titanic.csv', index=False)



submission5 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_random_forest_pred

    })

submission5.to_csv('random_forest_pred_titanic.csv', index=False)



submission7 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_xgbClassifier_pred

    })

submission7.to_csv('xgbClassifier_pred_titanic.csv', index=False)



submission8 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_ensemble_pred

    })

submission8.to_csv('ensemble_pred_titanic.csv', index=False)



submission9 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_extraTreesClassifier_pred

    })

submission9.to_csv('extraTreesClassifier_pred_titanic.csv', index=False)



submission10 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_adaBoostClassifier_pred

    })

submission10.to_csv('adaBoostClassifier_pred_titanic.csv', index=False)



submission11 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_gradientBoostingClassifier_pred

    })

submission11.to_csv('gradientBoostingClassifier_pred_titanic.csv', index=False)



submission12 = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": Y_ensemble_vote_pred

    })

submission12.to_csv('ensemble_vote_pred_pred_titanic.csv', index=False)