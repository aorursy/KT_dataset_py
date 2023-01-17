import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

from sklearn.ensemble import RandomForestClassifier #Classification by trees ensemble
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

#Function to append sets with an idetifying partition field for later seperation or other uses

def append_train_test(train,test): 

    train["IsTrain"]=1

    test["IsTrain"]=0

    df=train.append(test)

    return df;



full=append_train_test(train,test)



train.info()

print ("------------------------")

test.info()

print ("------------------------")

full.info()

full.head()
#remember that \w is equivalent to [A-Za-z] all alphanumeric chars, dot must be escaped with "\" 

#and a match group (like the word) is enclosed in "()", the + is a quantifier meaning one or more of \w.

def get_title(df,name_column,new_column): #call function example: get_title(train,"Name","Title")

    df[new_column] = df[name_column].str.extract('(\w+)\.', expand=True)

    return df;

#I then pass the entire dataset to the function to generate the Title column for all the records

get_title(full,"Name","Title")

full.Title.value_counts() #Looks at all available values and their counts
full.loc[(full["Title"].isin(["Rev","Dr","Col","Capt","Major"])) & (full["Sex"]!="male")]
full.loc[(full["Title"].isin(["Rev","Capt","Major","Col","Jonkheer","Don","Sir","Dr"])) & (full["Sex"]=="male"),"Title"] = "Mr"

full.loc[(full["Title"].isin(["Countess","Lady","Dona","Mme"])),"Title"]="Mrs"

full.loc[(full["Title"].isin(["Mlle","Ms","Dr"])) & (full["Sex"]=="female"),"Title"]="Miss"



full["Title"].value_counts(normalize=True)
sns.countplot('Title',hue='Survived',data=full.iloc[:891])

plt.show()
full["FamilySize"]=full["SibSp"]+full["Parch"]+1



f,ax=plt.subplots(1,2,figsize=(18,8))

#Let's look at how family size is connected with Title

sns.countplot('FamilySize',hue='Title',data=full.iloc[:891],ax=ax[0])

ax[0].set_title('Family Sizs vs Title')

#Let's look at how survival is distributed along family sizes

sns.countplot('FamilySize',hue='Survived',data=full.iloc[:891],ax=ax[1])

ax[1].set_title('Family size vs Survived')

plt.show()
full["FamilySizeBand"]=np.nan

full.loc[full["FamilySize"]==1,"FamilySizeBand"]="Loner"

full.loc[(full["FamilySize"]<5) & (full["FamilySize"]>1),"FamilySizeBand"]="SmallFam"

full.loc[full["FamilySize"]>4,"FamilySizeBand"]="BigFam"

#Let's look a survival rate within classes and family sizes

sns.factorplot('FamilySizeBand','Survived',hue='Pclass',col="Sex",data=full.iloc[:891])

plt.show()
full.isnull().sum()
full[full.Fare.isnull()]
fare=full[(full.Embarked=="S") & (full.Title=="Mr") & (full.FamilySizeBand=="Loner") & (full.Pclass==3) & (full.Fare.notnull())]

sns.distplot(fare.Fare)

fare.Fare.describe() #this gives us the amount of records, and description of distribution
# imputing the missing fare value

full.loc[(full.Fare.isnull()),"Fare"]=8.05
full[full.Embarked.isnull()]
sns.boxplot(x="Embarked", y="Fare",hue="Sex",data=full[full.Fare.notnull()])

plt.show()
#impute missing Embarked values

full.loc[(full.Embarked.isnull()),"Embarked"]="C"
data=full[full.Age.notnull()]

sns.boxplot(x="Title", y="Age",hue="Pclass",data=data)

plt.show()
ImpAge=pd.DataFrame({'median' : data.groupby( [ "Title", "Pclass"] ).Age.median()}).reset_index()

ImpAge.head(n=20)
#now we define a function to impute according to these conditions

def imputeAges(df):

    classes=[1,2,3]

    titles=["Mr","Mrs","Miss","Master"]

    for title in titles:

        for pclass in classes:

            x=ImpAge[((ImpAge.Title==title) & (ImpAge.Pclass==pclass))]["median"].values[0]

            df.loc[((df.Title==title) & (df.Pclass==pclass) & (df.Age.isnull())),"Age"]=x

    return df

imputeAges(full)

full.isnull().sum()
full['Sex'].replace(['male','female'],[0,1],inplace=True)

full['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

full['Title'].replace(['Mr','Mrs','Miss','Master'],[0,1,2,3],inplace=True)

full['FamilySizeBand'].replace(['Loner',"BigFam","SmallFam"],[0,1,2],inplace=True)
features=["Age","Embarked","Fare","Parch","Pclass","Sex","SibSp","Title","FamilySize","FamilySizeBand"]

target="Survived"

full[features].head()
train_features=full[:891][features]

train_target=full[:891]["Survived"]

test_features=full[891:][features]

train_features.info()

print ("------------------------")

test_features.info()
clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf = clf.fit(train_features, train_target)

train_preds=clf.predict(train_features)

print (clf)

## variable importance



for i in range(0,10):

        print (features[i],"       ",clf.feature_importances_[i])
# Create confusion matrix to evaluate training

pd.crosstab(train_target, train_preds, rownames=['Actual Outcome'], colnames=['Predicted Outcome'])
test_preds=clf.predict(test_features).astype(int)

test_ids=full[891:]["PassengerId"]



final = pd.DataFrame({

        "PassengerId": test_ids,

        "Survived": test_preds

    })



final.info()
final.to_csv('submission2.csv', index=False)