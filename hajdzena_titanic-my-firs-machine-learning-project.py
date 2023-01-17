# import modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

%matplotlib inline
# import data
df_train = pd.read_csv("../input/train.csv")
# view info of data
df_train.info()
# view head of data
df_train.head()
# plot barplot
sns.countplot(x="Survived",data=df_train,hue="Sex")
#drop unimportant colomns
df_train_data = df_train.drop(["PassengerId", "Pclass","Embarked","Ticket"],axis=1) 
# create new data set Has_cabin
df_train_data["Has_cabin"]=[pd.isnull(value) for value in df_train_data["Cabin"]]

# create new data set Title
# extract Title from Name column
df_train_data["Title"] = [re.search('([A-z]+)\.',title).group(1)  for title in df_train_data["Name"]]
# view head of data
df_train_data.head(2)
# plot barplot
sns.countplot(x="Title",data=df_train_data)
plt.xticks(rotation = 90)
# replace data in Title column
df_train_data["Title"]=df_train_data["Title"].replace(["Miss","Ms","Lady","Mlle","Countess",
                                                                "Mme"],"Mrs")

df_train_data["Title"]= df_train_data["Title"].replace(["Don","Sir","Jonkheer"],"Mr")
df_train_data["Title"]= df_train_data["Title"].replace(["Rev","Dr","Major","Col","Capt",],"Special")
            
# plot barplot
sns.countplot(x="Title",data=df_train_data)
plt.xticks(rotation = 90)
#view info of data
df_train_data.info()
# view head of data 
df_train_data.head(2)
# addition Age data
df_train_data["Age"]=df_train_data.Age.fillna(df_train_data.Age.median())
# drop columns
df_train_data = df_train_data.drop(["Survived","Name","Cabin"],axis=1)
# Transform variables into numerical variables
df_data_dummies = pd.get_dummies(df_train_data,drop_first=True)
# split into test, train data and  transform into arrays for scikit-learn
X_train =df_data_dummies.iloc[:800].values
y_train =df_train.Survived.iloc[:800].values
X_test = df_data_dummies.iloc[800:].values
y_test = df_train.Survived.iloc[800:].values


# build a score function for random forest classifier
def get_score(X_train,X_test,y_train,y_test,max_leaf_nodes):
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes,random_state=1)
    model.fit(X_train,y_train)
    predict_value = model.predict(X_test)
    score = accuracy_score(y_test,predict_value)
    return score
    
# call get_score function for different max_leaf_nodes
for max_leaf_nodes in [3,5,50,500,5000]:
    my_score = get_score(X_train,X_test,y_train,y_test,max_leaf_nodes)
    print("Max leaf nodes: %d  \t\t Accuracy_score:  %f" %(max_leaf_nodes, my_score))
    
    
