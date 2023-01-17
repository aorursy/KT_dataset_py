import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#create train dataframe
train_df=pd.read_csv("../input/train.csv")
#create test dataframe
test_df=pd.read_csv("../input/test.csv")
#view my train data
train_df.head()
print("Train data Information")
train_df.info()
print("Test data Information")
test_df.info()
print("######UNIQUE VALUES#######")
print("Unique values for Embarked")
print(pd.unique(train_df['Embarked']))
print("Unique values for Pclass")
print(pd.unique(train_df['Pclass']))
print("Unique values for Survived")
print(pd.unique(train_df['Survived']))
print("Unique values for Sex")
print(pd.unique(train_df['Sex']))
print("Unique values for SibSp")
print(pd.unique(train_df['SibSp']))
print("Unique values for Parch")
print(pd.unique(train_df['Parch']))
print("")
print("########TRAIN TEST#########")
print("No of NULL/NaN in Cabin")
print(sum(pd.isnull(train_df['Cabin'])))
print("No of NULL/NaN in Age")
print(sum(pd.isnull(train_df['Age'])))
print("No of NULL/NaN in Embarked")
print(sum(pd.isnull(train_df['Embarked'])))
print("")
print("########TRAIN TEST#########")
print("No of NULL/NaN in Cabin")
print(sum(pd.isnull(test_df['Cabin'])))
print("No of NULL/NaN in Age")
print(sum(pd.isnull(test_df['Age'])))
print("No of NULL/NaN in Fare")
print(sum(pd.isnull(test_df['Fare'])))

combined_df=pd.concat([train_df,test_df])
print("mean: ",combined_df["Age"].mean())
print("median: ",combined_df["Age"].median())
print("mode: ",combined_df["Age"].mode())
x=combined_df.groupby(["Age"])["Age"].agg(["count"]).reset_index()
plt.bar(x["Age"],x["count"])
plt.show()

combined_df.loc[combined_df["Age"].isnull()].groupby(["Pclass","Sex"])["Sex"].agg(["count"])
x=combined_df.loc[combined_df["Age"].isnull()].groupby(["Fare"])["Fare"].agg(["count"])
plt.plot(x)
plt.show()
combined_df.groupby(["Sex","SibSp","Parch"])["Age"].agg(["median"])
combined_df.loc[combined_df["SibSp"]==8]
xx=combined_df.groupby(["Fare"])["Age"].agg(["mean"])
plt.plot(xx)
plt.show()

#Name title prefix
combined_df["Name_key"]=""
combined_df["Name_key"]=combined_df["Name"].str.split(',',expand=True)[1].str.split(' ',expand=True)[1]
combined_df.groupby(["Sex","Name_key","Parch"])["Age"].agg(["median"])
combined_df["Fare_Group"]=pd.cut(combined_df["Fare"],range(0,350,50),right=False)
combined_df["Fare_Group"]=combined_df["Fare_Group"].astype("object")
combined_df["Fare_Group"].fillna("[300,600)",inplace=True)
#Final Age median values depending on the key features to which the Age depends on
Age_values=combined_df.groupby(["Sex","Pclass","Name_key","Parch"])["Age"].agg(["median"]).reset_index()
Age_values.loc[Age_values["median"].isnull()]
null_age=combined_df.loc[combined_df["Age"].isnull()][['PassengerId','Sex','Name_key','Pclass','Parch']]
Age_values_f=pd.merge(Age_values,null_age,how='inner',on=['Sex','Name_key','Pclass','Parch'])
Age_values_f=Age_values_f.rename(columns={'median':"Age"})
Age_values_f.loc[Age_values_f["Age"].isnull()] #gives passengerID 1257,980,1234
combined_df.loc[combined_df["PassengerId"].isin([1257,980,1234])]
combined_df.groupby(["Sex"])["Age"].agg(["median"])
#substitute the 3 missing ages with median value for individual sex
Age_values_f.loc[(Age_values_f["Age"].isnull())& (Age_values_f["Sex"]=='female'),"Age"]=27.0
Age_values_f.loc[(Age_values_f["Age"].isnull())& (Age_values_f["Sex"]=='male'),"Age"]=28.0
#Update missing age in train data
train_df.set_index('PassengerId',inplace=True)
Age_values_f.set_index('PassengerId',inplace=True)
train_df.update(Age_values_f)
train_df.reset_index(inplace=True)

#Update missing age in test data
test_df.set_index('PassengerId',inplace=True)
test_df.update(Age_values_f)
test_df.reset_index(inplace=True)

train_df.groupby(["Embarked"])["Embarked"].agg(["count"])
train_df.loc[train_df["PassengerId"]==62.0,"Embarked"]='S'
train_df.loc[train_df["PassengerId"]==830,"Embarked"]='S'
train_df["Cabin_prefix"]=train_df["Cabin"].str[0:1]
train_df.groupby(["Cabin_prefix","Pclass"])["Pclass"].agg(["count"]).reset_index()
people=train_df.groupby(["Cabin_prefix"])["Cabin_prefix"].agg(["count"]).reset_index()
people["survived"]=train_df.loc[train_df["Survived"]==1].groupby(["Cabin_prefix"])["Cabin_prefix"].agg(["count"]).reset_index()["count"]
people["per_survived"]=(people["survived"]/people["count"])*100
plt.bar(people["Cabin_prefix"],people["per_survived"])
plt.show()
#dropping Cabin columns
train_df.drop(columns={"Cabin","Cabin_prefix"},inplace=True)
#take a look on the data
test_df.loc[test_df["Fare"].isnull()]
#Checking mean values for age>50 
train_df.loc[train_df["Pclass"]==3].loc[train_df["Age"]>50].groupby(["Age","Sex"])["Fare"].agg(["mean"])
train_df.loc[train_df["Pclass"]==3].loc[train_df["Age"]>50].groupby(["Sex"])["Fare"].agg(["mean"])
test_df.loc[test_df["Fare"].isnull(),"Fare"]=7.518522
#copy the required coloumns in a seperate dataframe
related_people=combined_df[["PassengerId","Name","SibSp","Parch","Ticket","Embarked"]].copy()
related_people["Last_Name"]=""
related_people["Last_Name"]=related_people["Name"].str.split(",",expand=True)[0]
related_people["total_related"]=related_people["SibSp"]+related_people["Parch"]+1

X=related_people.loc[(related_people["SibSp"]>0)|(related_people["Parch"]>0)].groupby(["Last_Name","total_related"])["PassengerId"].agg(["count"])
X.reset_index(inplace=True)

#Group related people where titles are same and the SibSp and Parch count matches
Y=pd.DataFrame(X.loc[X["count"]==X["total_related"]]["Last_Name"])

######If we explore the data in Y we can see for 'Carter' we have two rows so we need to take special care of this.
######Z=Y.groupby(["Last_Name"])["Last_Name"].agg(["count"])
######Z.loc[Z["count"]>1]

Y["RGroup"]=Y["Last_Name"]
Y.drop_duplicates(inplace=True)
Y.set_index("Last_Name",inplace=True)
related_people["RGroup"]=""
related_people.set_index("Last_Name",inplace=True)
related_people.update(Y)
related_people.reset_index(inplace=True)

#update the RGroup for 'Carter'
#related_people.loc[related_people["RGroup"]=='Carter']
#tickets are 113760,224252
related_people.loc[related_people["Ticket"]=='113760',"RGroup"]="Carter_1"
related_people.loc[related_people["Ticket"]=='224252',"RGroup"]="Carter_2"


####Update group for people who are travelling alone
Y=pd.DataFrame(related_people.loc[(related_people["SibSp"]==0)&(related_people["Parch"]==0)][["Last_Name","Ticket"]])
Y["RGroup"]=Y["Last_Name"]+'_'+Y["Ticket"]
Y.drop_duplicates(inplace=True)
Y.set_index(["Last_Name","Ticket"],inplace=True)
related_people.set_index(["Last_Name","Ticket"],inplace=True)
related_people.update(Y)
related_people.reset_index(inplace=True)
####Update group for people who are related by ticket number
X=related_people.loc[related_people["RGroup"]==""].groupby(["Ticket","total_related"])["PassengerId"].agg(["count"])
X.reset_index(inplace=True)
#Group related people where tickets are same and the SibSp and Parch count matches
Y=pd.DataFrame(X.loc[X["count"]==X["total_related"]]["Ticket"])
Y["RGroup"]=Y["Ticket"]+'_R'
Y.drop_duplicates(inplace=True)
Y.set_index("Ticket",inplace=True)
related_people.set_index("Ticket",inplace=True)
related_people.update(Y)
related_people.reset_index(inplace=True)

related_people.loc[related_people["PassengerId"]==249,"RGroup"]="Beckwith_M"
related_people.loc[related_people["PassengerId"]==872,"RGroup"]="Beckwith_M"
related_people.loc[related_people["PassengerId"]==137,"RGroup"]="Beckwith_M"
related_people.loc[related_people["PassengerId"]==572,"RGroup"]="Lamson_M"
related_people.loc[related_people["PassengerId"]==1248,"RGroup"]="Lamson_M"
related_people.loc[related_people["PassengerId"]==969,"RGroup"]="Lamson_M"
related_people.loc[related_people["PassengerId"]==1200,"RGroup"]="Hays_M"
related_people.loc[related_people["PassengerId"]==821,"RGroup"]="Hays_M"
related_people.loc[related_people["PassengerId"]==588,"RGroup"]="Frolicher_M"
related_people.loc[related_people["PassengerId"]==1289,"RGroup"]="Frolicher_M"
related_people.loc[related_people["PassengerId"]==540,"RGroup"]="Frolicher_M"
related_people.loc[related_people["PassengerId"]==581,"RGroup"]="Jacobsohn_M"
related_people.loc[related_people["PassengerId"]==1133,"RGroup"]="Jacobsohn_M"
related_people.loc[related_people["PassengerId"]==218,"RGroup"]="Jacobsohn_M"
related_people.loc[related_people["PassengerId"]==601,"RGroup"]="Jacobsohn_M"
related_people.loc[related_people["PassengerId"]==756,"RGroup"]="Hamalainen_M"
related_people.loc[related_people["PassengerId"]==248,"RGroup"]="Hamalainen_M"
related_people.loc[related_people["PassengerId"]==1130,"RGroup"]="Hamalainen_M"
related_people.loc[related_people["PassengerId"]==313,"RGroup"]="Lahtinen_M"
related_people.loc[related_people["PassengerId"]==1041,"RGroup"]="Lahtinen_M"
related_people.loc[related_people["PassengerId"]==418,"RGroup"]="Lahtinen_M"
related_people.loc[related_people["PassengerId"]==530,"RGroup"]="Hocking_M"
related_people.loc[related_people["PassengerId"]==944,"RGroup"]="Hocking_M"
related_people.loc[related_people["PassengerId"]==775,"RGroup"]="Hocking_M"
related_people.loc[related_people["PassengerId"]==832,"RGroup"]="Richards_M"
related_people.loc[related_people["PassengerId"]==408,"RGroup"]="Richards_M"
related_people.loc[related_people["PassengerId"]==438,"RGroup"]="Richards_M"
related_people.loc[related_people["PassengerId"]==105,"RGroup"]="Gustafsson_M"
related_people.loc[related_people["PassengerId"]==393,"RGroup"]="Gustafsson_M"
related_people.loc[related_people["PassengerId"]==207,"RGroup"]="Gustafsson_M"
related_people.loc[related_people["PassengerId"]==86,"RGroup"]="Gustafsson_M"
related_people.loc[related_people["PassengerId"]==69,"RGroup"]="3101281"
related_people.loc[related_people["PassengerId"]==480,"RGroup"]="Hirvonen_M"
related_people.loc[related_people["PassengerId"]==896,"RGroup"]="Hirvonen_M"
related_people.loc[related_people["PassengerId"]==477,"RGroup"]="Renouf_M"
related_people.loc[related_people["PassengerId"]==727,"RGroup"]="Renouf_M"
related_people.loc[related_people["PassengerId"]==70,"RGroup"]="Kink_M"
related_people.loc[related_people["PassengerId"]==1268,"RGroup"]="Kink_M"
related_people.loc[related_people["PassengerId"]==185,"RGroup"]="Kink_M"
related_people.loc[related_people["PassengerId"]==1286,"RGroup"]="Kink_M"
related_people.loc[related_people["PassengerId"]==1057,"RGroup"]="Kink_M"
related_people.loc[related_people["PassengerId"]==1037,"RGroup"]="Vander Planke_M"
related_people.loc[related_people["PassengerId"]==19,"RGroup"]="Vander Planke_M"
related_people.loc[related_people["PassengerId"]==39,"RGroup"]="Vander Planke_M"
related_people.loc[related_people["PassengerId"]==334,"RGroup"]="Vander Planke_M"
related_people.loc[related_people["PassengerId"]==206,"RGroup"]="Strom_M"
related_people.loc[related_people["PassengerId"]==252,"RGroup"]="Strom_M"
related_people.loc[related_people["PassengerId"]==443,"RGroup"]="347076"
related_people.loc[related_people["PassengerId"]==268,"RGroup"]="Strom_M"
related_people.loc[related_people["PassengerId"]==1106,"RGroup"]="347091"
related_people.loc[related_people["PassengerId"]==193,"RGroup"]="350046"
related_people.loc[related_people["PassengerId"]==722,"RGroup"]="350048"
related_people.loc[related_people["PassengerId"]==893,"RGroup"]="Hocking_M"
related_people.loc[related_people["PassengerId"]==41,"RGroup"]="7546"
related_people.loc[related_people["PassengerId"]==566,"RGroup"]="Davies_M"
related_people.loc[related_people["PassengerId"]==901,"RGroup"]="Davies_M"
related_people.loc[related_people["PassengerId"]==1079,"RGroup"]="Davies_M"
related_people.loc[related_people["PassengerId"]==923,"RGroup"]="Renouf_M"
related_people.loc[related_people["PassengerId"]==1211,"RGroup"]="Renouf_M"
related_people.loc[related_people["PassengerId"]==672,"RGroup"]="Hays_M"
related_people.loc[related_people["PassengerId"]==984,"RGroup"]="Hays_M"
related_people.loc[related_people["PassengerId"]==274,"RGroup"]="PC 17596"
related_people.loc[related_people["PassengerId"]==665,"RGroup"]="Hirvonen_M"

related_people.drop(columns={"Ticket","Last_Name","Name","SibSp","Parch","total_related"})
related_people.set_index("PassengerId",inplace=True)
train_df["RGroup"]=""
train_df.set_index("PassengerId",inplace=True)
train_df.update(related_people)
train_df.reset_index(inplace=True)
related_people.reset_index(inplace=True) 
 
related_people.set_index("PassengerId",inplace=True)
test_df["RGroup"]=""
test_df.set_index("PassengerId",inplace=True)
test_df.update(related_people)
test_df.reset_index(inplace=True)
related_people.reset_index(inplace=True)
train_df["Sex"]=train_df["Sex"].astype("category")
train_df["Pclass"]=train_df["Pclass"].astype("category")
train_df["Embarked"]=train_df["Embarked"].astype("category")
train_df["RGroup"]=train_df["RGroup"].astype("category")
#dropping column name and fare_group and Name Key
train_df.drop(columns={"Name","Ticket","PassengerId"},inplace=True)
#denormalize category data
train_df=pd.get_dummies(train_df,columns=["Pclass"])
train_df=pd.get_dummies(train_df,columns=["Sex"])
train_df=pd.get_dummies(train_df,columns=["Embarked"])

train_df.rename(columns={"Pclass_1.0":"Pclass_1","Pclass_2.0":"Pclass_2","Pclass_3.0":"Pclass_3"},inplace=True)

#categorical data
test_df["Sex"]=test_df["Sex"].astype("category")
test_df["Pclass"]=test_df["Pclass"].astype("category")
test_df["Embarked"]=test_df["Embarked"].astype("category")
test_df["RGroup"]=test_df["RGroup"].astype("category")
#test_df["Name_key"]=test_df["Name_key"].astype("category")
#dropping column name and Cabin 
test_passengerId=test_df["PassengerId"].copy()
test_df.drop(columns={"Name","Cabin","Ticket","PassengerId"},inplace=True)
#denormalize category data
test_df=pd.get_dummies(test_df,columns=["Pclass"])
test_df=pd.get_dummies(test_df,columns=["Sex"])
test_df=pd.get_dummies(test_df,columns=["Embarked"])
#test_df=pd.get_dummies(test_df,columns=["Name_key"])
import seaborn as sns

corr_data=train_df.corr(method='pearson')
fig,ax=plt.subplots(figsize=(15,8))
mask=np.zeros_like(corr_data)
mask[np.tril_indices_from(mask)]=True
ax=sns.heatmap(corr_data,mask=mask,cmap="Blues",annot=True)
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split

# split data in test and cross validation
train_df_dec=train_df.copy()
train_df_dec = train_df_dec.sample(frac=1).reset_index(drop=True) #shuffle the train data
row_count = train_df_dec.shape[0]
split_point = int(row_count*0.30)
cv_data, train_data = train_df_dec[:split_point].copy(), train_df_dec[split_point:].copy()
#calculate 'percentage' -survival ration in same RGroup
train_df_dec
R=train_data.groupby(["RGroup"])["Survived"].agg(["count","sum"])
R.reset_index(inplace=True)
R["percentage"]=R["sum"]/R["count"]
R.drop(columns={"count","sum"},inplace=True)
R.set_index("RGroup",inplace=True)
train_data["percentage"]=np.nan
train_data.set_index("RGroup",inplace=True)
train_data.update(R)
train_data.reset_index(inplace=True)
R.reset_index(inplace=True) 
#train_data["percentage"]=train_data["percentage"].astype("float64")
R.set_index("RGroup",inplace=True)
cv_data["percentage"]=np.nan
cv_data.set_index("RGroup",inplace=True)
cv_data.update(R)
cv_data.reset_index(inplace=True)
R.reset_index(inplace=True)
cv_data.loc[(cv_data["percentage"].isnull()),"percentage"]=0.5 #replace null values with 
R.set_index("RGroup",inplace=True)
test_df["percentage"]=np.nan
test_df.set_index("RGroup",inplace=True)
test_df.update(R)
test_df.reset_index(inplace=True)
R.reset_index(inplace=True)
test_df.loc[(test_df["percentage"].isnull()),"percentage"]=0.5 #replace null values with 



#checking accuracy score for different dept
Y_train=train_data["Survived"].copy()
X_train=train_data.drop(columns={"Survived"})
Y_cv=cv_data["Survived"].copy()
X_cv=cv_data.drop(columns={"Survived"})

X_train.drop(columns={"RGroup"},inplace=True)
X_cv.drop(columns={"RGroup"},inplace=True)

scores=[]
for x in range(1,7):
    clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=x) #I have tested and found the default depth is 11; accuracy score on validation set is 70% only
    clf=clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_cv)
    scores.append(metrics.accuracy_score(Y_cv,Y_pred))
print(scores)
X_train.drop(columns={"percentage"},inplace=True)
X_cv.drop(columns={"percentage"},inplace=True)
test_df.drop(columns={"percentage"},inplace=True)
test_df.drop(columns={"RGroup"},inplace=True)
scores=[]
for x in range(1,18):
    clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=x) #I have tested and found the default depth is 11; accuracy score on validation set is 70% only
    clf=clf.fit(X_train,Y_train)
    Y_pred=clf.predict(X_cv)
    scores.append(metrics.accuracy_score(Y_cv,Y_pred))
print(scores)
train_df_dec.drop(columns={"RGroup"},inplace=True)
Target_df=train_df_dec["Survived"]
train_df_dec.drop(columns={"Survived"},inplace=True)
from sklearn.model_selection import cross_val_score
clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=4) 
scores_cv = cross_val_score(clf, train_df_dec,Target_df, cv=10)
scores_cv
# max performance is in depth 3, depth 4 is near to it
clf=tree.DecisionTreeClassifier(criterion='entropy',max_depth=4) 
clf=clf.fit(X_train,Y_train)
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=X_train.columns.values.tolist()) 
graph = graphviz.Source(dot_data)
graph
test_df.head()
X_train.head()
Y_pred=clf.predict(test_df)
my_submission = pd.DataFrame({'PassengerId': test_passengerId, 'Survived': Y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

