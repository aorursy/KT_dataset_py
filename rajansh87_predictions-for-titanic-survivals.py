import pandas as pd
data=pd.read_csv("../input/titanic/train.csv")
data.head()
data.shape
data.info()
# we can see we have null values in age and cabin column
data.columns
import seaborn as sns

sns.set()
gender = data["Sex"]
# bar plot

# we create graph only of catagorical data

sns.countplot(gender)
sns.countplot(data["Survived"],hue="Sex",data=data)
sns.countplot(data["Survived"],hue="Pclass",data=data)
age=data["Age"]
# univariate : histogram :freq disttribution

fare=data["Fare"]
type(fare)
fare.hist(bins=50, color="red", figsize=(10,5))   #histogram
#fare.plot()
# check missing values

data.isnull()
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")

# cbar for scale at right side

# yticklabel for left side

# cmap for color theme

# light color means null values
# so we can drop cabin from our data also
sns.distplot(age.dropna(),bins=50) #histogram  and density curver

# bins for group ex:group of 50
sns.countplot(data["SibSp"],data=data,hue="Survived")     # siblings
data.columns
age.mean()
sns.boxplot(data=data,y="Age",x="Pclass")
overall_mean=age.mean()
mean_class_1 = data["Age"][data["Pclass"]==1].mean() #mean of age of people of Pclass==1
mean_class_2 = data["Age"][data["Pclass"]==2].mean() #mean of age of people of Pclass==1
mean_class_3 = data["Age"][data["Pclass"]==3].mean()  #mean of age of people of Pclass==1
def impute_age(cols):            #replace Missing Age

    age=cols[0]

    Pclass=cols[1]

    if pd.isnull(age):

        if Pclass==1:

            return int(mean_class_1)

        elif Pclass==2:

            return int(mean_class_2)

        elif Pclass==3:

            return int(mean_class_3)

        else:

            return int(overall_mean)

        

    else:

        return age

        

        
data["Age"] = data[["Age","Pclass"]].apply(impute_age,axis=1)   # without loop these values will go inside this function and will return back



# axis = 0 for row wise operation

# axis = 1 for column wise operation

# this takes tuple

sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")
# hence we removed null values from age
data.drop("Cabin",axis=1,inplace=True)   # drop cabin
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")
 #########################
y=data["Survived"]
data.columns
X= data[["Pclass","Sex","Age","SibSp","Parch","Embarked"]]
X
# since sex is catagorical and also a string so we have to do label encoding i.e dummy variable using One Hot Encoding andthen remove one dummy variable for avoiding dummy trap
sex=data["Sex"]
sex=pd.get_dummies(sex, drop_first=True)
X
# also Pclass is catagorical sowe have to do the same
pclass=data["Pclass"]
pclass=pd.get_dummies(pclass,drop_first="True")
X
#similary SibSp and Parch  and Embarked are also catagorical
sibsp=data["SibSp"]

parch=data["Parch"]

embarked=data["Embarked"]
sibsp=pd.get_dummies(sibsp,drop_first=True)

parch=pd.get_dummies(parch,drop_first=True)

embarked=pd.get_dummies(embarked,drop_first=True)
age= data["Age"]
X = pd.concat([age,embarked,parch,sibsp,sex,pclass],axis=1) #concanate multiple dataframes   
X
####### remove it later
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X,y)
model.coef_
test=pd.read_csv("../input/titanic/test.csv")
#we have to process this data as well similary we had done for train data
test
# check missing values

test.isnull()
sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap="viridis")

# cbar for scale at right side

# yticklabel for left side

# cmap for color theme

# light color means null values
# so we can drop cabin from our data also
test.columns
age.mean()
overall_mean=age.mean()
mean_class_1 = test["Age"][test["Pclass"]==1].mean() #mean of age of people of Pclass==1
mean_class_2 = test["Age"][test["Pclass"]==2].mean() #mean of age of people of Pclass==1
mean_class_3 = test["Age"][test["Pclass"]==3].mean()  #mean of age of people of Pclass==1
def impute_age(cols):            #replace Missing Age

    age=cols[0]

    Pclass=cols[1]

    if pd.isnull(age):

        if Pclass==1:

            return int(mean_class_1)

        elif Pclass==2:

            return int(mean_class_2)

        elif Pclass==3:

            return int(mean_class_3)

        else:

            return int(overall_mean)

        

    else:

        return age

        

        
test["Age"] = test[["Age","Pclass"]].apply(impute_age,axis=1)   # without loop these values will go inside this function and will return back



# axis = 0 for row wise operation

# axis = 1 for column wise operation

# this takes tuple

sns.heatmap(test.isnull(), cbar=False, yticklabels=False, cmap="viridis")
# hence we removed null values from age
test.drop("Cabin",axis=1,inplace=True)   # drop cabin
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")
# since sex is catagorical and also a string so we have to do label encoding i.e dummy variable using One Hot Encoding andthen remove one dummy variable for avoiding dummy trap
sex=test["Sex"]
sex=pd.get_dummies(sex, drop_first=True)
# also Pclass is catagorical sowe have to do the same
pclass=test["Pclass"]
pclass=pd.get_dummies(pclass,drop_first="True")
#similary SibSp and Parch  and Embarked are also catagorical
sibsp=test["SibSp"]

parch=test["Parch"]

embarked=test["Embarked"]
sibsp=pd.get_dummies(sibsp,drop_first=True)

parch=pd.get_dummies(parch,drop_first=True)

embarked=pd.get_dummies(embarked,drop_first=True)
age= test["Age"]
X_test = pd.concat([age,embarked,parch,sibsp,sex,pclass],axis=1) #concanate multiple dataframes   
X_test.drop(9,axis=1,inplace=True)   # drop cabin
X_test
X
y_pred=model.predict(X_test)
y_pred
ans=pd.read_csv("../input/titanic/gender_submission.csv")
ans
y_test=ans["Survived"]
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
# this shows that:

# array([[152,  23],

#       [  85, 35]], dtype=int64)

# 152 and 85 predicted values were right and 23 and 35 values were wrong

# 1st column is for dead  and 2nd column is for survived

# i.e in 152+35=187 people our model predicted correctly for 152 ( i.e True Negative )

# similary in 23+85=108 people our model predicted correctly for 85 ( i.e True Positive )

#

#

# confusion matrix= [[TrueNegative,FalsePositive],

#                     [FalseNegative,TruePositive]]
y_test.shape
total_records=152+23+35+85
total_records
correct_answer=152+85
correct_answer
wrong_answer=35+23
wrong_answer
accuracy=correct_answer/total_records*100
accuracy
error=35+23
error_per=error/total_records*100
error_per   #error percentage
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
type(y_pred)
p=pd.DataFrame(y_pred, columns=['Survived']).to_csv('mysolution.csv')   #save predictions to csv file
