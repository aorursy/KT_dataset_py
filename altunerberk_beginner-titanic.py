import pandas as pd #Data processing
from sklearn.linear_model import LogisticRegression #For the machine learning modelling
from sklearn.metrics import accuracy_score #Calculating accuracy for end of the evulation
from sklearn.neighbors import KNeighborsClassifier #KNeighborsClassifier Algorithm in Sklearn Library
from sklearn.tree import DecisionTreeClassifier #Decision Tree Algorithm in Sklearn Library
from sklearn.ensemble import RandomForestClassifier #RandomForest Algortihm in Sklearn Library
from sklearn.naive_bayes import GaussianNB #Naive Bayes Algorithm
from sklearn.svm import SVC #Support Vector Machine
from sklearn.model_selection import cross_val_score,train_test_split #Data proccessing for the modelling
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt #Visual modelling
import seaborn as sns

test_df = pd.read_csv("../input/test.csv", sep=",")
train_df = pd.read_csv("../input/train.csv", sep=",")
# Input data files in program.


train_df.info()
print("_"*50)
test_df.info()
#For the comparisons train and test.
train_df.describe(include='all') #Statical values from train_df
train_df.head() #First 5 value from to train_df
#Creat the chosen column and survive value comparision
def chart(feature):
  survived = train_df[train_df["Survived"] == 1] #Transfer survival values of 1 to a new series
  died = train_df[train_df["Survived"] == 0]#Transfer survival values of 0 to a new series
  survived[feature].plot.hist(alpha=0.5,color='red',bins=25)#Survived[column we chose].plot.hist(alpha=Visibilty value, color=color of graphic,bins=width of boxes)
  died[feature].plot.hist(alpha=0.5,color='blue',bins=25)
  plt.legend(['Survived','Died'])
  plt.show()
#We write "Age" for age-survive value comprasion    
chart("Age")
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived') #Comprasion female and male about alive or dead.
#Cut the age and create new age categories. These catagories are more cleaner our data
def cutting_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5) #Some age values are missing. We fill -0.5 these NaN values.
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names) # pd.cut function cut from the points we want
    return df

cut_points = [-1,0,5,12,18,35,60,100] #cut points we chose
label_names = ["Missing","Baby","Child","Teenager","Young Adult","Adult","Old"] #Every 2 range will describe one label in order.

train_df = cutting_age(train_df,cut_points,label_names) #Cut Age column in train data
test_df = cutting_age(test_df,cut_points,label_names) #Cut Age column in test data

sns.barplot(x="Age_categories", y="Survived", data=train_df)
#Cutting fare values for clean data.
def cutting_fare(df,label_names): 
    df["Fare_categories"] = pd.qcut(df["Fare"],4,labels=label_names) # pd.qcut() function cut from the equal points. We choose 4 for that value.
    return df

label_names = ["Least","Less-Middle","Middle","High"]

train_df = cutting_fare(train_df,label_names)
test_df = cutting_fare(test_df,label_names)

sns.barplot(x="Fare_categories", y="Survived", data=train_df)

train_df[["Fare_categories", "Survived"]].groupby(['Fare_categories'], as_index=False).mean().sort_values(by='Survived') #Every fare catagories survive values.
#We create dummies for every created values. After this process we have a lot of columns for every value.
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name) #Seperate to columns for every diffrent value
    df = pd.concat([df,dummies],axis=1) # created columns adding to  actual data 
    return df

for column in ["Pclass","Sex","Age_categories","Embarked","Fare_categories"]: #Columns who diveded
    train_df = create_dummies(train_df,column)
    test_df = create_dummies(test_df,column)
train_df = train_df.drop(['Name','Pclass','Age','Ticket','Sex','SibSp','Parch','Fare','Cabin','Embarked','Age_categories','Fare_categories'], axis=1) 
#We dont need the other columns. We drop columns for cleaner data
train_df.head() # First 5 value of new train_df 
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male','Age_categories_Missing','Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young Adult', 'Age_categories_Adult','Age_categories_Old','Embarked_C','Embarked_S','Embarked_Q','Fare_categories_Least','Fare_categories_Less-Middle',
           'Fare_categories_Middle','Fare_categories_High'] #We use these columns


#Splitting data
X = train_df[columns] #Train data
y = train_df['Survived'] #Targer data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20,random_state=15) #Splitting data to train and test datas. We chose the test size %20.
#Random state is change rate to data when program started.

#Cross Validation function. We call this function when try every model.
def cr_val(model,tr_data,test_data):
    accuracy = (cross_val_score(model, tr_data, test_data, cv=10)).mean() #cross_val_score(ModelWeUse, TrainData, TestData, cv=Number of pieces data will we divide).
    # .mean() is for take the average after getting the results
    return accuracy
#Logistic Regression process
lr=LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)
accuracy = accuracy_score(test_y, predictions)

#Now call the Cross Validation Function
accuracy = cr_val(lr,X,y)

print(accuracy)

hyperparameters = {
    "n_neighbors": range(1,20),
}
knn=KNeighborsClassifier()
grid=GridSearchCV(knn,param_grid=hyperparameters,cv=10)
grid.fit(X,y)
best_score=grid.best_score_

print(best_score)

hyperparameters={ "criterion":["entropy","gini"], 
                "max_depth":[5,10],
                "max_features":["log2","sqrt"],
                "min_samples_leaf":[1,5],
                "min_samples_split":[3,5],
                "n_estimators":[6,9],
               }

clf=RandomForestClassifier(random_state=1)
grid=GridSearchCV(clf,param_grid=hyperparameters,cv=10)
grid.fit(X,y)
best_params=grid.best_params_
best_score=grid.best_score_

print(best_score)
clf=DecisionTreeClassifier()

#Cross Validation Function
accuracy=cr_val(clf,X,y)
print(accuracy)
clf=GaussianNB()

#Cross Validation Function
accuracy=cr_val(clf,X,y)
print(accuracy)
clf=SVC()

#Cross Validation Function
accuracy=cr_val(clf,X,y)
print(accuracy)

best_rf=grid.best_estimator_
test_predictions=best_rf.predict(test_df[columns])
submission_df = {"PassengerId":test_df["PassengerId"],
                 "Survived": test_predictions}
submission = pd.DataFrame(submission_df)
print(submission)
submission.to_csv("submission.csv",index=False)