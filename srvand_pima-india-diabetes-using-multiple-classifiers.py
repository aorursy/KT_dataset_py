import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#Load the file from local directory using pd.read_csv which is a special form of read_table
#while reading the data, supply the "colnames" list

df = pd.read_csv("../InputData/pima-indians-diabetes.data", names= colnames)
df.shape
df.dtypes
df
pd.DataFrame(
    data={
        'Null': df.isnull().any(),
        'NaN': np.isnan(df).any()
    }
)
df[~df.applymap(np.isreal).all(1)]
df = df.fillna(df.median(),inplace=True)
df
df.describe(include="all")
comparision_dataFrame = (pd.DataFrame(
                                        columns=[
                                                    "Model_Name", 
                                                    "Model_Type",
                                                    "Train_score", 
                                                    "Test_score"
                                        ]
                                    )
                        )
comparision_dataFrame
array = df.values
X = array[:,0:7] # select all rows and first 8 columns which are the attributes
Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, Y_train,Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

print(X_test.shape,Y_test.shape)
Model_Name="Logistic Regression"
Model_Type="Normal"
logisticRegression = LogisticRegression()
model = logisticRegression.fit(X_train, Y_train)
train_score = model.score(X_train, Y_train)
Y_predict = model.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
print(accuracy_score(Y_test,Y_predict))
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame
Model_Name="Naive Bayes"
Model_Type="Normal"
naive_bayes = GaussianNB()
model = naive_bayes.fit(X_train,Y_train)
train_score = model.score(X_train, Y_train)
Y_predict = model.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
print(accuracy_score(Y_test,Y_predict))
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame
Model_Name="KNN"
Model_Type="Normal"
kNeighborsClassifier = KNeighborsClassifier()
model = kNeighborsClassifier.fit(X_train, Y_train)
train_score = model.score(X_train, Y_train)
Y_predict = kNeighborsClassifier.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
print(accuracy_score(Y_test,Y_predict))
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame

for iter_max_depth in range(2,20):
    dt = (DecisionTreeClassifier(
                                            criterion = 'entropy',
                                            max_depth = iter_max_depth
                                    )
                   )
    model = dt.fit(X_train, Y_train)
    train_score = model.score(X_train, Y_train)
    Y_predict = model.predict(X_test)
    test_score = model.score(X_test, Y_test)
    print("Training Score is: ",train_score)
    print("Test Score is: ",test_score)
    print(metrics.confusion_matrix(Y_test,Y_predict))
    temp_accuracy_Score=accuracy_score(Y_test,Y_predict)
    # Accuracy
    print("Max_Depth_{0}, Accuracy Score : {1}".format(iter_max_depth, temp_accuracy_Score))
    df_confusion = (
                pd.crosstab(np.array(Y_test).flatten(),Y_predict, 
                rownames=['Actual'],  colnames=['Predicted'])
    )
Model_Name="Decision Tree"
Model_Type="Normal"
dt = (DecisionTreeClassifier(
                                            criterion = 'entropy',
                                            max_depth = 1
                                    )
                   )
model = dt.fit(X_train, Y_train)
train_score = model.score(X_train, Y_train)
Y_predict = model.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
temp_accuracy_Score=accuracy_score(Y_test,Y_predict)
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame
Model_Name="Random Forest Classifier"
Model_Type="Ensembles"
randomForestClassifier = RandomForestClassifier(criterion = 'entropy', max_depth=9)
model = randomForestClassifier.fit(X_train,Y_train)
train_score = model.score(X_train, Y_train)
Y_predict = model.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
print(accuracy_score(Y_test,Y_predict))
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame
Model_Name="Bagging Classifier"
Model_Type="Ensembles"
baggingClassifier = BaggingClassifier()
model = baggingClassifier.fit(X_train, Y_train)
train_score = model.score(X_train, Y_train)
Y_predict = model.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
print(accuracy_score(Y_test,Y_predict))
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame
Model_Name="Ada Boost Classifier"
Model_Type="Ensembles"
adb_regressor = AdaBoostClassifier()
model = adb_regressor.fit(X_train, Y_train)
train_score = model.score(X_train, Y_train)
Y_predict =model.predict(X_test)
test_score = model.score(X_test, Y_test)
print("Training Score is: ",train_score)
print("Test Score is: ",test_score)
print(metrics.confusion_matrix(Y_test,Y_predict))
print(accuracy_score(Y_test,Y_predict))
result_df = pd.DataFrame([[Model_Name,Model_Type,train_score,test_score]], columns=comparision_dataFrame.columns)
comparision_dataFrame = comparision_dataFrame.append(result_df)
comparision_dataFrame
comparision_dataFrame.index = list(range(1,len(comparision_dataFrame)+1))
(comparision_dataFrame
 .sort_values(
             by=['Train_score', 'Test_score'], 
             inplace=True, 
             ascending=False
 )
)
comparision_dataFrame