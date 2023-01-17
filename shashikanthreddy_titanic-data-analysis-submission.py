import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split, KFold, cross_val_score , cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
#Now importing the dataset into python frame

df_train = pd.read_csv ("../input/train.csv")
df_test = pd.read_csv ("../input/test.csv")

print (df_test.shape)
print (df_train.isnull().sum())
print (df_train.info())
total = df_train.isnull().sum().sort_values(ascending = False)
percentage = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat ([total,percentage], axis = 1, keys =  ['total', 'percentage'])
print (missing_data)
#Checking the dataframe values
print (df_train.head())
# First checking the target feature (column)
sns.countplot (x = "Survived", data = df_train)
plt.legend ("Survival count of the passengers")
plt.show()
# Relating Suvived feature to the "Sex" feature
sns.countplot (x = "Survived" , hue = "Sex" , data = df_train)
plt.show()
# Checking the realtion ship of "Survived" columns with "Pclass"
sns.countplot (x = "Survived" , hue = "Pclass", data = df_train )
plt.show()
#Plotting the "Age" columns from the dataset

df_train ["Age"].plot.hist()
# from this result we see that we have more of young passengers followed by children between 0-10
# Plotting the "Fare" column
df_train ["Fare"].plot.hist(bins = 20)
# Checking out Sibsp column(family relations)
sns.countplot ( x = "SibSp" , data = df_train)
# After performing some intitial analysis, decided to drop few columns now
df_train.drop (['Ticket','PassengerId','Cabin'], axis = 1, inplace = True)
df_test.drop (['Ticket','Cabin'], axis = 1, inplace = True) # didn't drop passengerid column here, will be dropped in the end when doing submission fie

#Lets fill the Embarked feature nan value
#To fill this lets see the most frequently occured value

S = df_test[df_test['Embarked'] == 'S'].shape [0]
print ("S in the data set is " , S)

Q = df_test [df_test ['Embarked'] == 'Q'].shape[0]
print ("Q in the data set is " , Q)
C = df_test [df_test ['Embarked'] == 'C'].shape[0]
print ("C in the data set is " , C)

S = df_train[df_train['Embarked'] == 'S'].shape [0]
print ("S in the data set is " , S)

Q = df_train [df_train ['Embarked'] == 'Q'].shape[0]
print ("Q in the data set is " , Q)
C = df_train [df_train ['Embarked'] == 'C'].shape[0]
print ("C in the data set is " , C)

#So S is the most frequent feature so replacing NAN with S
# Replacing NAN of EMBARKED

df_train['Embarked'] = df_train['Embarked'].fillna ( "S")
df_test['Embarked'] = df_test['Embarked'].fillna ( "S")

#checking the training and testing values to see nan in "Embarked column"
print (df_train.isnull().sum())
print (df_test.isnull().sum())
#Filing the missing values in the age feature, here trying to reate age feature with columns "Name" 
#To do that first extract the name field and relate it to age

df_train ['Title'] = df_train.Name.str.extract (' ([A-Za-z]+)\.', expand=False)
df_test ['Title'] = df_test.Name.str.extract (' ([A-Za-z]+)\.', expand=False)
crosstab_train = pd.crosstab (df_train['Title'], df_train['Sex'])
crosstab_test = pd.crosstab (df_test['Title'], df_test['Sex'])
#print (crosstab_train)

# replacing various titles with common names
combine = [df_train, df_test]
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# map each of the title groups to a numerical values

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
#print (df_train.head())
    
#Now I would like to categorize 

def simplify_ages (df_train , df_test):
    df_train['Age'] = df_train.Age.fillna (-0.5)
    df_test ['Age'] = df_test.Age.fillna (-0.5)
    bins = (-1,0,5,12,18,25,35,60,120)
    print (df_train.isnull().sum())
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut (df_train.Age , bins, labels = group_names)
    categories_test = pd.cut (df_test.Age , bins, labels = group_names)
    df_train.Age = categories
    print (df_train.isnull().sum())
    df_test.Age = categories_test
    print (df_train.isnull().sum())
    return df_train.Age , df_test.Age

def simplify_fares (df_train, df_test):
    df_train ['Fare'] = df_train.Fare.fillna (-0.5)
    df_test ['Fare'] = df_test.Fare.fillna (-0.5)
    bins = ( -1,0, 8,15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut (df_train.Fare, bins, labels = group_names)
    categories_test = pd.cut (df_test.Fare, bins, labels = group_names)
    df_train.Fare = categories
    df_test.Fare = categories_test
    return df_train.Fare, df_test.Fare
df_train.Age , df_test.Age = simplify_ages (df_train , df_test)
df_train.Fare , df_test.Fare = simplify_fares (df_train , df_test)
print (df_train.isnull().sum())
print (df_train.Age[418])
#print (df_train["Title"].head(20))
age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(df_train["Age"])):
    #print (x)
    if df_train["Age"][x] == "Unknown":
        df_train["Age"][x] = age_title_mapping[df_train["Title"][x]]
    
    
for x in range(len(df_test["Age"])):
    #print (x)
    if df_test["Age"][x] == "Unknown":
        df_test["Age"][x] = age_title_mapping[df_test["Title"][x]]
        #print (df_train["Age"][x])
#Now performing age mapping

#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
df_train['Age'] = df_train['Age'].map(age_mapping)
df_test['Age'] = df_test['Age'].map(age_mapping)

#Now the missing values of "Age" is filled now "Name" column needs to be dropped because it cannot have much importance
#our target value
df_train.drop(['Name'], axis = 1, inplace = True)
df_test.drop(['Name'], axis = 1, inplace = True)
print (df_train.head())
print (df_test.head())
print (  "The shape of dataframe train is :" ,df_train.shape)
print ("The shape of dataframe test is :" ,df_test.shape)
#One hot encoding

#Converting categorical values to new columns

df_train_dummy = pd.get_dummies (df_train, drop_first = True)

df_test_dummy = pd.get_dummies (df_test, drop_first = True)
#print (df_train_dummy.head())
print ( "The shape of dataframe train after encoding is :" ,df_train_dummy.shape)
print ( "The shape of dataframe train after encoding is :" ,df_test_dummy.shape)
#Dropping target feature and assigning it to target feature
df_train_target = df_train_dummy ['Survived'] 
df_train_dummy.drop(['Survived'], axis = 1, inplace = True)

X_train,X_validation,y_train,y_validation = train_test_split (df_train_dummy,
                                                              df_train_target, shuffle = False,
                                                              test_size = 0.25, random_state = 42)
## Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform (X_train)
X_validation = sc.transform (X_validation)

### print proportions
print('train: {}% | validation: {}% |'.format(round(len(y_train)/len(df_train_target),2),
                                                       round(len(y_validation)/len(df_train_target),2)))

# create regularization hyperparameter space logistic regression
#penalty = (['l1','l2'])
#C_var = np.array ([10.0,1.0,0.1,0.01,0.001])
#fit_interceptOptions = ([ 'False'])
#solveroptions = ([  'liblinear', 'saga'])
reg= LogisticRegression(random_state = 42)
#reg = GridSearchCV(estimator = clf, param_grid = dict(C= C_var, solver = solveroptions))
reg.fit (X_train, y_train)

#y_predict_train = reg.predict (X_test)
y_predict_val = reg.predict(X_validation)
print ( 'Model score calculated TRAIN DATA:' , reg.score (X_train, y_train))
# print(reg.best_score_)
# print(reg.best_estimator_.solver)



print ('Accuracy score test mode:' ,  accuracy_score (y_validation, y_predict_val))
print (   'Confusion matrix'    , confusion_matrix (y_validation, y_predict_val))
print ('Precision score test mode:' ,  precision_score (y_validation, y_predict_val))
print ('Recall score test mode:' ,  recall_score (y_validation, y_predict_val))
print ('F1 score test mode:' ,  f1_score (y_validation, y_predict_val))
# With other classifier
from sklearn.tree import DecisionTreeClassifier
reg = DecisionTreeClassifier()

reg.fit (X_train, y_train)
y_predict_val = reg.predict (X_validation)
accuracy_prediction = round (accuracy_score (y_predict_val, y_validation) *100,2)
print (accuracy_prediction)
print (   'Confusion matrix'    , confusion_matrix (y_validation, y_predict_val))
print ('Precision score test mode:' , round(precision_score (y_validation, y_predict_val)*100,2))
print ('Recall score test mode:' ,  round (recall_score (y_validation, y_predict_val)*100,2))
print ('F1 score test mode:' ,  round (f1_score (y_validation, y_predict_val)*100,2))
#Checking accuracy with Randomforestclassifier
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier(n_estimators = 500 , max_depth = 7, criterion = 'gini')

reg.fit (X_train, y_train)
y_predict_val = reg.predict (X_validation)
accuracy_prediction = round (accuracy_score (y_predict_val, y_validation) *100,2)
print (accuracy_prediction)
print (   'Confusion matrix'    , confusion_matrix (y_validation, y_predict_val))
print ('Precision score test mode:' , round(precision_score (y_validation, y_predict_val)*100,2))
print ('Recall score test mode:' ,  round (recall_score (y_validation, y_predict_val)*100,2))
print ('F1 score test mode:' ,  round (f1_score (y_validation, y_predict_val)*100,2))
from sklearn.neighbors import KNeighborsClassifier
reg = KNeighborsClassifier(n_neighbors = 5, weights='uniform', algorithm = 'auto')

reg.fit (X_train, y_train)
y_predict_val = reg.predict (X_validation)
accuracy_prediction = round (accuracy_score (y_predict_val, y_validation) *100,2)
print (accuracy_prediction)
print (   'Confusion matrix'    , confusion_matrix (y_validation, y_predict_val))
print ('Precision score test mode:' , round(precision_score (y_validation, y_predict_val)*100,2))
print ('Recall score test mode:' ,  round (recall_score (y_validation, y_predict_val)*100,2))
print ('F1 score test mode:' ,  round (f1_score (y_validation, y_predict_val)*100,2))
# By using gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
reg = GradientBoostingClassifier(n_estimators  = 500)

reg.fit (X_train, y_train)
y_predict_val = reg.predict (X_validation)
accuracy_prediction = round (accuracy_score (y_predict_val, y_validation) *100,2)
print (accuracy_prediction)
print (   'Confusion matrix'    , confusion_matrix (y_validation, y_predict_val))
print ('Precision score test mode:' , round(precision_score (y_validation, y_predict_val)*100,2))
print ('Recall score test mode:' ,  round (recall_score (y_validation, y_predict_val)*100,2))
print ('F1 score test mode:' ,  round (f1_score (y_validation, y_predict_val)*100,2))
#print (df_train_target)
## Feature Scaling again for Kfold
sc = StandardScaler()
df_train_dummy1 = pd.DataFrame(sc.fit_transform (df_train_dummy))

from sklearn.model_selection import KFold  # u should do STANDARD SCALER HERE feature scaling
from sklearn.model_selection import StratifiedKFold
reg = GradientBoostingClassifier(n_estimators  = 500)  # Choosing Gradient boosting estimator because of its good score 
# for F1 and accuracy
def run_kfold(reg):
    kf = KFold (n_splits = 10, shuffle = True, random_state = 45)
     
    outcomes = []
    fold = []
    for train_index, test_index in kf.split (df_train_dummy1, df_train_target):
        #fold +=1
        X_train, X_test = df_train_dummy1.iloc[train_index] , df_train_dummy1.iloc [test_index]
        y_train, y_test = df_train_target.iloc [train_index], df_train_target.iloc [test_index]
        reg.fit (X_train,y_train)
        y_pred = reg.predict (X_test)
        accuracy = accuracy_score (y_test,y_pred)
        outcomes.append (accuracy)
        print("Fold  accuracy: {}".format(accuracy))
    mean_outcome = np.mean (outcomes)
    print ("Mean Accuracy {}" . format (mean_outcome))
run_kfold (reg)
passenger_id = df_test_dummy ['PassengerId']
df_test_dummy.drop (['PassengerId'], axis = 1, inplace = True)
#Submission fil

predict = reg.predict (df_test_dummy)

#Now creating submission dataframe

submission = pd.DataFrame ({'PassengerId' : passenger_id, 'Survived' : predict })
submission.to_csv ('submission.csv', index = False)