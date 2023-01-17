import os
os.getcwd() #Current working directory.
os.chdir('../input/titanic/') #Change the cwd to where the csv files are located.
import pandas as pd
import numpy as np
dataset = pd.read_csv('train.csv').dropna(subset = ['Embarked']) #Load the dataframe with train data, also drop rows with empty Embarked fields.
X = dataset.iloc[:, [2,4,5,6,11]].values   #This will hold all columns we need to train our model. These are PassengerClass,
                                           #Sex, Age, SiblingSpouse and Embarked.These form our Independent Variable 
y = dataset.iloc[:, 1].values   #This has our data on the people survived. Dependent Variable.
#Taking care of missing values in age column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')
imputer = imputer.fit(X[:, 2:3]) #Fitting the values from Age column with index 2 in our numpy array.
X[:, 2:3] = imputer.transform(X[:, 2:3]) #Now, transform the empty cells with median from the remaining cells.
#Label Encoding and OneHotEncoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#We will create labels for Sex as only 2 of them exist in data and so 1 column can hold data easily with 0 and 1 values.
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
#Since Embarked attribute has 3 different categories, we need atleast 2 columns and therefore we'll use ColumnTransformer Class for this.
ct = ColumnTransformer([('Embarked', OneHotEncoder(drop = 'first'), [4])], remainder = 'passthrough') 
X = ct.fit_transform(X)
#X = X.astype(float)
#Splitting data into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
#Fitting Random Forest to the Training set
#The parameters within RandomForestClassifier were found out using Grid Search algorithm that comes later in this tutorial.
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', max_depth = 110, max_features = 3, min_samples_leaf = 5, min_samples_split = 10)
classifier.fit(X_train, y_train)
#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
kcv = cross_val_score(estimator = classifier,X = X_train, y = y_train, cv = 10)
kcv.mean()
#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'min_samples_leaf': [3, 4, 5],
               'max_depth': [90, 100, 110, 120],
               'max_features': [2, 3],
               'min_samples_split': [10, 12, 14, 16],
               'n_estimators': [300, 500, 1000]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
#Test set for kaggle
test_data = pd.read_csv('test.csv')
X_kaggle = test_data.iloc[:, [1,3,4,5,10]].values

X_kaggle[:, 2:3] = imputer.transform(X_kaggle[:, 2:3])

X_kaggle[:, 1] = le.transform(X_kaggle[:, 1])

X_kaggle = ct.transform(X_kaggle)
X_kaggle = X_kaggle.astype(float)

y_pred2 = classifier.predict(X_kaggle)

y_kaggle = np.column_stack((test_data['PassengerId'].values, y_pred2))
y_kaggle = pd.DataFrame(y_kaggle)
y_kaggle.columns = ['PassengerId', 'Survived']
y_kaggle.to_csv('titanic_kaggle_submission.csv', index = False, index_label = None)