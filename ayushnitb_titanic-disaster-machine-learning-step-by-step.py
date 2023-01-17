# Loading Numpy and Pandas Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



import pandas as pd
## Creating panda dataframes from train and test CSV files
print("Loading Training and Testing Data =====>")
training_data = pd.read_csv('../input/train.csv')
testing_data = pd.read_csv('../input/test.csv')
print("<===== Training and Testing Data Loading finished")
'''
    Printing the 5 first samples in training_data dataframe 
'''
training_data.head(5)
'''
    Printing the 6 samples select randomly in training_data dataframe 
'''
training_data.sample(6)
training_data.columns
training_data.dtypes

%matplotlib inline
'''
    Creating dataframes separating survived and not survived passergers
'''
td_not_survived=training_data.loc[(training_data['Survived']==0)]
td_survived=training_data.loc[(training_data['Survived']==1)]
td_not_survived.head(5)
td_survived.sample(10)

df = training_data.groupby(['Sex','Survived']).size()
df=df.unstack()
df.head()

plt.figure();df.plot(kind='bar').set_title('Gender histogram training data')
df = td_survived.groupby('Sex').size()
#df=df.unstack()
df.head()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by gender');
df = td_not_survived.groupby('Sex').size()
plt.figure();df.plot(kind='bar').set_title(' Not Survived passengers by gender');
df = td_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by Pclass');
df = td_not_survived.groupby('Pclass').size()
plt.figure();df.plot(kind='bar').set_title('Not Survived passengers by Pclass');
plt.figure();
td_survived.Age.hist()
#Not taking cabin column as so much of missing data hence imputing based on mode can cause errors
#Titanic dataset from kaggle
dataset = training_data
dataset= dataset[['PassengerId','Name','Ticket','Cabin','Embarked','Sex','Pclass','Age','SibSp','Parch','Fare','Survived']]
X = dataset.iloc[:, 4:11].values #Name, PassengerId, Ticket and Cabin doesnot give any idea whether the passenger will survive or not
y = dataset.iloc[:, 11].values
#Imputing missing categorical and numeric variables respectively:
#Using mode for categorical variable along the column and mean for numeric variable along the column.
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, A, b=None):

        self.fill = pd.Series([A[c].value_counts().index[0]
            if A[c].dtype == np.dtype('O') else A[c].mean() for c in A],
            index=A.columns)

        return self

    def transform(self, A, b=None):
        return A.fillna(self.fill)

X = DataFrameImputer().fit_transform(pd.DataFrame(X))
#Above code is for imputing dataframe. Mode for categorical and mediun for numeric

X= X.values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#Converting into dataframe
y_pred_df = pd.DataFrame(y_pred, columns= ['Survived'])
#Column name in the square bracket
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# Gives 84% accuracy from confusion matrix

#Applying K-fold cross validation for further check for model performance
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X= X_train, y= y_train, cv =10)
accuracies.mean()
accuracies.std()  # 83% accuracy from this method
accuracies.mean()
# ===== ====== ======= ====== ======= ====== ====== ====== ====== ======
#Now predicting the survival of test_data(Actual)
dataset_test = testing_data

#Rearranging the columns for better understanding of dataset
dataset_test= dataset_test[['PassengerId','Name','Ticket','Cabin','Embarked','Sex','Pclass','Age','SibSp','Parch','Fare']]
X2 = dataset_test.iloc[:, 4:11].values

#Imputer categorical variables
X2 = DataFrameImputer().fit_transform(pd.DataFrame(X2))
#Above code is for imputing dataframe. Mode for categorical and mediun for numeric
X2= X2.values #Converting back to array format

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X2_1 = LabelEncoder()
X2[:, 1] = labelencoder_X_1.fit_transform(X2[:, 1])
labelencoder_X2_2 = LabelEncoder()
X2[:, 0] = labelencoder_X2_2.fit_transform(X2[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X2 = onehotencoder.fit_transform(X2).toarray()
X2 = X2[:, 1:]

#Prediction od test_data
y_pred_test = classifier.predict(X2)

#Converting into dataframe
d= {'PassengerId' : dataset_test.iloc[:, 0].values, 'Survived' : y_pred_test}
y_pred_test_df = pd.DataFrame(d)
#Column name in the square bracket



y_pred_test_df