# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
"""women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)

print(sum(women))

print(len(women))

print("% of women who survived:", rate_women)"""
""""#datavisualisation

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



#sns.set_style=('dark')

#figure size

plt.figure(figsize=(20,10))

#add Title

#plt.title('data visualisation wrt to Fare sex and age')

#plot regplot  

#sns.barplot(x=train_data['Sex'],y=train_data['Survived'])

#sns.scatterplot(x=train_data['Age'],y=train_data['Survived'],hue=train_data['Sex'])

#sns.barplot(x=train_data['Pclass'],y=train_data['Survived'])

#sns.scatterplot(x=train_data['Pclass'],y=train_data['Fare'],hue=train_data['Survived'])

#sns.kdeplot(shade=True,data=train_data['Age'])

#sns.regplot(x=train_data['Age'],y=train_data['Survived'])

#sns.swarmplot(x=train_data['Survived'],y=train_data['Age'])

#sns.barplot(x=train_data['Age'], y= train_data['Survived'])

#sns.distplot(a=train_data['Age'], kde=False,label='Age')

#sns.distplot(a=train_data['Survived'],kde=False,label='Age')

#plt.legend()

sns.jointplot(x=train_data['Age'],y=train_data['Survived'],kind='kde')"""

"""#data preprocessing,drop entries strategy

cols_missing=[col for col in train_data.columns

               if train_data[col].isnull().any()]

print('columns having missing entries {}'.format(cols_missing))



#shape of data

print('shape of the data {}'.format(train_data.shape))



#Remove cateogorical variables

X = train_data.select_dtypes(exclude=['object'])

X_test = test_data.select_dtypes(exclude=['object'])

print('shape of the data after removal of cateogrical columns shape of X:{},shape of X_test:{}'.format(X.shape,X_test.shape))



#for finding no of entries of missing column.

missing_val_count_by_column = (X.isnull().sum())

print('no of missing entries in each column {}'.format(missing_val_count_by_column[missing_val_count_by_column > 0]))



#droping missing entries.

#reduced_X =X.drop('Cabin', axis=1)      #******here one more parameter inplace is also used inplace=True means make changes in the existing data by default it is false which means make changes in another assigned dataframe if we use inplace=True here then it will give NoneType error because changes are made in existing data 

#print(reduced_X.shape)                         #we removes cabin because 687/891*100 ie 77% data is missing 



print(X.head())

"""
"""#data preprocessing with,simple imputer strategy

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split 

y=train_data['Survived']

X_train,X_valid,y_train,y_valid = train_test_split(X,y, train_size=0.8, test_size=0.2 ,random_state=0)



X_train_processed = pd.get_dummies(X_train)

print(X_train_processed.head())

X_valid_processed = pd.get_dummies(X_valid)

print(X_valid_processed.head())

my_imputer = SimpleImputer(strategy='mean')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train_processed,y_train))

imputed_X_valid =  pd.DataFrame(my_imputer.transform(X_valid_processed))



imputed_X_train.columns=X_train.columns

imputed_X_valid.columns=X_valid.columns

"""









"""#Get list of cateogrical cateogrical cols

from sklearn.model_selection import train_test_split

y=train_data['Survived']

X_train1,X_valid1,y_train1,y_valid1 = train_test_split(train_data,y,train_size=0.8,test_size=0.2,random_state=0)



cateogrical_col = [cols for cols in X_train1.columns if X_train1[cols].dtype=='object']

print(cateogrical_col)

#get list of numerical cols

numerical_cols  = [cols for cols in X_train1.columns if X_train1[cols].dtype in ['int64','float64'] ] 

print(numerical_cols)

print(y)

"""
""""from sklearn.preprocessing import LabelEncoder

label_X_train = X_train1.copy()

label_X_valid = X_valid1.copy()

print(label_X_train.unique())

print(label_X_valid.unique())

label_encoder = LabelEncoder()

for col in cateogrical_col:

    encoder_X_train = label_encoder.fit_transform(label_X_train[col])

    encoder_X_valid = label_encoder.transform(label_X_valid[col])

"""    

    



"""from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

def score_dataset(X_train_final,X_valid_final,y_train,y_valid):

    model =  RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

    model.fit(X_train_final,y_train)

    preds= model.predict(X_valid_final)

    mae =  mean_absolute_error(preds,y_valid)

    return mae

"""    
"""print(score_dataset(imputed_X_train,imputed_X_valid,y_train,y_valid))"""
"""from sklearn.ensemble import RandomForestRegressor



# Define the models

model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)



models = [model_1, model_2, model_3, model_4, model_5]"""
""""from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]

features = ["Pclass", "Sex","SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0,train_size=0.8, test_size=0.2)

model_1 = RandomForestClassifier(n_estimators=50, random_state=0)

model_2 = RandomForestClassifier(n_estimators=100, random_state=0)

model_3 = RandomForestClassifier(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestClassifier(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)

model_6 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

#model = [model_1, model_2, model_3, model_4, model_5]

model_6.fit(X_train,y_train)

predicts = model_6.predict(X_valid)

mae = mean_absolute_error(predicts,y_valid)

print(mae)

print(predicts)

#final_model = model_3

#final_model.fit(X,y)

#preds = final_model_3.predict(X_test)











# Function for comparing different models

def score_model(model_6, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



#for i in range(0, len(models)):

#    mae = score_model(models[i])

#    print("Model %d MAE: %d" % (i+1, mae))"""
"""from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)

print(mean_absolute_error())



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

"""
# Read the data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()



import pandas as pd

from sklearn.model_selection import train_test_split



# Remove rows with missing target, separate target from predictors

train_data.dropna(axis=0 , subset=['Survived'], inplace=True)

y = train_data.Survived

train_data.drop(['Survived'], axis=1, inplace=True)



train_data.set_index('PassengerId')



#print(train_data.head())



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y, 

                                                                train_size=0.8, test_size=0.2,random_state=0)



X_train_full.set_index('PassengerId')

X_train_full.drop(['PassengerId'],axis=1,inplace=True)

print(X_train_full.head())



#_____________________________________

# Get number of unique entries in each column with categorical data

#for object cols

object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == 'object']



object_nunique = list(map(lambda col: X_train_full[col].nunique(), object_cols))

d = dict(zip(object_cols, object_nunique))#May be the difference between unique and nunique is that unique tells you unique entries name only but nunique tells you name with value_count too.



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])



#____________________________________________









# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]





print(categorical_cols)



# Select numerical columns



numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



print(numerical_cols)

# Keep selected columns only

my_cols = categorical_cols + numerical_cols       #we can simply add two list with addition operator...

X_train = X_train_full[my_cols].copy()             

X_valid = X_valid_full[my_cols].copy()

X_test = test_data[my_cols].copy()

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier

# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

#model = RandomForestClassifier(n_estimators=100, random_state=0)

model =  XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=0 )





# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

#clf.fit(X_train, y_train)

clf.fit(X_train,y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)

print(preds)

print('MAE:', mean_absolute_error(y_valid, preds))

print(test_data.head())
print(X_test.head())



#add passengerid as index of X_test

#extracting passangerid col from train_data

col = test_data.iloc[:,0]

#adding it to test data

X_test.insert(0,'PassengerId',col)

print(X_test.head())

X_test.set_index('PassengerId')

print(X_test.head())



#Prediction on X_test

predictions = clf.predict(X_test)

#Now submitting

output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

"""# Read the data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

"""
"""

#test_data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

"""
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier





# Remove rows with missing target, separate target from predictors

train_data.dropna(axis=0 , subset=['Survived'], inplace=True)

y = train_data.Survived

train_data.drop(['Survived'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)







# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test_data[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)   #https://stackoverflow.com/questions/51645195/pandas-align-function-illustrative-example

X_train, X_test = X_train.align(X_test, join='left', axis=1)     #here join=left meaning left df i.e. X_train will be retained and since axis =1 so column labels of X_train will be retained and X_valid will acquire values acco to this.

"""
"""# Define the model

my_model_2 = XGBClassifier(random_state=0, n_estimators=1000,learning_rate=0.05,n_jobs=4) # Your code here



# Fit the model

my_model_2.fit(X_train,y_train,verbose=False,early_stopping_rounds=5, eval_set=[(X_valid,y_valid)])# Your code here



# Get predictions

predictions_2 = my_model_2.predict(X_valid) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2,y_valid) # Your code here

print(predictions_2)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)

"""
"""import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier



# Read the data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()



#test_data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()



#define featues

features = ["Pclass","SibSp","Sex", "Parch"]

X_train = train_data[features].copy()

test_data = test_data[features].copy()



#drop missing values in target variable and separate the target variable from training data

train_data.dropna(subset=['Survived'],axis=0,inplace=True)

y= train_data.Survived

#X_train.drop('PassengerId',axis=1,inplace=True)



#Break the dataset

train_X,valid_X,train_y,valid_y = train_test_split(X_train,y, train_size=0.8, test_size=0.2, random_state=0)



#Now define the model

model = XGBClassifier(random_state=0,n_estimators=1000,learning_rate=0.05)

#fit the model

model.fit(train_X,train_y,eval_set=[(valid_X,valid_y)],early_stopping_rounds=5,verbose=False)

#predicts the model

predicts=model.predict(valid_X) 

#calculate mae

mae = mean_absolute_error(predicts,valid_y)

print(mae)

"""



"""import pandas as pd

import numpy as np

"""
"""#Read the training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv",index_col='PassengerId')

train_data.head()

"""
"""

#test_data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv",index_col = 'PassengerId')

test_data.head()

"""