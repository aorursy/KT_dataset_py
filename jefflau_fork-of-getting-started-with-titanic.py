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
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

columns = list(train_data.columns)



train_data.head()


from sklearn.model_selection import train_test_split



# select the column we want to predict then include the rest as the features of our model 

y = train_data.Survived



columns = list(train_data.columns)



features = [col for col in columns if col!= "Survived"]



X = train_data[features]

X_test = test_data[features]



# now we split the data into valid and training part



X_train,X_valid,y_train,y_valid = train_test_split(X,y, random_state = 0)



#Check Categorical Columns and Numerical Columns

Cat_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

Num_cols = [col for col in X_train.columns if 

                X_train[col].dtype in ['int64', 'float64']]

print(Cat_cols)

print(Num_cols)
# Remove the name category

Cat_cols.remove('Name')



# Keep selected columns

my_cols = Cat_cols + Num_cols

X_tra = X_train[my_cols].copy()

X_val = X_valid[my_cols].copy()

X_test_final = X_test[my_cols].copy()



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



# Set up the steps for numerical data

numerical_transformer = SimpleImputer()



# Set up the steps for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('OneHot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, Num_cols),

        ('cat', categorical_transformer, Cat_cols)

    ])



#Set up the model



model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)



# Put everything into the bundle



run_model = Pipeline(steps =[

    ('preprocessor', preprocessor),

    ('model',model)

])









#fit the model

run_model.fit(X_tra,y_train)



#calculate mean absolute error



pred = run_model.predict(X_val)

mae = mean_absolute_error(pred, y_valid)



print("MAE scores:\n", mae)







#output the prediction 



predictions = run_model.predict(X_test_final)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")