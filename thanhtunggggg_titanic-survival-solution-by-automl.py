# pandas and numpy for data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

# Import the tpot Classifier
from tpot import TPOTClassifier
print("TPOT has been installed")
# Read data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#Show 10 fist datapoints
train_df.head(n=10)
#Dataset information
train_df.info()
print('_'*40)
test_df.info()
#numerical features stats
train_df.describe()
#Categorical features stats 
train_df.describe(include=['O'])
#Labels: Cái cần dự đoán
y_train = train_df["Survived"]
#Drop feature: ID, name, ticket number là không cần thiết. Cabin miss value quá nhiều
X_train = train_df.drop(["Survived","PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missed categorical value
freq_port = train_df.Embarked.dropna().mode()[0]
X_train['Embarked'] = X_train['Embarked'].fillna(freq_port)

# Convert categorical feature to numerical feature
X_train['Sex'] = X_train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
X_train['Embarked'] = X_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Show traning data feature 
X_train.head(n=10)
# Convert to numpy arrays
X_train = np.array(X_train)

# Sklearn wants the labels as one-dimensional vectors
y_train = np.array(y_train).reshape((-1,))
print(X_train.shape)
print(y_train.shape)
#Split data: traing and testing
#X_train , X_test , y_train , y_test = train_test_split( X_train , y_train , train_size = .8 )
print(X_train.shape)
print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)
# Create a tpot object with a few parameters
pipeline_optimizer = TPOTClassifier(generations=15, population_size=50, cv=6,
                                    random_state=42, verbosity=2)
print("Done")
# Fit the tpot model on the training data
pipeline_optimizer.fit(X_train, y_train)
print("Training finish")
# Show the final model
print(pipeline_optimizer.fitted_pipeline_)
# Export the pipeline as a python script file
pipeline_optimizer.export('tpot_exported_pipeline.py')
# To examine all fitted models
#pipeline_optimizer.evaluated_individuals_
# Evaluate the final model
#print(pipeline_optimizer.score(X_test, y_test))
test_df = pd.read_csv('../input/test.csv')
X_test_df = test_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

#freq_port = test_df.Embarked.dropna().mode()[0]
X_test_df['Embarked'] = X_test_df['Embarked'].fillna(freq_port)

# Convert categorical feature to numerical feature
X_test_df['Sex'] = X_test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
X_test_df['Embarked'] = X_test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Show traning data feature 
X_test_df.head(n=10)
X_test_df = np.array(X_test_df)
print(X_test_df.shape)
surv_pred = pipeline_optimizer.predict(X_test_df)

submit = pd.DataFrame({'PassengerId' : test_df.loc[:,'PassengerId'],
                       'Survived': surv_pred.T})
submit.to_csv("../working/submit.csv", index=False)
#submit.to_csv("submit.csv", index=False)
submit.head(n=20)