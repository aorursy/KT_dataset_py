# data processing
import numpy as np
import pandas as pd 

# machine learning
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# utils
import time
from datetime import timedelta

# some configuratin flags and variables
verbose=0 # Use in classifier

# Input files
file_train='../input/train.csv'
file_test='../input/test.csv'

# defeine random seed for reproducibility
seed = 69
np.random.seed(seed)

# read training data
train_df = pd.read_csv(file_train,index_col='PassengerId')


# Show the columns
train_df.columns.values
# Show the shape
train_df.shape
# preview the training dara
train_df.head()
# Show that there is NaN data (Age,Fare Embarked), that needs to be handled during data cleansing
train_df.isnull().sum()
nullAge = train_df[train_df.Age.isnull()].index.values
nullAge
train_df.loc[train_df.index.values == 889]
def prep_data(df):
    # Drop unwanted features
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing data: Age and Fare with the mean, Embarked with most frequent value
    df[['Age']] = df[['Age']].fillna(value=df[['Age']].mean())
    #df[['Fare']] = df[['Fare']].fillna(value=df[['Fare']].mean())
    df[['Embarked']] = df[['Embarked']].fillna(value=df['Embarked'].value_counts().idxmax())
    
    # Convert categorical  features into numeric
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
      
    # Convert Embarked to one-hot
    embarked_one_hot = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = df.drop('Embarked', axis=1)
    df = df.join(embarked_one_hot)
    
    #Map age in group under 10 and over 10yo
    df.loc[df.Age <11,'Age']=0
    df.loc[df.Age >=10,'Age']=1
    
    return df
#Prepare training data and show that there isn't any null data
train_df = prep_data(train_df)
#train_df.isnull().sum()
#train_df.Age.value_counts()
# X contains all columns except 'Survived'  
X = train_df.drop(['Survived'], axis=1).values.astype(float)

# Scaling

scale = StandardScaler()
X = scale.fit_transform(X)

# Y is just the 'Survived' column
Y = train_df['Survived'].values
def create_model(optimizer='rmsprop', init='uniform'):
    # create model
    #if verbose: 
    print("**Create model with optimizer: %s; init: %s" % (optimizer, init) )
    model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(16, kernel_initializer=init, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(4, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
model = create_model()
model.fit(X,Y,epochs=59, batch_size=32)
# Read test data
test_df = pd.read_csv(file_test,index_col='PassengerId')
# Prep and clean data
test_df = prep_data(test_df)
# Create X_test
X_test = test_df.values.astype(float)
# Scaling
X_test = scale.transform(X_test)
# Predict 'Survived'
prediction = model.predict_classes(X_test, verbose=1)
prediction_step = [int(round(x[0])) for x in prediction]

# Save and submit
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Survived': prediction_step,
})

submission.sort_values('PassengerId', inplace=True)    
submission.to_csv('submission-adam_simple.csv', index=False)
