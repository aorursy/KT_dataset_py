# import libraries

import numpy as np

import pandas as pd

import tensorflow as tf



print ( tf.__version__)
# read the dataset

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.info()
train_df.describe()
#dropping columns

train_df = train_df.drop(["PassengerId","Name","Ticket"], axis=1)

test_df = test_df.drop(["PassengerId","Name","Ticket"], axis=1)
print ("Train")

print (train_df.isnull().sum() )

print ("-------")

print ("Test")

print (test_df.isnull().sum() )
# combine the whole dataset to get the mean values of the total dataset

# (just be careful to not leak data)

combined_df = pd.concat([train_df, test_df])



# get mean values per gender

male_mean_age = combined_df[combined_df["Sex"]=="male"]["Age"].mean()

female_mean_age = combined_df[combined_df["Sex"]=="female"]["Age"].mean()

print ("female mean age: %1.0f" %female_mean_age )

print ("male mean age: %1.0f" %male_mean_age )



# fill the nan values 

train_df.loc[ (train_df["Sex"]=="male") & (train_df["Age"].isnull()), "Age"] = male_mean_age

train_df.loc[ (train_df["Sex"]=="female") & (train_df["Age"].isnull()), "Age"] = female_mean_age



test_df.loc[ (test_df["Sex"]=="male") & (test_df["Age"].isnull()), "Age"] = male_mean_age

test_df.loc[ (test_df["Sex"]=="female") & (test_df["Age"].isnull()), "Age"] = female_mean_age
train_df["Cabin"] = train_df["Cabin"].fillna("X")

test_df["Cabin"] = test_df["Cabin"].fillna("X")
train_df["Embarked"] = train_df["Embarked"].fillna("S")

test_df["Embarked"] = test_df["Embarked"].fillna("S")
mean_fare = combined_df["Fare"].mean()

test_df["Fare"] = test_df["Fare"].fillna(mean_fare)
print ("Train")

print (train_df.isnull().sum() )

print ("-------")

print ("Test")

print (test_df.isnull().sum() )
# sampling 80% for train data

train_set = train_df.sample(frac=0.8, replace=False, random_state=777)

# the other 20% is reserverd for cross validation

cv_set = train_df.loc[ set(train_df.index) - set(train_set.index)]



print ("train set shape (%i,%i)"  %train_set.shape)

print ("cv set shape (%i,%i)"   %cv_set.shape)

print ("Check if they have common indexes. The folowing line should be an empty set:")

print (set(train_set.index) & set(cv_set.index))
# defining numeric columns

pclass_feature = tf.feature_column.numeric_column('Pclass')

parch_feature = tf.feature_column.numeric_column('Parch')

fare_feature = tf.feature_column.numeric_column('Fare')

age_feature = tf.feature_column.numeric_column('Age')



#defining buckets for children, teens, adults and elders.

age_bucket_feature = tf.feature_column.bucketized_column(age_feature,[12,21,60])



#defining a categorical column with predefined values

sex_feature = tf.feature_column.categorical_column_with_vocabulary_list(

    'Sex',['female','male']

)

#defining a categorical columns with dynamic values

embarked_feature =  tf.feature_column.categorical_column_with_hash_bucket(

    'Embarked', 3 

)

cabin_feature =  tf.feature_column.categorical_column_with_hash_bucket(

    'Cabin', 100 

)



feature_columns = [ pclass_feature,age_feature, age_bucket_feature, parch_feature, 

                   fare_feature, embarked_feature, cabin_feature ]
estimator = tf.estimator.LinearClassifier(

    feature_columns=feature_columns)
# train input function

train_input_fn = tf.estimator.inputs.pandas_input_fn(

      x=train_set.drop('Survived', axis=1),

      y=train_set.Survived,

      num_epochs=None, #For training it can use how many epochs is necessary

      shuffle=True,

      target_column='target',

)



cv_input_fn = tf.estimator.inputs.pandas_input_fn(

      x=cv_set.drop('Survived', axis=1),

      y=cv_set.Survived,

      num_epochs=1, #We just want to use one epoch since this is only to score.

      shuffle=False  #It isn't necessary to shuffle the cross validation 

)
estimator.train(input_fn=train_input_fn, steps=400)
scores = estimator.evaluate(input_fn=cv_input_fn)

print("\nTest Accuracy: {0:f}\n".format(scores['accuracy']))
# DNN doesn't support categorical with hash bucket

embarked_embedding =  tf.feature_column.embedding_column(

    categorical_column = embarked_feature,

    dimension = 3,

)

cabin_embedding =  tf.feature_column.embedding_column(

    categorical_column = cabin_feature,

    dimension = 300,

)



# define the feature columns

feature_columns = [ pclass_feature,age_feature, age_bucket_feature, parch_feature, 

                   fare_feature, embarked_embedding, cabin_embedding ]



# instantiate the estimator

NNestimator = tf.estimator.DNNClassifier(

    feature_columns=feature_columns,

    hidden_units=[10, 30 , 10])



# call the train function using the train input function

NNestimator.train(input_fn=train_input_fn, steps=1000)
# evaluate and print the accuracy using the cross-validation input function

accuracy_score = NNestimator.evaluate(input_fn=cv_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
def prepare_datasets(df):

    df_copy = df[['Pclass', 'Parch',  'Sex', 'Embarked', "Age"]].copy()

    df_copy.loc[:,"Sex"] = df_copy.Sex.apply(lambda x: 0 if x =="male" else 1)



    e_map = {"C": 0,"Q":1, "S":2}

    df_copy.loc[:,"Embarked"] = df_copy.Embarked.apply(lambda x: e_map[x])



    df_copy.loc[:,"Age"]= df_copy.Age.astype(np.float32)



    x = df_copy[['Pclass', 'Parch', 'Age']].astype(np.float32)

#     y = train_set.Survived.astype(np.int32)

    y = df.Survived.astype(np.bool)

    return x, y



x_train, y_train = prepare_datasets(train_set)

x_cv, y_cv = prepare_datasets(cv_set)

def generate_tf_input_fn(x_input,y_input,num_epochs=None):

    #this is the function we are generating

    def _input_fn_():

        # generate a standard input function

        train_input_fn = tf.estimator.inputs.pandas_input_fn(

            x= x_input,  

            y= y_input,

            num_epochs=num_epochs,

            shuffle=True,

            target_column='target',

        )

        #execute the standard input function 

        x, y = train_input_fn()

        # expand the shape of the results (necessary for Tensor Forest)

        for name in x:

            x[name] = tf.expand_dims(x[name], 1, name= name) 

        return x, y

    

    return _input_fn_
# generate custom train input function

forest_train_input_fn = generate_tf_input_fn(x_train,y_train,num_epochs=None)



# instantiate the estimator

params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(

    num_classes=2, num_features=4, regression=False,

    num_trees=50, max_nodes=1000).fill()

classifier2 = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)

# train the estimator

classifier2.fit(input_fn=forest_train_input_fn)
# evaluate and print the accuracy using the cross-validation input function

forest_cv_input_fn = generate_tf_input_fn(x_cv, y_cv, num_epochs=1)

accuracy_score = classifier2.evaluate(input_fn=forest_cv_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))