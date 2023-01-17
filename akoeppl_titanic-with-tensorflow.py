#Importing initial libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

%matplotlib inline



import tensorflow as tf

print('-'*25)

print(tf.__version__)

sess=tf.InteractiveSession()



#ignore warnings

import warnings

warnings.filterwarnings('ignore')

#Since we will be running a lot of different models, it's important to rest  the tensorflow 

#graph between runs so that the various parts aren't erroneously linked together.

def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)



reset_graph()



# Load data

##### Load train and Test set

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

#Saving IDtest for the final result submission

IDtest = df_test["PassengerId"]

df_train.head()
print(df_train.shape)

print(df_test.shape)
## Join train and test datasets in order to obtain the same number of features during categorical conversion

train_len = len(df_train)

dataset =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)
dataset.head()
#Checking missing values

print(dataset.isnull().sum())
Fare_median = df_train.Fare.median()

dataset.loc[dataset.Fare.isnull(),'Fare'] = Fare_median
g = sns.factorplot("Embarked",  data=dataset,size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
dataset.loc[dataset.Embarked.isnull(),'Embarked'] = 'S'
print("% of NaN values in Cabin Variable for dataset:")

print(dataset.Cabin.isnull().sum()/len(dataset))
# Replace the Cabin number by the type of cabin 'X' if not

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])





g = sns.factorplot("Cabin",  data=dataset,size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = df_train["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & 

                               (dataset['Parch'] == dataset.iloc[i]["Parch"]) &

                               (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med



#Checking missing values

print(dataset.isnull().sum())
title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(title)





j = sns.countplot(x="Title",data=dataset)

j = plt.setp(j.get_xticklabels(), rotation=45) 
# Exploring Survival probability

g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar", size = 10 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Let's group the military and crew titles ensemble

dataset["Title"] = dataset["Title"].replace(['Capt', 'Col','Major'], 'Crew/military')

# Let's group the profession title ensemble

dataset["Title"] = dataset["Title"].replace(['Master', 'Dr'], 'Prof')

# Let's change the type of women title

dataset["Title"] = dataset["Title"].replace(['Miss', 'Mme','Mrs','Ms','Mlle'], 'Woman')

# Let's change the type of nobility title

dataset["Title"] = dataset["Title"].replace(['the Countess', 'Sir', 'Lady', 'Don','Jonkheer', 'Dona'], 'Noble')

# Let's change the type of religious title

dataset["Title"] = dataset["Title"].replace(['Rev'], 'Religious')



j = sns.countplot(x="Title",data=dataset)

j = plt.setp(j.get_xticklabels(), rotation=45) 
# Exploring Survival probability

g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset.Ticket.head(10)
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket

dataset["Ticket"].head(10)
# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

g = sns.factorplot(x="Fsize",y="Survived",data = dataset)

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)



g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")
#One-hot encoder for categorical variables

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")

dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix="Cab")

dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T_")

# convert Sex into categorical value 0 for male and 1 for female

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# Create categorical values for Pclass

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")



# Drop useless variables 

dataset.drop(labels = ["PassengerId","Name"], axis = 1, inplace = True)



dataset.head()
print(dataset.info())

#separation of the train/test

df_train = dataset[:train_len]

df_test = dataset[train_len:]

df_test.drop(labels=["Survived"],axis = 1,inplace=True)

df_test.columns

#X vs Y split

df_train_X = df_train.drop(labels =['Survived'], axis =1)

df_train_Y = df_train['Survived']
#Everything is numerical, so it's easier to create a list of variables

variable_names = df_train_X.columns



tf_variables =[]



for index in variable_names:

   tf_variables.append(tf.feature_column.numeric_column(index)) 



tf_variables
from sklearn.preprocessing import MinMaxScaler

Scaler = MinMaxScaler()

scaled_X_train = Scaler.fit_transform(df_train_X)

scaled_X_test = Scaler.transform(df_test)
#Converting numerical variables into categorical variables

scaled_X_train  = pd.DataFrame(scaled_X_train,columns=df_train_X.columns)

scaled_X_train.head()


scaled_X_train.Age.hist(bins=20)
Age_bucket = tf.feature_column.bucketized_column(tf_variables[0],boundaries=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,

                                                                             0.50,0.55,0.60,0.65,0.70,0.80,1.02])



scaled_X_train.Fare.hist(bins=40)


Fare_bucket = tf.feature_column.bucketized_column(tf_variables[1],boundaries=[0.05,0.1,0.15,0.2,0.4,0.6,1.02])


tf_variables.append(Age_bucket)

tf_variables.append(Fare_bucket)
from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(scaled_X_train,df_train_Y, test_size =0.3, random_state =101)
print(X_train.shape)

print(scaled_X_train.shape)
input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y= Y_train,

                                                     batch_size =10, num_epochs =1000, shuffle =True )
linear_model= tf.estimator.LinearClassifier(feature_columns=tf_variables, n_classes=2)
linear_model.train(input_fn=input_function,steps=1000)
#evaluate model

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=Y_test, batch_size=10, num_epochs=1, shuffle=False)
results =linear_model.evaluate(eval_input_func)
results
#Resetting tensorflow graphs.

tf.Session.reset

sess = tf.InteractiveSession()



#redefining the tensors

tf_variables =[]



for index in variable_names:

   tf_variables.append(tf.feature_column.numeric_column(index)) 



Age_bucket = tf.feature_column.bucketized_column(tf_variables[0],boundaries=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,

                                                                             0.50,0.55,0.60,0.65,0.70,0.80,1.02])



Fare_bucket = tf.feature_column.bucketized_column(tf_variables[1],boundaries=[0.05,0.1,0.15,0.2,0.4,0.6,1.02])

tf_variables.append(Age_bucket)

tf_variables.append(Fare_bucket)
dnn_input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y= Y_train,batch_size =10,

                                                         num_epochs =1000, shuffle =True )
dnn_model=tf.estimator.DNNClassifier(hidden_units=[10,10,10,10],feature_columns=tf_variables, 

                                     n_classes =2, optimizer=tf.train.AdamOptimizer(learning_rate=0.001),

                                    dropout=0.2)
dnn_model.train(input_fn = dnn_input_function, steps =1000)
#evaluate model

dnn_eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=Y_test, 

                                                          batch_size=10, num_epochs=1, shuffle=False)

dnn_results =dnn_model.evaluate(eval_input_func)
dnn_results
scaled_X_test = pd.DataFrame(scaled_X_test,columns=df_train_X.columns)

scaled_X_test.head()
#making predictions

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=scaled_X_test,batch_size=10,num_epochs=1, shuffle=False )
preds =list(dnn_model.predict(pred_input_func))

preds
predictions =[p['class_ids'][0] for p in preds]
predictions_df =pd.DataFrame(predictions,columns=['Survived'])

predictions_df.head()
df_test['Survived']=predictions



results =  pd.concat([IDtest,predictions_df], axis=1)



results.head()

results.to_csv("output_python_tf.csv",index=False)
