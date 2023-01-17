from __future__ import absolute_import, division, print_function, unicode_literals



import tensorflow as tf

from tensorflow.keras import regularizers

from tensorflow import keras

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt





from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split





df3 = pd.read_csv("../input/titanic/gender_submission.csv")

df1 = pd.read_csv("../input/titanic/test.csv")

df = pd.read_csv("../input/titanic/train.csv")

df.head()





df.isnull().sum()





df1.isnull().sum()


df.shape

df.pop('Cabin')

df.pop('PassengerId')



df.pop('Ticket')

df.pop('Fare')



df.head()



df1.shape

df1.pop('Cabin')

df1.pop('PassengerId')



df1.pop('Ticket')

df1.pop('Fare')



df1.head()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False)




df.info()


df['Age']=df['Age'].fillna(df['Age'].mean())
df1['Age']=df1['Age'].fillna(df1['Age'].mean())
df['Embarked']=df['Embarked'].fillna('S')
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
sns.heatmap(df1.isnull(),yticklabels=False,cbar=False)


df.info()
df['Family'] = df.apply(lambda row: row.Parch + (row.SibSp), axis = 1) 

df1['Family'] = df1.apply(lambda row: row.Parch + (row.SibSp), axis = 1) 

df1.info()


train, test = train_test_split(df, test_size=0.2)

train, val = train_test_split(df, test_size=0.2)

print(len(train), 'train examples')

print(len(val), 'validation examples')

print(len(test), 'test examples')
df1['Survived'] = df1.apply(lambda _: '', axis=1)
# A utility method to create a tf.data dataset from a Pandas Dataframe



def df_to_dataset(dataframe, shuffle=True, batch_size=32):

  dataframe = dataframe.copy()

  labels = dataframe.pop('Survived')

  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(dataframe))

  ds = ds.batch(batch_size)

  return ds
test=df_to_dataset(df1,shuffle=False,batch_size=32)



listOfDFRows = df_to_dataset(df1,shuffle=False)









for feature_batch, label_batch in listOfDFRows.take(1):

  print('Every feature:', list(feature_batch.keys()))

  print('A batch of ages:', feature_batch['Age'])

  print('A batch of targets:', label_batch )


batch_size = 32# A small batch sized is used for demonstration purposes

train_ds = df_to_dataset(df, batch_size=batch_size)
# We will use this batch to demonstrate several types of feature columns

batch_size = 32 # A small batch sized is used for demonstration purposes

train_ds = df_to_dataset(train, batch_size=batch_size)

val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)

test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)















for feature_batch, label_batch in train_ds.take(1):

  print('Every feature:', list(feature_batch.keys()))

  print('A batch of ages:', feature_batch['Age'])

  print('A batch of targets:', label_batch )








# We will use this batch to demonstrate several types of feature columns

example_batch = next(iter(train_ds))[0]

# A utility method to create a feature column

# and to transform a batch of data

def demo(feature_column):

  feature_layer = layers.DenseFeatures(feature_column)

  print(feature_layer(example_batch).numpy())
age = feature_column.numeric_column("Age")

demo(age)


age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

demo(age_buckets)


thal= feature_column.categorical_column_with_vocabulary_list(

      'Sex', ['male', 'female'])



thal_one_hot = feature_column.indicator_column(thal)

demo(thal_one_hot)

# Notice the input to the embedding column is the categorical column

# we previously created

title= feature_column.categorical_column_with_vocabulary_list(

      'Name', ['Mr', 'Miss', 'Mrs', 'Master', 'Rare'])

title_embedding = feature_column.embedding_column(title, dimension=8)

demo(title_embedding)
pclass = feature_column.numeric_column("Pclass")

demo(pclass)
sib = feature_column.numeric_column("SibSp")

demo(sib)
par = feature_column.numeric_column("Parch")

demo(par)


fam= feature_column.numeric_column("Family")

demo(fam)


emb= feature_column.categorical_column_with_vocabulary_list(

      'Embarked', ['C', 'Q','S'])



emb_one_hot = feature_column.indicator_column(emb)

demo(emb_one_hot)
feature_columns = []



# numeric cols

for header in ['Age', 'Family','Pclass']:

  feature_columns.append(feature_column.numeric_column(header))



# bucketized cols



feature_columns.append(thal_one_hot)

feature_columns.append(age_buckets)

feature_columns.append(title_embedding)

feature_columns.append(emb_one_hot)





feature_layer = tf.keras.layers.DenseFeatures(feature_columns)



STEPS_PER_EPOCH = 712//32

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

  0.001,

  decay_steps=STEPS_PER_EPOCH*1000,

  decay_rate=1,

  staircase=False)



def get_optimizer():

  return tf.keras.optimizers.Adam(lr_schedule)
step = np.linspace(0,100000)

lr = lr_schedule(step)

plt.figure(figsize = (8,6))

plt.plot(step/STEPS_PER_EPOCH, lr)

plt.ylim([0,max(plt.ylim())])

plt.xlabel('Epoch')

_ = plt.ylabel('Learning Rate')

model = tf.keras.Sequential([

  feature_layer,

  layers.Dense(64, activation='relu'),

  layers.Dense(64,activation='relu'),

  layers.Dense(1,activation='tanh')

])

optimizer = tf.optimizers.Adam(0.00005)



model.compile(optimizer,

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])



model.fit(train_ds,

          validation_data=val_ds,

          epochs=245)


loss, accuracy = model.evaluate(test_ds)

print("Accuracy", accuracy)
p=model.predict(listOfDFRows)


g=[]

for n in p:

  if n>0.5:

     g.append([1])



  else:

    g.append([0])

 

 



print("p", g)





pred=pd.DataFrame(g)
df3.pop('Survived')
df3.info()
df3=pd.concat([df3['PassengerId'],pred],axis=1)
df3.columns=['PassengerId','Survived']

df3.head()
df3.to_csv('gender_submissionf5.csv',index=False)