import numpy as np
import pandas as pd
df = pd.read_csv("../input/machine-learning-decision-tree-model/salaries.csv")

df.head(20)
# In this type of situation where we have more than 1 parametre wchich can effect the outcome ,we can apply decsion tree algorithm,our algorithm is classified into two values yes or No it is case of classification. 
# The first step is to drop the coloumn wchich has the output save it as varaible

inputs =df.drop('salary_more_then_100k',axis='columns')
target =df['salary_more_then_100k']
#labelEncoder will asing a numeric values to all valuves in columns
from sklearn.preprocessing import LabelEncoder 
le_company =LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n']= le_degree.fit_transform(inputs['degree'])
# here fit.transform method is incodeding the value of caompany name,job and degree and saving them as nw coloumn as incoded value.
inputs
# The next step after incodeing is to drop the original coloumn
inputs_n = inputs.drop(['company','job','degree'],axis = 'columns')
target
from sklearn import tree
model = tree.DecisionTreeClassifier(criterion = 'entropy')
model.fit(inputs_n,target)
model.score(inputs_n,target)
model.predict([[2,2,1]])
op = model.predict([[2,1,1]])
print(op)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
import numpy as np
import pandas as pd
import seaborn as sns
df = pd.DataFrame(
    np.random.rand(100, 5),
    columns=['a', 'b', 'c', 'd', 'e'])
df['a'][1] = 'NaN'
df['b'][5] = 'NaN'
df['a'][15] = 'NaN'
sns.heatmap(df.isnull(), cbar=False)