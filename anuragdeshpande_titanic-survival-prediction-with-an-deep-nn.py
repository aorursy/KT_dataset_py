import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # library for building machine learning models



from tensorflow.keras.models import Sequential, Model # keras API from tensorflow to easily construct an ML models.

from tensorflow.keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping # early stopping functionality to prevent overfitting.

from sklearn.model_selection import train_test_split # use sci-kit_learn ML library for data processing functions.
# Import training and prediction data

training_set = pd.read_csv("/kaggle/input/titanic/train.csv")

prediction_set = pd.read_csv("/kaggle/input/titanic/test.csv")



# Display the first items in the training set to check data has been loaded properly.

training_set.head()
# Carry out pre-processing on data to prepare it for use by the learning model.



# Start off by extracting the labels that the model will be trained on, in this case the 'Survived' column.

train_target_set = training_set['Survived']

# Then, create a data frame containing just the parameters that the model will be trained on.

# This means that the 'Survived' column is removed, since it is the target.

# The 'Name' and 'PassengerID' columns are also removed, as they are not sensible inputs by themselves.

#cleaned_input_train_set = training_set.drop(columns=['Name', 'Survived', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'])

cleaned_input_train_set = training_set.drop(columns=['Name', 'Survived', 'PassengerId', 'Ticket'])





# Now, need to convert input data into a consistent, readable format.

# First, we replace all NaNs with 0s.

cleaned_input_train_set = cleaned_input_train_set.fillna(0.0)



#The field 'Sex' contains only two different values, so can convert 'male' to 1.0 and 'female' to 2.0. The value 0.0 is reserved for NaNs.

sex_vals_new = []

for i in range(len(cleaned_input_train_set['Sex'].values)):

    if cleaned_input_train_set['Sex'].values[i] == 'male':

        sex_vals_new.append(1.0)

    else:

        sex_vals_new.append(2.0)

cleaned_input_train_set['Sex'] = np.array(sex_vals_new)



# Similarly 'Embarked' contains only three different values. The value 0.0 is reserved for NaNs.

embarked_vals_new = []

for i in range(len(cleaned_input_train_set['Embarked'].values)):

    if cleaned_input_train_set['Embarked'].values[i] == 'C':

        embarked_vals_new.append(1.0)

    elif cleaned_input_train_set['Embarked'].values[i] == 'S':

        embarked_vals_new.append(2.0)

    elif cleaned_input_train_set['Embarked'].values[i] == 'Q':

        embarked_vals_new.append(3.0)

    elif cleaned_input_train_set['Embarked'].values[i] == 0.0:

        embarked_vals_new.append(0.0)

cleaned_input_train_set['Embarked'] = np.array(embarked_vals_new)



# The 'Cabin' field is similar, but the important part is the section in which the cabin is located. I.e. the first letter

cabin_vals_new = []

for i in range(len(cleaned_input_train_set['Cabin'].values)):

    if cleaned_input_train_set['Cabin'].values[i] == 0.0:

        cabin_vals_new.append(0.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'A':

        cabin_vals_new.append(1.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'B':

        cabin_vals_new.append(2.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'C':

        cabin_vals_new.append(3.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'D':

        cabin_vals_new.append(4.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'E':

        cabin_vals_new.append(5.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'F':

        cabin_vals_new.append(6.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'G':

        cabin_vals_new.append(7.0)

    elif cleaned_input_train_set['Cabin'].values[i][0] == 'T':

        cabin_vals_new.append(8.0)

cleaned_input_train_set['Cabin'] = np.array(cabin_vals_new)

    

# Change ints to floats

cleaned_input_train_set['Parch'] = cleaned_input_train_set['Parch'].astype('float')

cleaned_input_train_set['Pclass'] = cleaned_input_train_set['Pclass'].astype('float')

cleaned_input_train_set['SibSp'] = cleaned_input_train_set['SibSp'].astype('float')



# Normalise data columns, so values for each field are between 0 and 1.

norm_peak_list = [cleaned_input_train_set['Age'].values.max(), cleaned_input_train_set['Fare'].values.max(), cleaned_input_train_set['Pclass'].values.max(),

                 cleaned_input_train_set['Embarked'].values.max(), cleaned_input_train_set['Sex'].values.max(), cleaned_input_train_set['Cabin'].values.max()]



cleaned_input_train_set['Age'] = cleaned_input_train_set['Age']/cleaned_input_train_set['Age'].values.max()

cleaned_input_train_set['Fare'] = cleaned_input_train_set['Fare']/cleaned_input_train_set['Fare'].values.max()

cleaned_input_train_set['Pclass'] = cleaned_input_train_set['Pclass']/cleaned_input_train_set['Pclass'].values.max()

cleaned_input_train_set['Embarked'] = cleaned_input_train_set['Embarked']/cleaned_input_train_set['Embarked'].values.max()

cleaned_input_train_set['Sex'] = cleaned_input_train_set['Sex']/cleaned_input_train_set['Sex'].values.max()

cleaned_input_train_set['Cabin'] = cleaned_input_train_set['Cabin']/cleaned_input_train_set['Cabin'].values.max()

cleaned_input_train_set.head()
# Split up the input training data into a 'test set' and 'train set'.

# The model will then be trained on the 'test set', and the 'train set' will be used to evaluate its performance on unseen data.



input_params_train, input_params_test, target_labels_train, target_labels_test = train_test_split(cleaned_input_train_set, train_target_set, random_state=42)
# Construct the ANN model with a few hidden layers.



titanic_survival_model = Sequential()



# Add an input layer, with a shape matching the number of input parameters in our data set.

titanic_survival_model.add(Dense(input_params_train.shape[1], kernel_initializer='normal', input_dim=input_params_train.shape[1], activation='relu'))



# Add hidden fully-connected layers

titanic_survival_model.add(Dense(input_params_train.shape[1]*4, kernel_initializer='normal', activation='sigmoid'))

titanic_survival_model.add(Dense(input_params_train.shape[1]*4, kernel_initializer='normal', activation='sigmoid'))

titanic_survival_model.add(Dense(input_params_train.shape[1]*4, kernel_initializer='normal', activation='sigmoid'))

titanic_survival_model.add(Dense(input_params_train.shape[1]*4, kernel_initializer='normal', activation='sigmoid'))

titanic_survival_model.add(Dense(input_params_train.shape[1]*8, kernel_initializer='normal', activation='sigmoid'))

titanic_survival_model.add(Dense(input_params_train.shape[1]*8, kernel_initializer='normal', activation='sigmoid'))



# Add output layer

titanic_survival_model.add(Dense(2))



# Compile the model

titanic_survival_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])



# Print a discription of the model

titanic_survival_model.summary()

# Now fit the model.

callback = EarlyStopping(monitor='accuracy', patience=200, verbose=1, restore_best_weights=True)



titanic_survival_model.fit(input_params_train, target_labels_train, epochs=1000, batch_size=20, validation_split=0.2, callbacks=[callback])
#Evaluate the model accuracy

test_loss, test_acc = titanic_survival_model.evaluate(input_params_test, target_labels_test, verbose=2)

prediction_set.head()
cleaned_prediction_set = prediction_set.drop(columns=['Name', 'PassengerId', 'Ticket'])

cleaned_prediction_set.head()



# First, we replace all NaNs with 0s.

cleaned_prediction_set = cleaned_prediction_set.fillna(0.0)



#The field 'Sex' contains only two different values, so can convert 'male' to 1.0 and 'female' to 2.0. The value 0.0 is reserved for NaNs.

sex_vals_new = []

for i in range(len(cleaned_prediction_set['Sex'].values)):

    if cleaned_prediction_set['Sex'].values[i] == 'male':

        sex_vals_new.append(1.0)

    else:

        sex_vals_new.append(2.0)

cleaned_prediction_set['Sex'] = np.array(sex_vals_new)



# Similarly 'Embarked' contains only three different values. The value 0.0 is reserved for NaNs.

embarked_vals_new = []

for i in range(len(cleaned_prediction_set['Embarked'].values)):

    if cleaned_prediction_set['Embarked'].values[i] == 'C':

        embarked_vals_new.append(1.0)

    elif cleaned_prediction_set['Embarked'].values[i] == 'S':

        embarked_vals_new.append(2.0)

    elif cleaned_prediction_set['Embarked'].values[i] == 'Q':

        embarked_vals_new.append(3.0)

    elif cleaned_prediction_set['Embarked'].values[i] == 0.0:

        embarked_vals_new.append(0.0)

cleaned_prediction_set['Embarked'] = np.array(embarked_vals_new)

    

# The 'Cabin' field is similar, but the important part is the section in which the cabin is located. I.e. the first letter

cabin_vals_new = []

for i in range(len(cleaned_prediction_set['Cabin'].values)):

    if cleaned_prediction_set['Cabin'].values[i] == 0.0:

        cabin_vals_new.append(0.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'A':

        cabin_vals_new.append(1.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'B':

        cabin_vals_new.append(2.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'C':

        cabin_vals_new.append(3.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'D':

        cabin_vals_new.append(4.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'E':

        cabin_vals_new.append(5.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'F':

        cabin_vals_new.append(6.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'G':

        cabin_vals_new.append(7.0)

    elif cleaned_prediction_set['Cabin'].values[i][0] == 'T':

        cabin_vals_new.append(8.0)

cleaned_prediction_set['Cabin'] = np.array(cabin_vals_new)

    

# Change ints to floats

cleaned_prediction_set['Parch'] = cleaned_prediction_set['Parch'].astype('float')

cleaned_prediction_set['Pclass'] = cleaned_prediction_set['Pclass'].astype('float')

cleaned_prediction_set['SibSp'] = cleaned_prediction_set['SibSp'].astype('float')



cleaned_prediction_set.head()



# Normalise values

cleaned_prediction_set['Age'] = cleaned_prediction_set['Age']/norm_peak_list[0]

cleaned_prediction_set['Fare'] = cleaned_prediction_set['Fare']/norm_peak_list[1]

cleaned_prediction_set['Pclass'] = cleaned_prediction_set['Pclass']/norm_peak_list[2]

cleaned_prediction_set['Embarked'] = cleaned_prediction_set['Embarked']/norm_peak_list[3]

cleaned_prediction_set['Sex'] = cleaned_prediction_set['Sex']/norm_peak_list[4]

cleaned_prediction_set['Cabin'] = cleaned_prediction_set['Cabin']/norm_peak_list[5]



cleaned_prediction_set.head()
# Make predictions and save submission.



probability_model = tf.keras.Sequential([titanic_survival_model, 

                                         tf.keras.layers.Softmax()])



predictions = probability_model.predict(cleaned_prediction_set)



submissions = pd.DataFrame(columns=['PassengerId', 'Survived'])



submissions['PassengerId'] = prediction_set['PassengerId'].values



survlist = []

for j in range(len(predictions)):

    survlist.append(np.argmax(predictions[j]))



submissions['Survived'] = np.array(survlist)



submissions.head()

submissions.to_csv('my_submission.csv', index=False)