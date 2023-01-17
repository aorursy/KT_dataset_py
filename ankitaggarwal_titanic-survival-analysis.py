import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sys
# load datasets

training_df = pd.read_csv("../input/train.csv", encoding = "ISO-8859-1", low_memory=False)

testing_df = pd.read_csv("../input/test.csv", encoding = "ISO-8859-1", low_memory=False)

training_df.head()
def female_survival_rate(df):

    total_females = df[df['Sex'] == 'female'].shape[0]

    survived_females = df[(df['Sex'] == 'female') & (df['Survived'] == 1)].shape[0]

    return survived_females / total_females * 100



print('Female Survived Rate: ' + str(female_survival_rate(training_df)))



def male_survival_rate(df):

    total_males = df[df['Sex'] == 'male'].shape[0]

    survived_males = df[(df['Sex'] == 'male') & (df['Survived'] == 1)].shape[0]

    return survived_males / total_males * 100



print('Male Survived Rate: ' + str(male_survival_rate(training_df)))
def fill_random_age(df):

    nan_age_count = df['Age'].isnull().sum()

    if nan_age_count > 0:

        avg_age = df['Age'].mean()

        std_age = df['Age'].std()

        random_age = np.random.randint(avg_age - std_age, avg_age + std_age, size = nan_age_count)

        df['Age'][np.isnan(df['Age'])] = random_age

    return df



def survival_rate_by_age(df):

    survived_traning_df = df[training_df['Survived'] == 1]

    survived_by_age = survived_traning_df.groupby(pd.cut(survived_traning_df.Age, 8, precision=0)).count()['Age']

    total_by_age = df.groupby(pd.cut(df.Age, 8, precision=0)).count()['Age']

    return survived_by_age / total_by_age * 100

    

training_df = fill_random_age(training_df)

survival_rate_by_age = survival_rate_by_age(training_df)



# plot survival_rate_by_age graph

survival_rate_by_age.plot(kind='bar', figsize=(12,5), title='Survival Rate By Passenger Age')

plt.xlabel('Passenger Age')

plt.ylabel('Survival Rate')

plt.show()
def survival_by_passenger_class(df):

    passenger_by_pclass = df[['Survived', 'Pclass']].groupby('Pclass').count()

    survived_by_pclass = df[['Survived', 'Pclass']].groupby('Pclass').sum()

    return survived_by_pclass / passenger_by_pclass * 100



survival_by_passenger_class = survival_by_passenger_class(training_df)



# plot survival_by_passenger_class graph 

survival_by_passenger_class.plot(kind='bar', figsize=(12,5), legend=None, title='Survival Rate By Passenger Class')

plt.xlabel('Survival Rate')

plt.ylabel('Passenger Class')

plt.show()
# convert male-female with binary value

training_df['Sex_Index'] = training_df['Sex'].replace(['female', 'male'], [0, 1])



# convert age into age-range index

age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

training_df['Age_Index'] = pd.cut(training_df.Age, bins=age_bins, labels=age_labels)



# convert training features into numpy matrix

x_train = training_df.as_matrix(columns=['Pclass','Sex_Index','Age_Index'])

y_train = training_df.as_matrix(columns=['Survived']).reshape(-1)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score



# train model with LogisticRegression

lr_model = LogisticRegression(random_state = 1)

lr_model.fit(x_train, y_train)

lr_model_score = lr_model.score(x_train, y_train)

print('LogisticRegression Score: ' + str(lr_model_score))



# train model with RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 100)

rf_model.fit(x_train, y_train)

rf_model_score = rf_model.score(x_train, y_train)

print('RandomForestClassifier Score: ' + str(rf_model_score))



# train model with SVM Classifier

svm_model = svm.SVC(gamma = 0.001, C = 100.)

svm_model.fit(x_train, y_train)

svm_model_score = svm_model.score(x_train, y_train)

print('SVM Classifier Score: ' + str(svm_model_score))



#tranin model with NN MLPClassifier

nn_mlp_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,2), max_iter=500, random_state=1, activation='tanh')

nn_mlp_model.fit(x_train, y_train)

nn_mlp_model_score = nn_mlp_model.score(x_train, y_train)

print('NN MLPClassifier Score: ' + str(nn_mlp_model_score))





""" Check average prediction score by cross-validation on various algorithms """



print("\n\nAgerage prediction score with 10 cross-validations\n")



# cross-validation with LogisticRegression model

lr_model_avg_score = cross_val_score(lr_model, x_train, y_train, cv = 10).mean()

print('LogisticRegression Average Score: ' + str(lr_model_avg_score))



# cross-validation with RandomForestClassifier model

rf_model_avg_score = cross_val_score(rf_model, x_train, y_train, cv = 10).mean()

print('RandomForestClassifier Average Score: ' + str(rf_model_avg_score))



# cross-validation with SVM Classifier model

svm_model_avg_score = cross_val_score(svm_model, x_train, y_train, cv = 10).mean()

print('SVM Classifier Average Score: ' + str(svm_model_avg_score))



# cross-validation with NN MLPClassifier model

nn_mlp_model_avg_score = cross_val_score(nn_mlp_model, x_train, y_train, cv = 10).mean()

print('NN MLPClassifier Average Score: ' + str(nn_mlp_model_avg_score))

""" Prepare test data for actual prediction """

# replace 'NAN' Age By Random Value

testing_df = fill_random_age(testing_df)



# convert male-female with binary value

testing_df['Sex_Index'] = testing_df['Sex'].replace(['female', 'male'], [0, 1])



# convert age into age-range index

age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

age_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

testing_df['Age_Index'] = pd.cut(testing_df.Age, bins=age_bins, labels=age_labels)



# convert testing features into numpy matrix

x_test = testing_df.as_matrix(columns=['Pclass','Sex_Index','Age_Index'])





""" Predict test data with various algorithms """

lr_model_prediction = lr_model.predict(x_test)

rf_model_prediction = rf_model.predict(x_test)

svm_model_prediction = svm_model.predict(x_test)

nn_mlp_model_prediction = nn_mlp_model.predict(x_test)



# convert prediction into submission format

rf_model_prediction = rf_model_prediction.astype(int)

submission = pd.DataFrame({

    'PassengerId': testing_df['PassengerId'],

    'Survived': rf_model_prediction

})

submission.head(10)