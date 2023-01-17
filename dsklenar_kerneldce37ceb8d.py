
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# First, let's look at the first few rows. There are 12 columns.
train_df.head()
# Let's look at missing data. Cabin (687), Age (177) are the most missing values. Embarked has 2 missing values.
train_df.isnull().sum()
# Next, let's print some basic statistics about each available feature.
train_df.describe()
# PassengerId is sequential and has no correlation with survival. Let's prove that here.
import matplotlib.pyplot as plt 

import seaborn as sns
sns.boxplot(x="Survived", y="PassengerId", data=train_df)

# The distribution looks the same, so we can drop PassengerId

train_df = train_df.drop('PassengerId', axis=1)
# Continue with the univariate analysis.

# 1. Age

age_plot = sns.FacetGrid(train_df, col='Survived')
age_plot.map(plt.hist, 'Age', bins=25)

# Visually, it looks like there are distinct groups of ages, this might be a good candidate for categorization

# Embarked -- not sure what to think of this, does it have importance? Let's visualize it first
sns.barplot(x="Embarked", y="Survived", data=train_df)

# It looks like there's a significant difference.
# Let's look at the fare.
# sns.boxplot(x="Survived", y="Fare", data=train_df)

# There are significant outliers that skew that graph, but it appears that increased fare increased chances of survival.
# Lets visualize the data with outliers removed

sns.boxplot(x="Survived", y="Fare", data=train_df[train_df.Fare < 150])
# Next, visualize the Pclass.
sns.barplot(x="Pclass", y="Survived", data=train_df)

# First class users are clearly more likely to survive.
# Look at sibling and spouse data.
sns.barplot(x="SibSp", y="Survived", data=train_df)

# Wow! those with 5 or 8 siblings have no chance of survival. Those with 1 or 2 have a higher mean probability
# Look at parent or child data
sns.barplot(x="Parch", y="Survived", data=train_df)

# That completes the univariate analysis. Now lets look at bivariate to see which variables look correlated.

# Naively drop rows with missing data so we can visualize. Also drop Cabin, since it has the most missing data. We will revisit that.
dropped_train_df = train_df.drop('Cabin', axis=1).dropna()
g = sns.pairplot(dropped_train_df, hue='Survived')

# Whoa, that's a lot of information. We can refer to this 
# We want to imput the missing age values intelligently. Let's look at the correlation matrix and see which parameters will be good at estimating
# the missing age values.

# Drop survived because it shouldn't be used to estimate the training data.
corr = dropped_train_df.drop('Survived', axis=1).corr()

print(corr)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# The following was taken from the Seaborn docs:

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Age has the highest correlation with Pclass, Sibsp and Parch, so we'll build a model to fill that way.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

linear_regression = LinearRegression()
linear_reg_df = dropped_train_df.drop('Survived', axis=1)

age = linear_reg_df.Age.values.reshape(-1,1)
age_input_df = linear_reg_df[['Pclass', 'SibSp', 'Parch']].values.reshape(-1, 3)                  

linear_regression.fit(age_input_df, age)

# Show summary statistics for the model:
print('Coefficients: \n', linear_regression.coef_)
print('Intercept: \n', linear_regression.intercept_)
# We have our age, let's fill missing data.
train_df['Age'] = train_df.apply(
    lambda row: 
            -6.01730031 * row.Pclass + -3.93983573 * row.SibSp + 1.40296411 * row.Parch + 45.75401239
            if np.isnan(row.Age) else row.Age, axis=1)
train_df.isnull().sum() 

# We have age data now for all passengers!
# Lets look at the missing embarked data:

train_df[pd.isna(train_df['Embarked'])]

embarked_dropped_train_df = train_df.drop(['Cabin', 'SibSp', 'Parch', 'Ticket', 'Survived'], axis=1).dropna()
g = sns.pairplot(embarked_dropped_train_df, hue='Embarked')

# Hard to see a clear trend, so we'll drop Embarked.

# Also, I'll drop the Cabin data for now, since it's hard to make anything out of the different values.
train_df = train_df.drop('Cabin', axis=1).dropna()
train_df.isnull().sum()

# We have our training data set! Let's build a logistic model.

# Still need to convert the class data (Pclass and Embarked) into one-hot encoded data.
final_training_df = pd.get_dummies(train_df, columns=["Sex", "Embarked", "Pclass"]).drop(['Ticket', 'Name'], axis=1)

final_training_df.head()
# Time to build the Logistic Regression

y_train = final_training_df.Survived.values.reshape(-1,1)
X_train = final_training_df.drop('Survived', axis=1).values.reshape(-1, 12)   

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Show summary statistics for the model:
print('Coefficients: \n', logistic_regression.coef_)
print('Intercept: \n', logistic_regression.intercept_)

# Let's evaluate the accuracy. Since we don't have the test data labels, look at the training data.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = logistic_regression.predict(X_train)

print(confusion_matrix(y_train, y_pred))  
print(classification_report(y_train, y_pred))
print(accuracy_score(y_train, y_pred))

# well, we defintely didn't overfit the data :-)
# Let's perform the same transform on the test data.

test_df.describe()
test_df.isnull().sum()

dropped_test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) # Drop unused features
dropped_test_df['Age'] = dropped_test_df.apply(
    lambda row: 
            -6.01730031 * row.Pclass + -3.93983573 * row.SibSp + 1.40296411 * row.Parch + 45.75401239
            if np.isnan(row.Age) else row.Age, axis=1)
final_training_df = pd.get_dummies(dropped_test_df, columns=["Sex", "Embarked", "Pclass"])

# One Fare value is still missing. Impute with the average.

final_training_df = final_training_df.fillna(final_training_df.mean())
final_training_df.isnull().sum()

submission = logistic_regression.predict(final_training_df)
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'],'Survived': submission})
submission_df.to_csv('Titanic Predictions 1.csv',index=False)

# This scores a 77% on the test data -- let's see if we can do better. We'll still use the same training data, but add
# the name title trick that was shown in class.
# Output the unique title names.
train_df['Title'] = train_df['Name']
train_df['Title'] = train_df['Title'].str.extract('([A-Za-z]+)\.', expand=True)
train_df.Title.value_counts()
# Since gender is captured in a separate feature, we can combine these.
# replace rare titles with more common ones
mapping = {
    'Mlle': 'Miss', 
    'Major': 'Military', 
    'Col': 'Military', 
    'Sir': 'Honorific',
    'Don': 'Honorific', 
    'Mme': 'Miss', 
    'Jonkheer': 'Honorific', 
    'Lady': 'Honorific',
    'Capt': 'Military', 
    'Countess': 'Honorific', 
    'Ms': 'Miss', 
    'Dona': 'Mrs',
    'Master': 'Professional',
    'Dr': 'Professional',
    'Rev': 'Professional'
}

train_df.replace({'Title': mapping}, inplace=True)
train_df.Title.value_counts()

# Great! we reduced it down to six classes: Mr, Miss, Mrs, Professional, Military, Honorific
# Let's see how much that improved our model.
final_training_df = pd.get_dummies(train_df, columns=["Sex", "Embarked", "Pclass", "Title"]).drop(['Ticket', 'Name'], axis=1)

final_training_df.head()
final_training_df.columns

y_train = final_training_df.Survived.values.reshape(-1,1)
X_train = final_training_df.drop('Survived', axis=1).values.reshape(-1, 18)   

from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Show summary statistics for the model:
print('Coefficients: \n', logistic_regression.coef_)
print('Intercept: \n', logistic_regression.intercept_)

y_pred = logistic_regression.predict(X_train)

print(confusion_matrix(y_train, y_pred))  
print(classification_report(y_train, y_pred))
print(accuracy_score(y_train, y_pred)) 

# 2% increase on our training set up to 82%!
# Submit this to see if it improves our test set.
updated_test_data = test_df
updated_test_data['Title'] = test_df['Name']
updated_test_data['Title'] = updated_test_data['Title'].str.extract('([A-Za-z]+)\.', expand=True)
updated_test_data.replace({'Title': mapping}, inplace=True)

updated_test_data.head()

passenger_ids = updated_test_data['PassengerId']
dropped_test_df = updated_test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) # Drop unused features
dropped_test_df['Age'] = dropped_test_df.apply(
    lambda row: 
            -6.01730031 * row.Pclass + -3.93983573 * row.SibSp + 1.40296411 * row.Parch + 45.75401239
            if np.isnan(row.Age) else row.Age, axis=1)
final_test_data = pd.get_dummies(dropped_test_df, columns=["Sex", "Embarked", "Pclass", "Title"])
final_test_data = final_test_data.fillna(final_test_data.mean())
final_test_data['Title_Honorific'] = 0

submission_classes = logistic_regression.predict(final_test_data)
submission_df = pd.DataFrame({'PassengerId': passenger_ids,'Survived': submission_classes})
submission_df.to_csv('Titanic Predictions 2.csv',index=False)

# That made our submission worse (down to 70.7%)! We'll drop that feature on the next iteration
# Let's try one more thing, using categorical age data, rather than the the age alone.

# Since age is a continuous variable, there won't be any age clustering, so let's start by
# arbitrarily creating age categories: 

# - Child (0-12)
# - Teen (13-18)
# - Young Adult (19-30)
# - Adult (31-65)
# - Senior (> 65)

def age_category(age):
    if age < 12:
        cat='child'
    elif age < 19:
        cat='teen'
    elif age < 31:
        cat='young adult'
    elif age < 66:
        cat='adult'
    else:
        cat='senior'
    return cat

age_categories = train_df.Age.apply(age_category)
sns.countplot(age_categories) # The age category histogram looks correct.
# Training the new model.

age_categories_train_df = train_df
age_categories_train_df['Age Category'] = age_categories

# # Still need to convert the categorical data (Sex, Pclass and Embarked and Age Category) into one-hot encoded data.
final_training_df = pd.get_dummies(age_categories_train_df, columns=["Sex", "Embarked", "Pclass", "Age Category"]).drop(['Ticket', 'Title', 'Name', 'Age'], axis=1)
final_training_df.head()

# Time to build the Logistic Regression

y_train = final_training_df.Survived.values.reshape(-1,1)
X_train = final_training_df.drop('Survived', axis=1).values.reshape(-1, 16)   

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Show summary statistics for the model:
print('Coefficients: \n', logistic_regression.coef_)
print('Intercept: \n', logistic_regression.intercept_)

y_pred = logistic_regression.predict(X_train)

print(confusion_matrix(y_train, y_pred))  
print(classification_report(y_train, y_pred))
print(accuracy_score(y_train, y_pred)) # 81.2% accuracy on the training set, that seems to be where we plateau.

# Predict the survival, and save the output.

updated_test_data = test_df
passenger_ids = updated_test_data['PassengerId']

dropped_test_df = updated_test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Title'], axis=1) # Drop unused features
dropped_test_df['Age'] = dropped_test_df.apply(
    lambda row: 
            -6.01730031 * row.Pclass + -3.93983573 * row.SibSp + 1.40296411 * row.Parch + 45.75401239
            if np.isnan(row.Age) else row.Age, axis=1)

test_age_categories = train_df.Age.apply(age_category)

dropped_test_df['Age Category'] = test_age_categories

final_test_data = pd.get_dummies(dropped_test_df, columns=["Sex", "Embarked", "Pclass", "Age Category"])

final_test_data = final_test_data.drop('Age', axis=1)
final_test_data = final_test_data.fillna(final_test_data.mean()) # Fill Missing Fare data

submission_classes = logistic_regression.predict(final_test_data)
submission_df = pd.DataFrame({'PassengerId': passenger_ids,'Survived': submission_classes})
submission_df.to_csv('Titanic Predictions 3.csv',index=False)

# Sweet, that got us to 78.95% -- in the top 32%!
# Let's see what other sklearn ML models we can use. Hany keeps mention SVCs, so let's give those a try.

from sklearn.svm import SVC

svc = SVC(gamma='auto') # Use the default values, since I'm not sure how this works :-)
svc.fit(X_train, y_train) 

svc_predictions = svc.predict(final_test_data)
svc_submission_df = pd.DataFrame({'PassengerId': passenger_ids,'Survived': svc_predictions})
svc_submission_df.to_csv('Titanic Predictions 4.csv',index=False)
# Final notes:

# - Now that I'm at the end, I realize that I should have used k-fold cross validation in order to prevent overfitting of the data
#   and returning an accurate assessment of the accuracy of my various logistic models.
# - The Pipeline functionality in sklearn seems really nice for not repeating a lot of code
# - SVC has a lot of hyperparamets that could be tuned.

# I am pretty happy with top 32% for my first attempt!
# Bonus round: let's try deep learning.

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import plot_model
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(dropout_rate = 0.2, number_of_units = 16, regularization_rate = 0.001):
    classifier = Sequential()

    classifier.add(Dense(units = number_of_units, kernel_initializer = 'uniform', activation = 'relu', input_dim = 16, kernel_regularizer=regularizers.l2(regularization_rate)))
    classifier.add(Dropout(rate = dropout_rate))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.summary()
    return classifier

classifier = create_model()    
history = classifier.fit(X_train, y_train, batch_size = 300, epochs = 1000)

sns.tsplot(history.history['acc']) # Seems like we can keep training....
# Let's try a grid search for hyperparameters.

from sklearn.model_selection import GridSearchCV

def run_grid_search():
    model = KerasClassifier(build_fn=create_model)

    number_of_units = [14, 16]
    batch_size = [10, 100]
    epochs = [300, 400]
    dropout_rate = [0.15]
    regularization_rate = [0.001]

    param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate, number_of_units=number_of_units, regularization_rate=regularization_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X_train, y_train)

    # Summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# run_grid_search()
nn_predictions = classifier.predict(final_test_data)
nn_predictions = nn_predictions.round().astype(int)

nn_predictions = nn_predictions.reshape(-1)


print(nn_predictions)

nn_submission_df = pd.DataFrame({'PassengerId': passenger_ids,'Survived': nn_predictions})
nn_submission_df.to_csv('Titanic Predictions 5.csv',index=False)
