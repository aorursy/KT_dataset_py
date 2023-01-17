import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



# Combine both train and test in data

reind = range(len(train_data), len(train_data) + len(test_data) )

test_data.index=list(reind)

data = pd.concat([train_data, test_data])

print("Full data shape:", data.shape)

print("Train data shape:", train_data.shape)

print("Test data shape:", test_data.shape)

print("Columns (train):", list(train_data.columns))

print("Columns (test):", list(test_data.columns))



# Print a random passenger record

print(data.iloc[456])



print("Survival rate: %.2f %%" % (data.Survived.sum() * 100 / data.Survived.count()))
import math

import numpy as np



# Copy the original data into data_dumb variable

data_dumb = data.copy()



# Print out number of unique cabins, and assign a number to each of them 

print("Unique cabins:", len(data_dumb['Cabin'].unique()))

cabin_dict = {}

for i, cabin_number in enumerate(data_dumb['Cabin'].unique()):

    cabin_dict[cabin_number] = i

data_dumb['Cabin'] = [cabin_dict[cabin_code] for cabin_code in list(data_dumb['Cabin'])]



# In the Sex column modify 'male' to 0 and 'female' to 1

sex_dict = {'male': 0, 'female': 1}

print("Unique sex entries:", data_dumb['Sex'].unique())

data_dumb['Sex'] = [sex_dict[entry] for entry in data_dumb['Sex']]



# It's easy to replace the nan values in Embarked column with the most used value

print("number of nan values in the Embarked column:", data_dumb['Embarked'].isnull().sum())

most_freq_port = data_dumb.Embarked.dropna().mode()[0]

data_dumb['Embarked'] = data_dumb['Embarked'].fillna(most_freq_port)



# Add numeric value for the Embarked column

data_dumb['Embarked'] = data_dumb['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



# Finally the Age column has many missing values. 

# For this dumb approach it's easiest to just drop all these rows

print("Number of missing Age values:", sum(data_dumb['Age'].isnull()))

data_dumb.dropna(subset=['Age'], inplace=True)



# Drop the name and ticket columns

data_dumb.drop("Name", axis=1, inplace=True)

data_dumb.drop("Ticket", axis=1, inplace=True)
# Create an input matrix X_train, and output Y_train

train_data_dumb = data_dumb[np.isfinite(data_dumb['Survived'])]

test_data_dumb = data_dumb[data_dumb['Survived'].isnull()]

test_data_dumb.drop("Survived", axis=1, inplace=True)

Y_train_dumb = train_data_dumb['Survived']

X_train_dumb = train_data_dumb.drop("Survived", axis=1)



# Assure there's no more NA values

X_train_dumb = X_train_dumb.dropna()

Y_train_dumb = Y_train_dumb.dropna()

X_test_dumb = test_data_dumb.dropna()
def compare_classifiers(X, Y):

    

    import warnings

    warnings.filterwarnings("ignore")

    

    from sklearn.model_selection import cross_val_score

    from sklearn.model_selection import KFold

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.svm import SVC

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

    from sklearn.naive_bayes import GaussianNB

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import accuracy_score, log_loss



    classifiers = [

        KNeighborsClassifier(),

        SVC(),

        DecisionTreeClassifier(),

        RandomForestClassifier(),

        AdaBoostClassifier(),

        GradientBoostingClassifier(),

        GaussianNB(),

        LinearDiscriminantAnalysis(),

        QuadraticDiscriminantAnalysis(),

        LogisticRegression()]



    list_results = []

    kfold = KFold(n_splits=10, random_state=0)

        

    for classifier in classifiers:

        name = classifier.__class__.__name__

        classifier.fit(X, Y)



        # It's useless to calculate the score on the training set (for eg. the Decision Tree will always have score 100)

        train_score = round(classifier.score(X, Y) * 100, 2)



        # Calcuate a cross-validation score

        cv_score = cross_val_score(classifier, X, Y, cv=kfold, scoring='accuracy')



        # Append the results to a list

        list_results.append({'classifier': name, 

                             'train_score': train_score, 

                             'cv_score': round(cv_score.mean() * 100, 2)})



    results = pd.DataFrame(list_results, columns=['classifier', 

                                                  'train_score', 

                                                  'cv_score']).sort_values(by='cv_score', ascending=False)

    return results
results_dumb = compare_classifiers(X_train_dumb, Y_train_dumb)

print(results_dumb)
data.info()
# Print out some names. Some possible titles are: Mr., Mrs., Miss., Master. etc.

print(train_data.Name[10:30])
import re



# The title is always followed by a '. ' (dot and empty space)

data['Title'] = [re.search(r'([a-zA-Z]*)\. ', name).group(1) for name in data['Name'].tolist()]



# Group the data by Sex and Title columns, to see the present titles for each sex category.

data.groupby(['Sex', 'Title']).size()
from collections import Counter



data['Surname'] = [name.split(',')[0] for name in list(data['Name'])]

print("Number of unique surnames:", len(data['Surname'].unique()))

count_surnames = Counter(data['Surname'])



# Print the 10 most common surnames

count_surnames.most_common(10)
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

familysize_counter = Counter(data['FamilySize'])

print(familysize_counter.most_common())
plt.rcParams['figure.figsize'] = (12, 7)

groupby_familysize = data.groupby(['FamilySize', 'Survived'])['Survived'].count().unstack()

groupby_familysize.plot.bar(fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xlabel('Family size', fontsize=14)

plt.legend(['Died', 'Survived'], fontsize=14)
data.Surname, _ = pd.factorize(data.Surname)

data.Title, _ = pd.factorize(data.Title)



data.drop('Name', axis=1, inplace=True)
most_freq_port = data.Embarked.dropna().mode()[0]

data['Embarked'] = data['Embarked'].fillna(most_freq_port)
# Print the survival percentage for each Embarked valeu

data.Survived.groupby(data.Embarked).mean()
data.Embarked = data.Embarked.map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
most_freq_fare = data.Fare.dropna().mode()[0]

data.Fare = data.Fare.fillna(most_freq_fare)
data.Survived.groupby(data.Sex).mean()
data.Sex = data.Sex.map({'male': 0, 'female': 1}).astype(int)
print("Number of NaN values in Cabin:", sum(data.Cabin.isnull()))

print("Number of unique cabins:", len(data.Cabin.unique()))
cabin_count = Counter(data.Cabin)

cabin_count.most_common(20)
data['Deck'] = [record[0] if record is not np.nan else 0 for record in data.Cabin ]

deck_count = Counter(data.Deck)

print("Most common decks:", deck_count.most_common())

survival_deck = data['Survived'].groupby(data['Deck']).mean()

print("Survival rate:", survival_deck)
data.Deck, _ = pd.factorize(data.Deck)

data.drop('Cabin', axis=1, inplace=True)
print("Number of unique Tickets:", len(data.Ticket.unique()))
tickets = [entry.split() for entry in data.Ticket]

ticket_part1 = [item[0] if len(item) > 1 else 0 for item in tickets]

ticket_part2 = [item[1] if len(item) > 1 else item[0] for item in tickets]
tp1_count = Counter(ticket_part1)

tp1_count.most_common(20)
# Create new columns in the dataframe

data['Ticket_desc'] = ticket_part1

data['Ticket_num'] = ticket_part2



# Correct some of the ticket descriptive values

data.Ticket_desc = data.Ticket_desc.replace('SC/Paris', 'SC/PARIS')

data.Ticket_desc = data.Ticket_desc.replace('S.C./PARIS', 'SC/PARIS')

data.Ticket_desc = data.Ticket_desc.replace('SOTON/OQ', 'SOTON/O.Q.')

data.Ticket_desc = data.Ticket_desc.replace('SW/PP', 'S.W./PP')

data.Ticket_desc = data.Ticket_desc.replace('CA.', 'C.A.')

data.Ticket_desc = data.Ticket_desc.replace('CA', 'C.A.')

data.Ticket_desc = data.Ticket_desc.replace('SW/PP', 'S.W./PP')
ticket_groups = data.Fare.groupby(data.Ticket_desc).mean().sort_values(ascending=False)

ticket_groups.plot(kind='bar')

plt.ylabel('Fare')
train_data = data[np.isfinite(data.Survived)].copy()

count_survivals = data.Survived.groupby(data.Ticket_desc).sum()

count_tickets = data.Survived.groupby(data.Ticket_desc).count()

# count_tickets = count_tickets[count_tickets != 0]



# Calculate an average survival rate and assign in to the Ticket_desc values for which we don't have surival information 

#(they're in the test set, but not in the training)

avg_survival = np.sum(train_data['Survived']) / len(train_data)



ticket_desc = []

for cs_index, survived, tickets in zip(count_survivals.index, count_survivals, count_tickets):

    ticket_desc.append({'Ticket': cs_index, 'survival_perc': survived * 100 / float(tickets) if tickets > 0 else avg_survival, 'num_passengers': tickets})



sorted_desc = sorted(ticket_desc, key=lambda k:k['survival_perc'], reverse=True)

for item in sorted_desc[:30]:

    print(item)

    

# Assign the corresponding numbers to the distinct Ticket_desc values

dict_map = {}

for i, item in enumerate(sorted_desc):

    dict_map[item['Ticket']] = i



# Finally, map the descriptive ticket values to the corresponding numeric value

data.Ticket_desc = data.Ticket_desc.map(dict_map).astype(int)



# data.Ticket_desc, _ = pd.factorize(data.Ticket_desc)

data.Ticket_num, _ = pd.factorize(data.Ticket_num)

data.drop('Ticket', axis=1, inplace=True)
print("Number of NaN values in Age:", sum(data.Age.isnull()))
# Group passengers in 10 bands using their age

age_bins = [0, 10, 20, 30, 40, 50, 60, 70]

data['AgeBand'] = pd.cut(data.Age, age_bins)



# Group female and male data according to survival

female_data = data[data.Sex==1]

male_data = data[data.Sex==0]

female_survival = female_data.Survived.groupby(female_data.AgeBand).mean()*100

male_survival = male_data.Survived.groupby(male_data.AgeBand).mean()*100



# Plot the survival rate (percentage)

plt.plot([5, 15, 25, 35, 45, 55, 65], female_survival, 'o-', label='female')

plt.plot([5, 15, 25, 35, 45, 55, 65], male_survival, 'o-', label='male')

plt.ylabel('Survival rate [%]')

plt.xlabel('Age band')

data.drop('AgeBand', axis=1, inplace=True)
print("Number of missing Age values: %d, (training: %d, test: %d)" %

      (sum(data.Age.isnull()), sum(train_data.Age.isnull()), sum(test_data.Age.isnull())))
# Assign Age column to Y_age, and drop Fare and PassengerId as they seem unrelated to passenger's Age

data_copy = data.copy()

data_copy_train = data_copy.dropna(subset=['Age'])

data_copy_test = data_copy[data_copy['Age'].isnull()]



X_age_train = data_copy_train.drop(['Age', 'Fare', 'PassengerId', 'Survived'], axis=1)

Y_age_train = data_copy_train.Age

X_age_test = data_copy_test.drop(['Age', 'Fare', 'PassengerId', 'Survived'], axis=1)
from sklearn.svm import SVR



# I found the best parameters with trial-error

svr_rbf = SVR(kernel='rbf', C=7e3, gamma=0.0003)

y_predicted = svr_rbf.fit(X_age_train, Y_age_train).predict(X_age_test)



print("Mean age (of available data): %.2f, standard deviation: %.2f" % (Y_age_train.mean(), Y_age_train.std()))

print("Mean predicted age: %.2f, standard deviation: %.2f" % (y_predicted.mean(), y_predicted.std()))



# Discretize Age values and plot histograms

y_predicted = pd.DataFrame([int(value) for value in y_predicted], index=data_copy_test.index, columns=['Age'])

plt.hist(y_predicted.Age, 30, normed=True, color='blue', alpha=0.5, label='Predicted Age')

plt.hist(Y_age_train, 30, normed=True, color='red', alpha=0.5, label='Age')

plt.xlabel('Age')

plt.ylabel('Normalised count')

plt.legend()
data_copy.Age.fillna(y_predicted.Age, inplace=True)

data_copy.head(20)
# Create a new copy of the data so we can run this cell over-and-over, without running the above ones

data_predict = data_copy.copy()



data_train = data_predict[np.isfinite(data_predict.Survived)]

data_test = data_predict[data_predict.Survived.isnull()]



X_train = data_train.drop(['Survived'], axis=1)

Y_train = data_train.Survived

X_test = data_test.drop(['Survived'], axis=1)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

import operator



# Create the model, Y_pred is the survival prediction, acc_log1 is the accuracy of the model



logreg = LogisticRegression(C=0.1)

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

dict_ = {}

for coef_, feature in zip(logreg.coef_[0], X_train.columns):

    dict_[feature] = coef_

sorted_dict = sorted(dict_.items(), key=operator.itemgetter(1), reverse=True)

for item in sorted_dict:

    print(item)

    

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print("Model accuracy:", acc_log)
from sklearn.preprocessing import scale



# Create a new copy of the data

data_scaled = data_copy.copy()

X_train_scaled = data_scaled[np.isfinite(data_copy.Survived)]

X_test_scaled = data_scaled[data_scaled.Survived.isnull()]

X_test_scaled = X_test_scaled.drop('Survived', axis=1)



# no need to scale the output (target) column

Y_train_scaled = X_train_scaled.Survived  

X_train_scaled = X_train_scaled.drop('Survived', axis=1)



# Go through all columns and scale the values in each

for column in X_train_scaled.columns:

    column_data = scale(X_train_scaled[column].dropna())

    X_train_scaled[column] = column_data

    

for column in X_test_scaled.columns:

    column_data = scale(X_test_scaled[column].dropna())

    X_test_scaled[column] = column_data
res_scaled = compare_classifiers(X_train_scaled, Y_train_scaled)

print(res_scaled)
results_new = compare_classifiers(X_train, Y_train)

results_compare = results_dumb.copy()

results_compare['feng'] = results_new.cv_score

results_compare['dumb'] = results_compare.cv_score

results_compare['scale'] = res_scaled.cv_score

results_compare.drop(["train_score", "cv_score"], axis=1, inplace=True)

results_compare['dumb->feng [%]'] = round((results_compare.feng - results_compare.dumb) * 100 / results_compare.dumb, 2)

results_compare['feng->scale [%]'] = round((results_compare.scale - results_compare.feng) * 100 / results_compare.feng, 2)

results_compare = results_compare[['classifier', 'dumb', 'feng', 'scale', 'dumb->feng [%]', 'feng->scale [%]']]

print(results_compare)