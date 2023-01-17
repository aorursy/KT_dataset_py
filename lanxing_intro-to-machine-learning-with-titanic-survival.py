import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

import seaborn as sns

import os



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.head(5))
full_data = [train, test]

appended_data = train.append(test)

print(appended_data)
print(train.shape[0] - train.count(0)) # Missing 179 for age, 2 for embarked, and 687 for cabin

print(test.shape[0] - test.count(0)) # Missing 1 fare, 86 fare, and 327 cabin



for df in full_data:

    # Fill in missing fare with median value

    df['Fare'] = df['Fare'].fillna(appended_data['Fare'].median())

    

    # Fill in missing embarcation point with most common value

    df.loc[df['Embarked'].isnull()==True, 'Embarked'] = 'S'

    

    # Fill in missing age with random values

    age_avg = appended_data['Age'].mean()

    age_std = appended_data['Age'].std()

    age_nan_count = df['Age'].isnull().sum()

    age_random_filler = np.random.normal(loc=age_avg, scale=age_std, size=age_nan_count)

    df.loc[df['Age'].isnull()==True, 'Age'] = age_random_filler

    df.loc[df['Age'] < 0, 'Age'] = 0

    df['Age'] = df['Age'].astype(int)
for df in full_data:

    ## Dummy for Gender

    df['male'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

    

    ## Dummy for Has_Cabin

    df['has_cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

    

    ## If Cabin is available, which floor were they on?

    df['cabin_level'] = (df.loc[df['Cabin'].str.contains('[a-zA-Z]', na=False, regex=True),

                                'Cabin'].str.get(0))

    df.loc[df['has_cabin']==0, 'cabin_level'] = 'T' # T isn't actually a floor of the ship

    df['cabin_level'] = df['cabin_level'].map({'A':0, 'B':1, 'C':2,

                                               'D':3, 'E':4, 'F':5,

                                               'G':6, 'T':7}).astype(int)

    

    ## Size of family

    # Sum of the number of parents/children with the number of siblings

    df['family_size'] = df['Parch'] + df['SibSp']

    

    ## Is Travelling Alone

    df['Alone'] = df['family_size'].apply(lambda x: 1 if x==0 else 0)

    

    ## Get First and Last Name as well as title

    df[['last_name', 'first_name']] = df['Name'].str.split(',', expand=True, n=2)

    df['title'] = (df['first_name']

                   .str.strip()

                   .str.split(' ')

                   .str.get(0))

    df['title'] = df['title'].replace('Ms.', 'Miss.')

    df['title'] = df['title'].replace('Mme.', 'Mrs.')

    df['title'] = df['title'].replace('Mlle.', 'Miss.')

    df.loc[df['Name'].str.contains('Countess'), 'title'] = 'Countess'

    df['title'] = df['title'].replace(['Dr.', 'Rev.', 'Col.', 

                                       'Major.', 'Jonkheer.', 

                                       'Sir.', 'Capt.', 'Don.'],

                                       'Special_male')

    df['title'] = df['title'].replace(['Countess', 'Lady.', 'Dona.'], 'Special_female')

    df.loc[pd.isnull(df['title']), 'title'] = 'none'

    

    ## It's worth noting that the fare is for the room, but this room may be

    ## shared by multiple people:

    df['ticket_count'] = df.groupby('Ticket')['PassengerId'].transform('count')

    df['ticket_count'].value_counts()

    # In order to get the per-person fare, we can divide by the number of people

    # with the same ticket

    df['fare_pp'] = df['Fare'] / df['ticket_count']

    # Shout out to (https://www.kaggle.com/arjoonn/ticket-fare-analysis)
# Identify good breaking points for the age variable

train['age_cat'] = pd.qcut(x=train['Age'], q=5)

print(train['age_cat'].value_counts())



# Identify good breaking points for the fare variable

train['fare_cat'] = pd.qcut(x=train['Fare'], q=5)

print(train['fare_cat'].value_counts())



# Identify good breaking points for the fare per person variable

train['fare_pp_cat'] = pd.qcut(x=train['fare_pp'], q=5)

print(train['fare_pp_cat'].value_counts())
for df in full_data:

    # Categorize Age

    df['age_cat'] = 0

    df.loc[df['Age'] <= 19, 'age_cat'] = 0

    df.loc[ (df['Age'] > 19) & (df['Age'] <= 25), 'age_cat'] = 1

    df.loc[ (df['Age'] > 25) & (df['Age'] <= 32), 'age_cat'] = 2

    df.loc[ (df['Age'] > 32) & (df['Age'] <= 42), 'age_cat'] = 3

    df.loc[ (df['Age'] > 42), 'age_cat'] = 4

    df['age_cat'] = df['age_cat'].astype(int)

    

    # Categorize fare

    df['fare_cat'] = 0

    df.loc[df['Fare'] <= 7.9, 'fare_cat'] = 0

    df.loc[ (df['Fare'] > 7.9) & (df['Fare'] <= 10.5), 'fare_cat'] = 1

    df.loc[ (df['Fare'] > 10.5) & (df['Fare'] <= 21.7), 'fare_cat'] = 2

    df.loc[ (df['Fare'] > 21.7) & (df['Fare'] <= 39), 'fare_cat'] = 3

    df.loc[ (df['Fare'] > 39), 'fare_cat'] = 4

    df['fare_cat'] = df['fare_cat'].astype(int)

    

    # Categorize fare per person

    df['fare_pp_cat'] = 0

    df.loc[df['fare_pp'] <= 7.73, 'fare_pp_cat'] = 0

    df.loc[ (df['fare_pp'] > 7.73) & (df['fare_pp'] <= 8.05), 'fare_pp_cat'] = 1

    df.loc[ (df['fare_pp'] > 8.05) & (df['fare_pp'] <= 11.73), 'fare_pp_cat'] = 2

    df.loc[ (df['fare_pp'] > 11.73) & (df['fare_pp'] <= 26.55), 'fare_pp_cat'] = 3

    df.loc[ (df['fare_pp'] > 26.55), 'fare_pp_cat'] = 4

    df['fare_pp_cat'] = df['fare_pp_cat'].astype(int)

    

    # Keep a Title variable separately

    df['Title'] = df['title']

    

# Map titles and Embarcation point

# These are categorical features and shoudln't be represented with an ordinal representation

train = pd.get_dummies(data=train, columns=['title', 'Embarked'], prefix=['title', 'embarked'])  

test = pd.get_dummies(data=test, columns=['title', 'Embarked'], prefix=['title', 'embarked'])  
train.columns
print(train.groupby('fare_cat', as_index=False)['Survived'].mean())

print(train.groupby('fare_pp_cat', as_index=False)['Survived'].mean())
features_of_interest = ['Survived', 'Pclass', 'Age', 'SibSp',

       'Parch', 'Fare', 'male', 'has_cabin',

       'cabin_level', 'family_size', 'Alone']

font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 12,

        }

colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title("Correlations of Basic Features", fontdict=font)

sns.heatmap(train[features_of_interest].corr(), cmap=colormap, square=True, fmt='.1f', 

            linewidths=0.1, linecolor='white', annot=True)
features_of_interest = ['Survived', 'Pclass', 'Age', 'Fare', 'male', 'has_cabin', 'family_size']



g = sns.pairplot(train[features_of_interest], hue='Survived', palette = 'seismic',

                 diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))

g.set(xticklabels=[])
# Plot

sns.set_style('white')

fig = plt.figure(figsize=(12,12))

ax = sns.stripplot(x='Title', y='fare_pp', data=train, jitter=0.2,

                  alpha=0.9, hue='Survived', split=False, palette="RdBu")



# Label

title = plt.title("Titles and Money", fontsize=14, fontweight='bold')

title.set_position([.5, 1.03])

plt.ylabel('Fare per Person ($)', fontsize=11, fontweight='bold')

plt.xlabel('Title', fontsize=11, fontweight='bold')

ax.set_ylim(-1,100);



# Y-Axis Ticks

def dollars(x, pos):

    #The two args are the value and tick position

    return '$%1.2f' % (x)

formatter = FuncFormatter(dollars)

ax.yaxis.set_major_formatter(formatter)



# Legend

legend = plt.legend(title='Survived', loc='upper right', ncol=2)

# Set the fontsize

legend.get_title().set_fontsize('11')

legend.get_title().set_fontweight('bold')

for label in legend.get_texts():

    label.set_fontsize('11')



# Remove border

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
facet = sns.FacetGrid(data=train, row='male', hue='Survived', aspect=4)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0,train['Age'].max()))

facet.set(ylim=(0, 0.04))

facet.add_legend()

plt.subplots_adjust(top=0.8)

facet.fig.suptitle("Survival Rates of Men and Women\nDepending on Age", fontweight='bold')
features_of_interest = ['Pclass', 'male', 'has_cabin', 'cabin_level', 

       'family_size', 'Alone', 'age_cat', 'fare_pp_cat', 

       'title_Master.', 'title_Miss.', 'title_Mr.', 'title_Mrs.', #'title_Special_female',

       'title_Special_male', 'embarked_Q', 'embarked_S'] #'embarked_C'



X, y = train[features_of_interest], train['Survived']



X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=829)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

knn.score(X_val, y_val)
k_range = np.arange(1,75)

scores_cols = ['Train', 'Cross-Validation']

scores = pd.DataFrame(columns=scores_cols)



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

    train_score = knn.score(X_train, y_train)

    cv_score = knn.score(X_val, y_val)

    new_entry = pd.DataFrame([[train_score, cv_score]], columns=scores_cols)

    scores = scores.append(new_entry)
plt.style.use('seaborn-deep')

fig, ax = plt.subplots()

fig.set_size_inches(8,6, forward=True)

plt.xlabel('Number of Nearest Neighbors', fontsize=11)

plt.ylabel('Accuracy (%)', fontsize=11)

plt.title('Sensitivity of KNN Classifier\nto Choice of K', fontsize=12, fontweight='bold')

plt.plot(k_range, scores['Train'], '-o', alpha=0.7);

plt.plot(k_range, scores['Cross-Validation'], '-o', alpha=0.7);

plt.legend();
# Scale the features

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_val_scaled = scaler.transform(X_val)



clf = LogisticRegression(C=100) 

clf.fit(X_train_scaled, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(clf.score(X_train_scaled, y_train)))

print('Accuracy of Logistic regression classifier on validation set: {:.2f}'

     .format(clf.score(X_val_scaled, y_val)))
c_range = np.arange(1,50)

scores_cols = ['Train', 'Cross-Validation']

scores = pd.DataFrame(columns=scores_cols)



for c in c_range:

    clf = LogisticRegression(C=c).fit(X_train_scaled, y_train)

    train_score = clf.score(X_train_scaled, y_train)

    cv_score = clf.score(X_val_scaled, y_val)

    new_entry = pd.DataFrame([[train_score, cv_score]], columns=scores_cols)

    scores = scores.append(new_entry)
plt.style.use('seaborn-deep')

fig, ax = plt.subplots()

fig.set_size_inches(8,6, forward=True)

plt.xlabel('Regularization Parameter', fontsize=11)

plt.ylabel('Accuracy (%)', fontsize=11)

plt.title('Sensitivity of Logistic Regression\nto Choice of Regularization Parameter', 

          fontsize=12, fontweight='bold')

plt.plot(c_range, scores['Train'], '-o', alpha=0.7);

plt.plot(c_range, scores['Cross-Validation'], '-o', alpha=0.7);

plt.legend();
this_C = 100

clf = SVC(kernel = 'linear', C=this_C).fit(X_train_scaled, y_train)



print('Accuracy of linear SVM classifier on training set: {:.2f}'

     .format(clf.score(X_train_scaled, y_train)))

print('Accuracy of linear SVM classifier on validation set: {:.2f}'

     .format(clf.score(X_val_scaled, y_val)))
c_range = np.arange(1,200,5)

scores_cols = ['Train', 'Cross-Validation']

scores = pd.DataFrame(columns=scores_cols)



for c in c_range:

    clf = SVC(kernel = 'linear', C=c).fit(X_train_scaled, y_train)

    train_score = clf.score(X_train_scaled, y_train)

    cv_score = clf.score(X_val_scaled, y_val)

    new_entry = pd.DataFrame([[train_score, cv_score]], columns=scores_cols)

    scores = scores.append(new_entry)

    

plt.style.use('seaborn-deep')

fig, ax = plt.subplots()

fig.set_size_inches(8,6, forward=True)

plt.xlabel('Regularization Parameter', fontsize=11)

plt.ylabel('Accuracy (%)', fontsize=11)

plt.title('Sensitivity of Linear SVM\nto Choice of Regularization Parameter', 

          fontsize=12, fontweight='bold')

plt.plot(c_range, scores['Train'], '-o', alpha=0.7);

plt.plot(c_range, scores['Cross-Validation'], '-o', alpha=0.7);

plt.legend();
this_C = 50

clf = SVC(kernel = 'rbf', C=this_C).fit(X_train_scaled, y_train)



print('Accuracy of Kernelized SVM classifier on training set: {:.2f}'

     .format(clf.score(X_train_scaled, y_train)))

print('Accuracy of Kernelized SVM classifier on validation set: {:.2f}'

     .format(clf.score(X_val_scaled, y_val)))
c_range = np.arange(0,201,10)

c_range[0] = 1

gamma_range = np.logspace(-3, 3, 7)

gamma_range = np.flip(gamma_range, axis=0)



train_scores = np.zeros((len(gamma_range),len(c_range),), dtype=np.float64)

train_scores[:] = np.NAN



cv_scores = np.empty((len(gamma_range),len(c_range),))

cv_scores[:] = np.NAN



for i, this_gamma in enumerate(gamma_range):

    for j, this_c in enumerate(c_range):

        clf = SVC(kernel = 'rbf', C=this_c, gamma=this_gamma).fit(X_train_scaled, y_train)

        train_scores[i,j] = clf.score(X_train_scaled, y_train)

        cv_scores[i,j] = clf.score(X_val_scaled, y_val)
cols = [str(c) for c in c_range]



cv_scores_df = pd.DataFrame(data = cv_scores,

                            columns = cols,

                            index = gamma_range)

train_scores_df = pd.DataFrame(data = train_scores,

                               columns = cols,

                               index = gamma_range)

font = {'family': 'sans-serif',

        'color':  'black',

        'weight': 'bold',

        'size': 12,

        }

colormap = plt.cm.viridis

plt.figure(figsize=(12,8))

plt.title("Validation Set Accuracy", fontdict=font)



sns.heatmap(cv_scores_df, cmap=colormap, fmt='.2f', 

            linewidths=0.1, linecolor='white', annot=False)



plt.xlabel('Regularization Parameter', fontsize=11, fontweight='bold')

plt.ylabel('Gamma', fontsize=11, fontweight='bold')
this_C = 50

clf = SVC(kernel = 'poly', degree=3, C=this_C).fit(X_train_scaled, y_train)



print('Accuracy of Kernelized SVM classifier (Poly Kernel) on training set: {:.2f}'

     .format(clf.score(X_train_scaled, y_train)))

print('Accuracy of Kernelized SVM classifier (Poly Kernel) on validation set: {:.2f}'

     .format(clf.score(X_val_scaled, y_val)))