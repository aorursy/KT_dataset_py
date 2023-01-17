import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.feature_selection import f_classif
df = pd.DataFrame(load_wine().data, columns=load_wine().feature_names) #transforming the dataset to dataframe type

df['wine'] = pd.Series(load_wine().target) #adding target column

df.head()
df.describe(include='all')
df.info()
sns.barplot(x=df.wine.value_counts(normalize=True).index, y=df.wine.value_counts(normalize=True).values, palette='winter')
for feature in df.columns:

    print(feature)

    print('Skewness: ', df[feature].skew())

    print('Kurtosis: ', df[feature].kurt())

    print('---------------')
plt.figure(figsize=(12,12))

plt.title('Paerson correlation coefficient between dataset variables \n')

sns.heatmap(df.drop('wine',axis=1).corr(), annot=True, cmap='viridis_r')
scores = f_classif(df.drop('wine', axis=1), df.wine)   #performing ANOVA for the features against the target



for i in range(len(df.columns)-1):                     #printing f-value and p-value for every feature

        print(df.drop('wine', axis=1).columns[i] + ': F-value --> ', scores[0][i], ' p-value --> ', scores[1][i])

        print('--------------------------------------------------------------------------------')
df_scores = pd.DataFrame({'features': df.drop('wine',axis=1).columns, 'ANOVA_f_values': scores[0]}).sort_values(by=['ANOVA_f_values'], ascending=False)

plt.figure(figsize=(10,6))

bar = sns.barplot(x='features', y='ANOVA_f_values', data=df_scores)

bar.set_xticklabels(bar.get_xticklabels(),rotation=90);

bar.set_title('ANOVA f_values for each label against target')
sns.pairplot(data=df[['flavanoids','proline','od280/od315_of_diluted_wines','alcohol','color_intensity','wine']], hue='wine')
scaler = MinMaxScaler() #initializing a scaler which will transform all data in numbers with range (0,1)

X = df.drop('wine', axis=1)

y = df.wine.values #separating labels from target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) #splitting the data into training set and test set

X_train = scaler.fit_transform(X_train) #scaling separately train and test sets

X_test = scaler.fit_transform(X_test)
clf = SVC()

cv = StratifiedKFold(n_splits=10, random_state=56, shuffle=True) #initializing StratifiedKFold for cross validation purpose



#dictionary of possible hyperparameters values of SVC model

param = {'C': [0.1,1, 10, 100],                        # C is a regularization parameter, squared l2 parameter

         'gamma': [0.01,0.1,1,10],                     # gamma is the kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’

         'kernel': ['poly','linear','rbf','sigmoid'],  # kernel type to be used in the algorithm

         #'degree': [2,3,4]                            # degree is the degree of poly function

        }





grid = GridSearchCV(clf, param_grid = param, cv=cv, scoring='accuracy', verbose=1) #initializing a cross validation for tuning

grid.fit(X_train, y_train) #fitting on the training set



print('Best estimator is:\n\n',grid.best_estimator_)

print('\n')

print('Average accuracy of the model on the training set is: ', grid.best_score_)
pred = grid.best_estimator_.predict(X_test) #predicting target for the test set

print(confusion_matrix(y_test, pred))

print('\nAccuracy Score: ', accuracy_score(y_test, pred))