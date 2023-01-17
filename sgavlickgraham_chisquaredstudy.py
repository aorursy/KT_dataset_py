import numpy as np

import pandas as pd

from scipy.stats import chi2_contingency

from scipy.stats import chi2
n = 130  # Number of samples.

d = 1  # Dimensionality. (sex)

c = 2  # Number of categories. (science, math, art)



sex_series = np.random.choice(['Male', 'Female'], size=(n,d))

interest_series = np.random.choice(['Art', 'Math', 'Science'], size=(n,d))



d = {'sex': sex_series, 'interests': interest_series}



data1 = pd.DataFrame.from_dict(list(d['sex']))

data2 = pd.DataFrame.from_dict(list(d['interests']))

data = pd.concat([data1, data2], axis=1)

data.columns = ['sex', 'interests']

data.head()
data.sex.value_counts()
data.interests.value_counts()
# Contingency table.

contingency = pd.crosstab(data['sex'], data['interests'])

contingency
type(contingency)
# Chi-square test of independence.

stat, p, dof, expected = chi2_contingency(contingency)
print('stat:', stat)

print('p:', p)

print('dof:', dof)

print('expected:', expected)
# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
# interpret p-value

alpha = 1.0 - prob

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
print('original contingency table:')

print(contingency)

print()



print('degrees of freedom (dof) = %d' % dof)

print()

print('calculated expected frequency table:')

print(expected)

print()



# interpret test-statistic

print('INTERPRET TEST-STATISTIC (prob = 0.95):')

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))

print('so compare absolute value of stat, %.3f, and critical, %.3f:' % (stat, critical))

print()

if abs(stat) >= critical:

    print('abs(stat) >= critical')

    print('%.3f >= %.3f' % (abs(stat), critical))

    print('Dependent (reject H0)')

else:

    print('abs(stat) < critical')

    print('%.3f < %.3f' % (abs(stat), critical))          

    print('Independent (fail to reject H0)')

print()

print('INTERPRET P-VALUE (alpha = 1 - prob = .05):')

# interpret p-value

alpha = 1.0 - prob

print('p=%.3f, significance=%.3f' % (p, alpha))

print('so compare p, %.3f, with alpha, %.3f:' % (p, alpha))

print()

if p <= alpha:

    print('p <= alpha')

    print('%.3f <= %.3f' % (p, alpha))

    print('Dependent (reject H0)')

else:

    print('p > alpha')

    print('%.3f > %.3f' % (p, alpha))

    print('Independent (fail to reject H0)')
# chi-squared test with similar proportions

from scipy.stats import chi2_contingency

from scipy.stats import chi2

# contingency table

table = [  [10, 20, 30],

           [6,  9,  17]]

print('TABLE USED FOR DATA:')

print(table)

print()



stat, p, dof, expected = chi2_contingency(table)

print('degrees of freedom (dof) =%d' % dof)

print()

print('calculated expected frequency table = ')

print(expected)

print()



# interpret test-statistic

prob = 0.95

critical = chi2.ppf(prob, dof)

print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))



if abs(stat) >= critical:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')

print()



# interpret p-value

alpha = 1.0 - prob

print('significance=%.3f, p=%.3f' % (alpha, p))

if p <= alpha:

    print('Dependent (reject H0)')

else:

    print('Independent (fail to reject H0)')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings

warnings.filterwarnings("ignore")



def wrangle_grades():

    grades = pd.read_csv("../input/student_grades.csv")

    grades.drop(columns='student_id', inplace=True)

    grades.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    df = grades.dropna().astype('int')

    return df



def split_my_data(data):

    '''the function will take a dataframe and returns train and test dataframe split 

    where 80% is in train, and 20% in test. '''

    return train_test_split(data, train_size = .80, random_state = 123)



def standard_scaler(train, test):

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train) # fit the object

    train = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])

    test = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

    return scaler, train, test



# acquire data and remove null values 

df = wrangle_grades()



# split into train and test

train, test = split_my_data(df)



# scale data using standard scaler

scaler, train, test = standard_scaler(train, test)



# to return to original values

# scaler, train, test = scaling.my_inv_transform(scaler, train, test)



X_train = train.drop(columns='final_grade')

y_train = train[['final_grade']]

X_test = test.drop(columns='final_grade')

y_test = test[['final_grade']]
#Using Pearson Correlation

plt.figure(figsize=(6,5))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor["final_grade"])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.5]

relevant_features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



def my_inv_transform(scaler, train, test):

    # return to original values:

    train = pd.DataFrame(scaler.inverse_transform(train), columns=train.columns.values).set_index([train.index.values])

    test = pd.DataFrame(scaler.inverse_transform(test), columns=test.columns.values).set_index([test.index.values])

    return scaler, train, test



scaler, train2, test2 = my_inv_transform(scaler, train, test)



X_train2 = train2.drop(columns='final_grade')

y_train2 = train2[['final_grade']]
chi_selector = SelectKBest(chi2, k=2)



chi_selector.fit(X_train2,y_train2)



chi_support = chi_selector.get_support()

chi_feature = X_train2.loc[:,chi_support].columns.tolist()

chi_feature

print(str(len(chi_feature)), 'selected features')

print(chi_feature)