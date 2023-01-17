''' This code will import the relevant files for data analysis purposes.

    The data file is located in ../input/train.csv '''



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # package to embellish graphs / plots

import matplotlib.pyplot as plt

import pylab as pl



# Import the train.csv file into a dataframe

titanic_data = pd.read_csv('../input/train.csv')
# Get an overview of the various statistics of titanic_data

titanic_data.describe()
# Identify the missing entries for the various indices in order to better assess what data needs to be cleansed

titanic_data.info()
def clean_names(df):

    df['LastName'], df['FirstName'] = df['Name'].str.split(', ', 1).str

    df['Title'], df['FirstName'] = df['FirstName'].str.split('. ', 1).str

    del df['Name']

    print('Split name columns in FirstName / LastName / Title')

    return df



# Remove passengers that didn't embark (i.e. resulting in survivor rate higher than expected)

def clean_unembarkedPassengers(df):

    numPassengers = len(df) 

    df = df.drop(titanic_data.Embarked.notnull())

    print('Removed {} passengers that have not embarked'.format(numPassengers - len(df)))

    return df



def clean_cabin(cabin):

    

    cabinString = str(cabin)

    if cabinString == 'nan':

        return 'Z'

    else:

        return cabinString[0]



def standardize_titles(title):

    if title == 'Mr':

        return 'Mr'

    elif title == 'Mrs':

        return 'Mrs' 

    elif title == 'Mme':

        return 'Mme'

    elif title == 'Miss':

        return 'Miss'

    else:

        return 'Special'

    

# Run data cleanup

titanic_data = clean_names(titanic_data)

titanic_data = clean_unembarkedPassengers(titanic_data)

titanic_data['Title'] = titanic_data['Title'].apply(standardize_titles)

print('Standardized all titles')

titanic_data['Cabin'] = titanic_data['Cabin'].apply(clean_cabin)

print('Cleansed cabin names')
def calculateSurvivorRatebyCriteria(df, criteria):

    print('Overall survival rate is: {}'.format(df['Survived'].mean()))

    survivalRateByCriteria = titanic_data.groupby(criteria).sum()['Survived'] / titanic_data.groupby(criteria).count()['Survived']

    #print survivalRateByCriteria

    ax = survivalRateByCriteria.plot(kind='barh',title='Survival rate by: {}'.format(criteria))    

    

def CompareWithPieCharts(df1,df2, desc1, desc2):

    survivalRate1 = df1['Survived'].mean()

    survivalRate2 = df2['Survived'].mean()

    pieData = [{desc1: survivalRate1, desc2: survivalRate2}, {desc1: 1-survivalRate1, desc2: 1-survivalRate2}]



    df = pd.DataFrame(pieData, index=['Survived', 'Deceased'])

    return df.plot.pie(subplots = True,figsize=(8,4), 

                       title = 'Survival rate of {} vs. {} (in %)'.format(desc1,desc2), colormap='Pastel2',#colors=['g','r'],

                      legend = False, autopct='%1.1f%%')
cherbourg_data = titanic_data.loc[titanic_data['Embarked'] == 'C']

CompareWithPieCharts(cherbourg_data, titanic_data, 'Cherbourg Passengers', 'All Passengers')
def plotTwoVars(clf, X_test, y_test):

    x_min = 0.0; x_max = 600.0

    y_min = 0.0; y_max = 100.0



    # Plot the decision boundary. For that, we will assign a color to each

    # point in the mesh [x_min, m_max]x[y_min, y_max].

    h = .01  # step size in the mesh

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])



    # Put the result into a color plot

    Z = Z.reshape(xx.shape)

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())



    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)



    # Plot also the test points

    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]

    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]

    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]

    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]



    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")

    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")

    plt.legend()

    plt.xlabel("Fare")

    plt.ylabel("Class")



    #plt.savefig("test.png")

    return plt
def classify(features_train, labels_train):   

    ### import the sklearn module for GaussianNB

    ### create classifier

    ### fit the classifier on the training features and labels

    ### return the fit classifier

    

    

    ### your code goes here!

    

    from sklearn.naive_bayes import GaussianNB

    return GaussianNB().fit(features_train, labels_train)
#men_data = titanic_data.loc[titanic_data['Sex'] == 'male']

#women_data = titanic_data.loc[titanic_data['Sex'] == 'female']

#age_data = titanic_data['Age'].fillna(titanic_data['Age'].mean())



features_data = [[data1, data2 == 'male'] for data1, data2 in zip(titanic_data['Fare'], titanic_data['Sex'])]

survivor_data = [data3 ==1 for data3 in titanic_data['Survived']]





clf = classify(features_data, survivor_data)



#plt.figure()

#plt = plotTwoVars(clf, features_data, survivor_data)

plt.show()