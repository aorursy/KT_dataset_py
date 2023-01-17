import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')

import numpy as np

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn import preprocessing, cross_validation

import pandas as pd

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

import seaborn as sns
# Read CSV files to fetch train, test data

df1 = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')

#print(df1.head())

#print(df2.head())

#print(df.columns.values)


#print(set(df['Survived'].values.tolist()))



#print(df.head())

df1.convert_objects(convert_numeric=True)

df1.fillna(0, inplace = True)



df2.convert_objects(convert_numeric=True)

df2.fillna(0, inplace = True)
def handle_non_numerical_data(df):

    columns = df.columns.values

    for column in columns:

        text_digit_vals = {}

        def convert_to_int(val):

            return text_digit_vals[val]

            

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            unique_elements = set(df[column].values.tolist())

            x = 0

            for unique in unique_elements:

                if unique not in text_digit_vals:

                    text_digit_vals[unique] = x

                    x +=1

            

            df[column] = list(map(convert_to_int, df[column])) 

            

            

    

    return df
print(df1.columns.values)



df1a = handle_non_numerical_data(df1)

df2a = handle_non_numerical_data(df2)

#print(df2.columns.values)

#print (np.array(df2).shape)



#print(df2.head())

#print(df.head())



# drop 'Survived' column from training data

# drop 'PassengerId' column as it has low covariance in predicting survivors



X_train = np.array(df1a.drop(['Survived'], 1).astype(float))

X_train = np.array(df1a.drop(['PassengerId'], 1).astype(float))

X_train = np.array(df1a.drop(['Ticket'], 1).astype(float))

X_train = np.array(df1a.drop(['Name'], 1).astype(float))





# process training data to scale

X_train = preprocessing.scale(X_train)



# create training validation data for training

y_train = np.array(df1a['Survived'].astype(int))





# use KMeans for training

#clf = KMeans(n_clusters = 2)

#clf.fit(X_train)



# use LogisticRegression for training



clf = LogisticRegression()

clf.fit(X_train, y_train)





# use SVM for training

#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],

#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

#clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=10)

#clf = clf.fit(X_train, y_train)



X_test = np.array(df2a.drop(['PassengerId'], 1).astype(float))

X_test = np.array(df2a.drop(['Ticket'], 1).astype(float))

X_test = np.array(df2a.drop(['Name'], 1).astype(float))



X_test = np.array(df2a.astype(float))

X_test = preprocessing.scale(X_test)



#X_test = np.array(df2.drop(['Survived'], 1).astype(float))



# declare an empty list for capturing predicted results

predict_test = []



i = 0

for i, test_features in enumerate(X_test):

    # reshape to list of list for prediction

    test_features_reshaped = test_features.reshape(-1,len(test_features))

    z = clf.predict(test_features_reshaped)

#    print(z)

    predict_test.append(z)



    

# count no. of predicted survivors vs age using dictionaries

count = 0

AgeVsSurvived = {}

NoOfPassengersOfGivenAge = {}

AgeUniqueVals = set(df2['Age'].values.tolist())

#print(max(AgeUniqueVals))





# populate indexes for the two dictionaries with same set of keys (ages)

for i in range(int(max(AgeUniqueVals))+1):

    if i not in NoOfPassengersOfGivenAge:

        AgeVsSurvived[int(i)] = 0

        NoOfPassengersOfGivenAge[int(i)] = 0

        

for i in df2['Age']:

    NoOfPassengersOfGivenAge[int(i)] += 1



                             

                             

#print(NoOfPassengersOfGivenAge)                             



# check that we go no. of passengers right in our dictionary

sumofpassengers = 0

for i in NoOfPassengersOfGivenAge:

    sumofpassengers += NoOfPassengersOfGivenAge[i]





i = 0    

age = []



# capture the predicted survivors against the 'age' index in the AgeVsSurvived dict

for i,j in enumerate(predict_test):

    AgeVsSurvived[int(df2['Age'][i])] += int(predict_test[i])



#print(sum(predict_test))    

#print(sumofpassengers)



# Plot age vs suvivors as line chart

AgeList = []

PredictedSurvivorsByAge = []

SurvivorsAgeList = []

ListofNoOfPassengersbyAge=[]

for i in AgeVsSurvived:

    AgeList.append(i)

    PredictedSurvivorsByAge.append(AgeVsSurvived[i])

    ListofNoOfPassengersbyAge.append(NoOfPassengersOfGivenAge[i])





#print(NoOfPassengersOfGivenAge)    



#print(AgeVsSurvived)    

    

width = 0.50       # the width of the bars

fig = plt.figure()

ax = plt.subplot2grid((1,10), (0,0), rowspan = 1, colspan = 10)

#rects1 = ax.bar(AgeList, SurvivorsByAgeList, width, color='r')

line = ax.plot(AgeList, PredictedSurvivorsByAge, color='r', label = 'Survived')

line2 = ax.plot(AgeList, ListofNoOfPassengersbyAge, color='g', label = 'Total Passengers')





# add some text for labels, title and axes ticks

ax.set_ylabel('No of people')

ax.set_xlabel('Age')

ax.set_title('Predicted Survivors by age')

ax.set_xticks(AgeList)

ax.set_xticklabels(AgeList)

plt.subplots_adjust(left=0.09, bottom= 0.18, right = 0.94, top = 0.85, wspace = 0.2, hspace = 0)

plt.legend()

plt.show()



#print(predict_test)

predicted_result = []

for i,j in enumerate(predict_test):

    predicted_result.append([int(df2['PassengerId'][i]), int(j)])

    



#print(type(predicted_result))        

#print(np.array(predicted_result)[:,(0)])    



#result = pd.DataFrame({

#        "PassengerId": np.array(predicted_result)[:,(0)],

#        "Survived": np.array(predicted_result)[:,(1)]

#    })

#result.to_csv('titanic_result.csv', index=False)

# Creating some trend graphs using training data itself



count_trn = 0

AgeVsSurvived_trn = {}

NoOfPassengersOfGivenAge_trn = {}

AgeUniqueVals_trn = set(df1['Age'].values.tolist())

#print(max(AgeUniqueVals_trn))



#print(len(X_train))



# populate indexes for the two dictionaries with same set of keys (ages)

for i in range(int(max(AgeUniqueVals_trn))+1):

    if i not in NoOfPassengersOfGivenAge_trn:

        AgeVsSurvived_trn[int(i)] = 0

        NoOfPassengersOfGivenAge_trn[int(i)] = 0

        

#print(NoOfPassengersOfGivenAge_trn)

                                     

for i in df1['Age'].astype(int):

    NoOfPassengersOfGivenAge_trn[int(i)] += 1



                                 

# check that we go no. of passengers right in our dictionary

sumofpassengers_trn = 0

for i in NoOfPassengersOfGivenAge_trn:

    sumofpassengers_trn += NoOfPassengersOfGivenAge_trn[i]





#print(sumofpassengers_trn)



i = 0    

age_trn = []



# capture the no. of survivors against the 'age' index in the AgeVsSurvived dict

for i,j in enumerate(df1['Age'].astype(int)):

    AgeVsSurvived_trn[j] += int(df1['Survived'][i])



#print(AgeVsSurvived_trn)



# Plot age vs suvivors as line chart

AgeList_trn = []

SurvivorsByAge_trn = []

SurvivorsAgeList_trn = []

ListofNoOfPassengersbyAge_trn=[]

for i in AgeVsSurvived_trn:

    AgeList_trn.append(i)

    SurvivorsByAge_trn.append(AgeVsSurvived_trn[i])

    ListofNoOfPassengersbyAge_trn.append(NoOfPassengersOfGivenAge_trn[i])



#print(ListofNoOfPassengersbyAge_trn)



width = 0.50       # the width of the bars

fig_trn = plt.figure()

ax_trn = plt.subplot2grid((1,10), (0,0), rowspan = 1, colspan = 10)

#rects1 = ax.bar(AgeList, SurvivorsByAgeList, width, color='r')

line1_trn = ax_trn.plot(AgeList_trn, SurvivorsByAge_trn, color='r', label = 'Survived')

line2_trn = ax_trn.plot(AgeList_trn, ListofNoOfPassengersbyAge_trn, color='g', label = 'Total Passengers')





# add some text for labels, title and axes ticks

ax_trn.set_ylabel('No of people')

ax_trn.set_xlabel('Age')

ax_trn.set_title('Survivors by age in training set')

ax_trn.set_xticks(AgeList_trn)

ax_trn.set_xticklabels(AgeList_trn)

plt.subplots_adjust(left=0.09, bottom= 0.18, right = 0.94, top = 0.85, wspace = 0.2, hspace = 0)

plt.legend()

plt.show()
#X_train

grid = sns.FacetGrid(df1, col='Survived', row='Pclass', size=3.0, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# Test prediction accuracy on the training set itself

       

correct = 0



#print(range(len(X_train)))



for i in range(len(X_train)):

    predict_me = np.array(X_train[i]).astype(float)

    #print('now preparing predict_me i = ' + str(i))

    #print(predict_me)

    predict_me = predict_me.reshape(-1, len(predict_me))



    #print('now reshaping predict_me i = ' + str(i))

    #print(predict_me)



    prediction = clf.predict(predict_me)

    

    #print('now check y[i] = ' )

    #print(y[i])

    #print('prediction', prediction, df1['Survived'][i])

    

    if prediction==df1['Survived'][i]:

        correct += 1

        

print ('Accuracy of prediction over the training set = ', correct/len(X_train), correct)
#def heatmap(labellist):

#    heatmatrix = df1[labellist].T.dot(df1[labellist])

#    sns.heatmap(heatmatrix)

#    return heatmatrix

    

# Compute the heatmap matrix

#heatmap(['Pclass', 'Survived', 'Sex'])



corr = df1.corr()

print(corr)

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 },ax=ax, annot = True, annot_kws = { 'fontsize' : 12 })
correlation = df1.corr()

print(correlation)

fig , ax = plt.subplots( figsize =( 12 , 12) )

cmap = sns.diverging_palette( 255 , 0 , as_cmap = True )

fig = sns.heatmap(correlation,cmap = cmap, square=True,cbar_kws={ 'shrink' : .9 },ax=ax,annot = True,annot_kws = { 'fontsize' : 10 })
pclass = pd.get_dummies( df1.Pclass , prefix='Pclass' )

pclass.head()

#print(pclass.shape)
embarked = pd.get_dummies( df1.Embarked , prefix='Embarked' )

embarked.head()

#print(set(df1['Embarked'].values.tolist()))
