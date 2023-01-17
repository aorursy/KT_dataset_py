import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from sklearn import preprocessing as prep



%matplotlib inline

plt.style.use('seaborn')


train_df = pd.read_csv(r"../input/adult-pmr3508/train_data.csv",na_values="?")

train_df.set_index('Id',inplace=True)

train_df.shape
train_df.head()
totalNull = train_df.isnull().sum()

percentageNull = (totalNull/train_df.shape[0]*100)

missing = pd.concat([totalNull,percentageNull],keys = ['Total Missing','% of Total'], axis = 1)

missing[(missing != 0).all(1)]
train_df['workclass']=train_df['workclass'].fillna(train_df['workclass'].mode().iat[0])

train_df['occupation']=train_df['occupation'].fillna(train_df['occupation'].mode().iat[0])

train_df['native.country']=train_df['native.country'].fillna(train_df['native.country'].mode().iat[0])
totalNull = train_df.isnull().sum()

totalNull.sum()
plt.figure(figsize=(14, 7))

train_df['age'].hist(rwidth=0.9)

plt.ylabel('Individuals')

plt.xlabel('Age')

plt.title('Adult age distribution')
plt.figure(figsize=(14,7))

plt.plot(xlabel='Individuals',ylabel='Race')

train_df['race'].value_counts().plot(kind='bar',width=0.3,align='center')
plt.figure(figsize=(14,7))

plt.plot(xlabel='Individuals',ylabel='Native Country')

train_df['native.country'].value_counts().plot(kind='bar',width=0.3,align='center')
low50 = train_df[train_df.income=='<=50K']

up50 = train_df[train_df.income=='>50K']



plt.figure(figsize=(14, 7))

ax1 = plt.subplot(1,2,1,ylabel='Individuals',xlabel='Age')

low50['age'].hist(rwidth=0.9)

ax1.set_ylim(0,6000)

ax1.set_title('Age distribution of individuals with income <50K')



ax2 = plt.subplot(1,2,2,xlabel='Age')

up50['age'].hist(rwidth=0.9)

ax2.set_ylim(0,6000)

ax2.set_title('Age distribution of individuals with income >=50K')
plt.figure(figsize=(14, 7))

ax1 = plt.subplot(1,2,1,ylabel='Individuals',xlabel='Worked hours per week')

low50['hours.per.week'].hist(rwidth=0.9)

#ax1.set_ylim(0,6000)

ax1.set_title('Individuals with income <50K')



ax2 = plt.subplot(1,2,2,xlabel='Worked hours per week')

up50['hours.per.week'].hist(rwidth=0.9)

ax2.set_ylim(0,14000)

ax2.set_title('Individuals with income >=50K')
males = train_df[train_df.sex=='Male']

females = train_df[train_df.sex=='Female']



labels = 'Male', 'Female'

sizes = [males.count()[0],females.count()[0]]



#function for the plot formatting

def make_autopct(values):

    def my_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return my_autopct



plt.figure(figsize=(14, 7))

plt.pie(sizes, labels = labels,autopct=make_autopct(sizes), colors=['blue','red'],textprops={'fontsize': 16})

plt.title('Adult sex distribution')



print("In the Adult dataset there are {0} females and {1} males.".format(females.count()[0],males.count()[0]))

print("That is: {0:3.2f}% are female and {1:3.2f}% are male.".format(females.count()[0]/train_df.shape[0]*100,males.count()[0]/train_df.shape[0]*100))

print("The same is shown in the graph below")
plt.figure(figsize=(14, 7))

ax1 = plt.subplot(1,2,1,ylabel='Individuals',xlabel='Years of Education')

males['education.num'].hist(rwidth=0.9)

ax1.set_ylim(0,7500)

ax1.set_title('Males')



ax2 = plt.subplot(1,2,2,xlabel='Years of Education')

females['education.num'].hist(rwidth=0.9)

ax2.set_ylim(0,7500)

ax2.set_title('Females')
plt.figure(figsize=(14, 12))

ax1 = plt.subplot(2,1,1,ylabel='Individuals')

males['occupation'].value_counts().plot(kind='bar',width=0.3,align='center')

plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)

ax1.set_ylim(0,4000)

ax1.set_title('Male occupation distribution')





ax2 = plt.subplot(2,1,2)

females['occupation'].value_counts().plot(kind='bar',width=0.3,align='center')

plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)

ax2.set_ylim(0,4000)

ax2.set_title('Female occupation distribution')



plt.tight_layout(pad=2.5)
plt.figure(figsize=(14, 7))

ax1 = plt.subplot(1,2,1,ylabel='Individuals',xlabel='Income')

males['income'].hist(rwidth=0.9)

ax1.set_ylim(0,15500)

ax1.set_title('Male income')



ax2 = plt.subplot(1,2,2,xlabel='Income')

females['income'].hist(rwidth=0.9)

ax2.set_ylim(0,15500)

ax2.set_title('Female income')
plt.figure(figsize=(14, 7))

ax1 = plt.subplot(1,2,1,ylabel='Individuals',xlabel='Worked hours per week')

males['hours.per.week'].hist(rwidth=0.9)

ax1.set_ylim(0,12500)

ax1.set_title('Worked hours per week by males')



ax2 = plt.subplot(1,2,2,xlabel='Worked hours per week')

females['hours.per.week'].hist(rwidth=0.9)

ax2.set_ylim(0,12500)

ax2.set_title('Worked hours per week by females')
train_df.columns
base = train_df

#####Quantitative

#Age: no need to process

#Workclass: 

base['workclass'].unique()
#setting: 0 for Without-pay and Never-worked, 1 for Private, Self-emp-inc, Self-emp-not-inc 

#and 2 for Local-gov, State-gov, Federal-gov:

def work2base(value):

    if value == 'Private' or value == 'Self-emp-inc' or value == 'Self-emp-not-inc':

        return 1

    if value == 'Local-gov' or value == 'State-gov' or value == 'Federal-gov':

        return 2

    return 0



work = pd.DataFrame({'workclassBase': base['workclass'].apply(work2base)})
#fnlwgt: Intuitively disconsidered

#education.num: Will be desconsidered as it is redundant to education and the last better groupped in classes

#capital.gain and capital.loss: Groupped using astype()

median = np.median(base[base['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(base['capital.gain'],

             bins = [-1, 0, median, base['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_grouped = pd.DataFrame({'capital.gain.grouped': aux})

capital_gain_grouped = capital_gain_grouped.astype(np.int)



median = np.median(base[base['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(base['capital.loss'],

             bins = [-1, 0, median, base['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_grouped = pd.DataFrame({'capital.loss.grouped': aux})

capital_loss_grouped = capital_loss_grouped.astype(np.int)



#hours.per.week: Groupped using astype()

aux = pd.cut(base['hours.per.week'], bins = [-1, 25, 40, 60, 200], labels = [0, 1, 2, 3])

hours_per_week_grouped = pd.DataFrame({'hours.per.week.grouped': aux})

hours_per_week_grouped = hours_per_week_grouped.astype(np.int)



#####Qualitative

#education: The education levels are as below.

eduCat = base['education'].unique()

eduCat
#the education will be categorized as follows, from lowest to highest. Number in () represent the index of this

#category in the eduCat array

# (15)Preschool < (11)1st-4th < (5)5th-6th < (12)7th-8th < (10)9th < (1)10th < (14)11th < (7)12th < (2)HS-grad 

#   < (8)Prof-school < (4)Assoc-acdm < (13)Assoc-voc < (0)Some-college < (3)Bachelors < (6)Masters < (9)Doctorate 



#vector with education order corresponding to idx of category in eduCat

eduLevels = [15, 11, 5, 12, 10, 1, 14, 7, 2, 8, 4, 13, 0, 3, 6, 9]

def classify(values, classes, order):

    if order is not None:

        order = np.arange(0,len(eduCat))

    for idx in order:

        if values == classes[idx]:

            return idx

eduBase = pd.DataFrame({'educationBase':base['education'].apply(classify, args =[eduCat,eduLevels])})        

#Finally, eduBase has entries translating the read category from base into the corresponding index of that category in 

#eduCat. That is, if a value of eduBase is 0, that individual as eduCat[0] = 'Some-college' education.

eduBase
#marital.status: Intuitively not considering



#using: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

enc = [prep.LabelEncoder()]

#occupation: 

occupation = pd.DataFrame({'occupationBase':enc[0].fit_transform(base['occupation'])})

occupation.index = base['occupation'].index

#relationship:

relationship = pd.DataFrame({'relationshipBase':enc[0].fit_transform(base['relationship'])})

relationship.index = base['relationship'].index

#race: 

race = pd.DataFrame({'raceBase':enc[0].fit_transform(base['race'])})

race.index = base['race'].index



#sex: Female (0), Male (1)

def sex2base(value):

    if value =='Male':

        return 1

    return 0



sex = pd.DataFrame({'sexBase':base['sex'].apply(sex2base)})



#native.country: As we have seen in the data analysis, most of individuals are from the USA and will be classified as 1.

def country2base(value):

    if value == 'United-States':

        return 1

    return 0



country = pd.DataFrame({'native.countryBase': base['native.country'].apply(country2base)})



#income: Either <=50K (set to 0) or >50K (set to 1). Similar to last country classification. Can use the same function:

def income2base(value):

    if value =='>50K':

        return 1

    return 0



income = pd.DataFrame({'incomeBase':base['income'].apply(income2base)})
newBase = pd.concat([work,capital_gain_grouped,capital_loss_grouped,hours_per_week_grouped,eduBase,occupation,relationship,race,sex,country,income],axis=1)

corr_mat = newBase.corr()

sns.set()

plt.figure(figsize=(20,13))

sns.heatmap(corr_mat, annot=True)
auxBase = base.drop(['fnlwgt', 'education', 'sex', 'native.country', 'workclass', 'marital.status','occupation','relationship','race','income'], axis = 1)

aux = pd.concat([newBase, auxBase], axis = 1)

saveAuxBase = aux

saveAuxBase.head()
aux = aux.astype(np.int)

corr_mat = aux.corr()

plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat,vmax=.99, square=True, annot = True)
dropCols = ['workclassBase','educationBase','native.countryBase','capital.gain','capital.loss','hours.per.week']

aux=aux.drop(dropCols, axis = 1)

corr_mat = aux.corr()

plt.subplots(figsize=(20, 13))

sns.heatmap(corr_mat,vmax=.99, square=True, annot = True)
from sklearn.neighbors import KNeighborsClassifier

#ref:https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

#for this function we must provide at least: number of neighbours and metric parameter  

from sklearn.model_selection import cross_val_score

#ref:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html

#for the cross validation we must provide: estimator, data to fit X, 

from sklearn.preprocessing import StandardScaler

#ref: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
readyBase = aux

#Data to fit in cross validation (X) and the target variables (y):

y = readyBase['incomeBase']

primeX = readyBase.drop(['incomeBase'], axis = 1) 



#For a improved learning process, let's standardize the features 

X = StandardScaler().fit_transform(primeX)
#to save means and deviations of iterations and to compute the maximum accuracy and error

means=[]

devs=[]

maxAcc = 0

maxErr = 0

i = 0

print('*****Initializing k analysis*****')

for k in range(1, 30):

    #for each k in the desired range, do the knn and evaluate the score using cross-validation

    knnClass = KNeighborsClassifier(n_neighbors=k, p = 1)

    score = cross_val_score(knnClass, X, y, cv = 5)

    

    #save iteration mean and standard deviation and update maximum values

    means.append(score.mean())

    devs.append(score.std())

    if means[i] > maxAcc:

        maxK = k

        maxAcc = means[i]

        maxErr = devs[i]

    i = i+1

    if not(k%5):

        print('   K = {0} done. Max Cross Validation accuracy = {1:2.2f}% +/-{3:4.2f}% (best for K = {2})'.format(k, maxAcc*100, maxK, maxErr*100))

print('\nBest choice: K={0} | Max Cross Validation accuracy = {1:2.2f}% +/-{2:4.2f}%'.format(maxK,maxAcc*100,maxErr*100))
plt.figure(figsize=(14, 7))

plt.errorbar(np.arange(1, 30), means, devs, marker = '8', markerfacecolor = 'darkslategrey' , linewidth = 2.5, markersize = 10, 

             color = 'dimgrey', ecolor = 'darkslategrey', elinewidth = 1.5)

yg = []

x = np.arange(0, 31)

for i in range(len(x)):

    yg.append(maxAcc)

plt.plot(x, yg, ':', color = 'red', linewidth = 4)

plt.xlabel('K')

plt.ylabel('Accuracy result (%)')

plt.title('Accuracy results for K from 1 to 30 using CV with 5 folds')

plt.axis([0, 30, min(means) - max(devs), max(means) + 1.5*max(devs)])
k = maxK

knnClass = KNeighborsClassifier(n_neighbors=k, p = 1)

knnClass.fit(X,y)
test_df = pd.read_csv(r"../input/adult-pmr3508/test_data.csv",na_values="?")

test_df.set_index('Id',inplace=True)

test_df.head()
#drop unwanted columns, same as before in Step 3

dropCols = ['workclass','fnlwgt','education','marital.status','native.country']

testBase = test_df.drop(dropCols, axis=1)



#check missing values

totalNull = testBase.isnull().sum()

percentageNull = (totalNull/testBase.shape[0]*100)

missing = pd.concat([totalNull,percentageNull],keys = ['Total Missing','% of Total'], axis = 1)

missing[(missing != 0).all(1)]
#replace missing values of occupation variable using same technique as in step 2:

testBase['occupation']=testBase['occupation'].fillna(testBase['occupation'].mode().iat[0])

totalNull = testBase.isnull().sum()

totalNull.sum()
#capital.gain and capital.loss: Groupped using astype()

median = np.median(testBase[testBase['capital.gain'] > 0]['capital.gain'])

aux = pd.cut(testBase['capital.gain'],

             bins = [-1, 0, median, testBase['capital.gain'].max()+1],

             labels = [0, 1, 2])

capital_gain_grouped = pd.DataFrame({'capital.gain.grouped': aux})

capital_gain_grouped = capital_gain_grouped.astype(np.int)



median = np.median(testBase[testBase['capital.loss'] > 0]['capital.loss'])

aux = pd.cut(testBase['capital.loss'],

             bins = [-1, 0, median, testBase['capital.loss'].max()+1],

             labels = [0, 1, 2])

capital_loss_grouped = pd.DataFrame({'capital.loss.grouped': aux})

capital_loss_grouped = capital_loss_grouped.astype(np.int)



#hours.per.week: Groupped using astype()

aux = pd.cut(testBase['hours.per.week'], bins = [-1, 25, 40, 60, 200], labels = [0, 1, 2, 3])

hours_per_week_grouped = pd.DataFrame({'hours.per.week.grouped': aux})

hours_per_week_grouped = hours_per_week_grouped.astype(np.int)



enc = [prep.LabelEncoder()]

#occupation: 

occupation = pd.DataFrame({'occupationBase':enc[0].fit_transform(testBase['occupation'])})

occupation.index = testBase['occupation'].index

#relationship:

relationship = pd.DataFrame({'relationshipBase':enc[0].fit_transform(testBase['relationship'])})

relationship.index = testBase['relationship'].index

#race: 

race = pd.DataFrame({'raceBase':enc[0].fit_transform(testBase['race'])})

race.index = testBase['race'].index

#sex: Female (0), Male (1)

sex = pd.DataFrame({'sexBase':testBase['sex'].apply(sex2base)})



cols = [capital_gain_grouped,capital_loss_grouped,hours_per_week_grouped,occupation,relationship,race,sex,testBase['age'],testBase['education.num']]

auxTest = pd.concat(cols,axis=1)

auxTest.head()
scalerX = StandardScaler()

primeX = scalerX.fit_transform(auxTest.values)

y_pred = knnClass.predict(primeX)



#results are 0 and 1. To encode back to correct labels:

encY = prep.LabelEncoder()

encodedY= encY.fit_transform(train_df['income'])

predicted = encY.inverse_transform(y_pred)

predicted=pd.DataFrame({'income':predicted})

predicted
predicted.to_csv("submission.csv", index = True, index_label = 'Id')