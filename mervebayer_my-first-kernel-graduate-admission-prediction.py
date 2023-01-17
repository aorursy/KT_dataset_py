# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visual libraries

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')

data.head()
data.info()        

                    #GRE Scores ( out of 340 )

                    #TOEFL Scores ( out of 120 )

                    #University Rating ( out of 5 ) (categorical variable -ordinal-)

                    #Statement of Purpose and Letter of Recommendation Strength (out of 5)(categorical variable -ordinal-)

                    #Undergraduate GPA ( out of 10 )

                    #Research Experience ( either 0 or 1 ) (categorical variable -nominal-)

                    #Chance of Admit ( ranging from 0 to 1 )
data.isnull().sum() #checking null values, there is no null values.
data.describe()
sns.pairplot(data) #Plot pairwise relationships in a dataset.
plt.figure(figsize=(15,10))

sns.heatmap(data.corr(), annot=True,fmt=".0%")

plt.show()  #Serial No. has no impact on dataset, no relation with other features

            #CGPA has the most impact on Chance of Admit to university

            #GRE Score is mostly related with CGPA and then Chance of Admit

            #TOEFL Score -> GRE then CGPA

            #LOR doesnt have a big impact on data

            #CGPA -> Chance of Admit, GRE Score and Toefl

            #Research's impact is less

            #Chance of Admit -> CGPA effect is the most and research is the less.
sns.lineplot(x='TOEFL Score', y='Chance of Admit ', data=data)

#When toefl score increased, chance of admit tends to be increased 
sns.lineplot(x='SOP', y='GRE Score', data=data)

#same as toefl
plt.bar(data['Research'].values, data['Chance of Admit '].values, color = 'violet')

bars = ('no research (0)', 'have research (1)')

plt.title('Research effect on Chance of Admit')

plt.xlabel('Research')

plt.ylabel('Chance of Admit')

y_pos = np.arange(len(bars))

plt.xticks(y_pos, bars)





#Having research doesn't have a big effect on chance of admit, in dataset amount of samples who have

#research are more than not having any, that's why we cannot say certain things.
data['Research'].value_counts().plot.pie() 

#in dataset, number of having research experience is bigger than having no research
sns.barplot(x='LOR ', y='Chance of Admit ', hue="Research", data=data, color='blue')

#we cannot get much information of research effect on LOR v Chance of Admit, there is no sample which has 1 recommandation letter and research in dataset.So it seems like there is no chance when you have research and 1 letter.

#But we can say if you have more letters your chance is getting higher.
lor=data['LOR ']

data[lor == 1] 
sns.boxplot(x='LOR ', y='Chance of Admit ', data=data)

#if recommendation letter is higher, possibility of admission is higher but, we can see max values, even 3 letter have a chance to get admitted

#1 letter is not enough for admission
sns.violinplot(x='Research', y='CGPA', data=data)

#in dataset, there are too many samples which doesnt have research and have CGPA between 8.0-8.7, and also have research

#and CGPA is between 8.5-9.5 
 #Using scatter plot

 #A scatter plot is a diagram where each value in the data set is represented by a dot.

 #The x array represents CGPA.

 #The y array represents Chance of Admit.

fig, ax = plt.subplots()



# scatter the CGPA against the Chance of Admit

ax.scatter(data['CGPA'], data['Chance of Admit '])

# set a title and labels

ax.set_title('CGPA effects on Chance of Admit')

ax.set_xlabel('CGPA')

ax.set_ylabel('Chance of Admit')

#As seen in the figure, if CGPA is high, chance of admit tends to be increased. We can say that they are correlated, CGPA has a huge effect on admission
fig, ax = plt.subplots()



# scatter the GRE against the Toefl

ax.scatter(data['GRE Score'], data['TOEFL Score'])

# set a title and labels

ax.set_title('TOEFL effects on GRE')

ax.set_xlabel('GRE')

ax.set_ylabel('TOEFL Score')

#When Toefl is higher, GRE is higher too.


fig, axs = plt.subplots(2)

fig.suptitle('CGPA vs TOEFL & GRE')

axs[0].scatter(data['CGPA'], data['TOEFL Score'])

axs[1].scatter(data['CGPA'], data['GRE Score'])



for ax in axs.flat:

    ax.set(xlabel='CGPA')

    

#If toelf score is higher CGPA tends to be increased. Same for GRE Score. 
data= data.drop(['Serial No.'], axis=1)
X = data.drop(['Chance of Admit '], axis=1)

y = data['Chance of Admit ']

data.head()
from sklearn.preprocessing import normalize



X = normalize(data, norm='l2')

X
from sklearn.model_selection import train_test_split



# 80% Train, 10% Validation, %10 Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
print(f'Total # of sample in whole dataset: {len(X)}')

print(f'Total # of sample in train dataset: {len(X_train)}')

print(f'Total # of sample in validation dataset: {len(X_valid)}')

print(f'Total # of sample in test dataset: {len(X_test)}')
#Use regression for Continuous data



models = {

    'DecisionTree :' : DecisionTreeRegressor(),

    'Linear Regression :' : LinearRegression(),

    'RandomForest :' : RandomForestRegressor(),

    'KNeighbours :' : KNeighborsRegressor(n_neighbors = 4)

}



for m in models:

  model = models[m]

  model.fit(X_train, y_train)

  score = model.score(X_valid, y_valid)

  print(f'{m} validation score => {score}')
model = LinearRegression()

model.fit(X_train, y_train)



validation_score = model.score(X_valid, y_valid)

print(f'Validation score of trained model: {validation_score}')



test_score = model.score(X_test, y_test)

print(f'Test score of trained model: {test_score}')
model = RandomForestRegressor()

model.fit(X_train, y_train)



validation_score = model.score(X_valid, y_valid)

print(f'Validation score of trained model: {validation_score}')



test_score = model.score(X_test, y_test)

print(f'Test score of trained model: {test_score}')