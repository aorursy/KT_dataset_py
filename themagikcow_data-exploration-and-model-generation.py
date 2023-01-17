import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv(os.path.join('../input', 'train.csv'))

test = pd.read_csv(os.path.join('../input', 'test.csv'))



df.head()
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)



df.head()
#To explore numerical corrlations, we need a number to work with.

#We will change text to a numerical value

df.head()    
sns.barplot(x=df['Pclass'] , y=df['Survived'])

plt.show()
sns.barplot(x=df['Sex'] , y=df['Survived'])

plt.show()
# Allow age_df for labels to age so plot is catergoric

age_df = df.drop(['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'], axis=1) 

age_df.head()

#rename intervals, 0-5 = baby, 6-12 = child, 13-19 = teen, 20-60 = adult, 60+ = senior

sns.barplot(x=df['SibSp'] , y=df['Survived'])

sns.barplot(x=df['Parch'] , y=df['Survived'])

sns.barplot(x=df['Fare'] , y=df['Survived'])

df = df.replace(to_replace='male', value =1)

df = df.replace(to_replace='female', value =0) #Male has value 1 and female 0



#Now we can explore how each value is correlated with survival



fig, ax = plt.subplots()



# size A4 

fig.set_size_inches(11.7, 8.27)

sns.heatmap(df.corr(method='spearman'))

plt.show()



df.corr(method='spearman')

sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df)
# Create a dataset of only the values we want

df=df.drop(['Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'], axis=1) 

df.head()
#We can now create our basic model

from sklearn import linear_model



x_train = df[['Pclass', 'Sex']][1:657]

y_train = df[['Survived']][1:657]



x_test = df[['Pclass', 'Sex']][658:-1]

y_test = df[['Survived']][658:-1]



model = linear_model.SGDClassifier()

model.fit(x_train, y_train)



results = model.predict(x_test)

se = pd.Series(results)



#combine prediction with actual answer

results_df = pd.DataFrame(data={'value': list(y_test['Survived']), 'prediction': se}) 



#create accuracy score. Accurate value if 0,0 or 1,1. Inaccurate if 0,1 or 1,1



results_df['score'] = results_df['prediction'] + results_df['value']

scores = results_df['score'].value_counts()



accuracy = ((scores[0] + scores[2])/(scores[0] + scores[1] + scores[2]))*100

print("The accuracy value is ", accuracy, "%")

scores.head()