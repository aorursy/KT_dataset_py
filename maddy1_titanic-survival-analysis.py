# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')


df.dtypes

df['Age'].count()

df['PassengerId'].count()

missing_age = np.where(df["Age"].isnull() == True)

print(len(missing_age[0]))
age_isnumber=df[df['Age']>=0]
print(len(age_isnumber))
not_survived=df[df['Survived']==0]
print(len(not_survived))
survived=df[df['Survived']==1]
print(len(survived))
%pylab inline
df.hist(column='Age',    # Column to plot
                   figsize=(10,6),   # Plot size
                   bins=20)         # Number of histogram bins
df['filled_ages'] = df['Age'].interpolate()

df['filled_ages']=df['filled_ages'].astype(int)
df['filled_ages'],df['Age']
ax=df['filled_ages'].hist( figsize=(10,6),   # Plot size
                   bins=20)         # Number of histogram bins
ax.set_xlabel('Age')
ax.set_ylabel('Number of Passengers')
#### Lets check the values of fares.How it is different and highes and lowest values.
df.sort_values(['Fare'], ascending=[True])
fig = plt.figure(figsize=(12,4))




ax=df.plot(x='Fare',y='filled_ages',kind='scatter')

fig = plt.figure(figsize=(12,4))
###The features ticket and cabin have many missing values and so can’t add much value to our analysis
df = df.drop(['Ticket','Cabin'], axis=1) 
survived_df=df[df.Survived ==1]
len(survived_df)
#Since we are intersted in the survival factors made a df of the survivors.
# So 342 of 891 survived and hence the survival percentage would be
float(len(survived_df))/len(df)
df.boxplot(column='Fare', by = 'Pclass')

df[df['Fare'] >500] 

temp1 = survived_df.groupby('Sex').Survived.count()
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(122)
ax1.set_xlabel('Sex')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers Survived by Sex")
temp1.plot(kind='bar')


temp2 = df.groupby('Sex').Survived.count()

ax2 = fig.add_subplot(121)
ax2.set_xlabel('Sex')
ax2.set_ylabel('Count of Passengers')
ax2.set_title("Total Passengers by Sex")
temp2.plot(kind='bar')
df.pivot_table('Survived', index='Sex', columns='Pclass')

temp1 = df.groupby('Pclass').Survived.count()
temp2 = df.groupby('Pclass').Survived.sum()/df.groupby('Pclass').Survived.count()
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Pclass')
ax1.set_ylabel('Count of Passengers')
ax1.set_title("Passengers by Pclass")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Pclass')
ax2.set_ylabel('Probability of Survival')
ax2.set_title("Probability of survival by Pclass")
#Some more plots
#Specifying Plot Parameters
# figsize = (x inches, y inches), dpi = n dots per inches
fig = plt.figure(figsize = (11, 8), dpi = 1600)


# Plot: 1
ax4 = fig.add_subplot(221) # .add_subplot(rcp): r = row, c = col, p = position
female_firstclass = df['Survived'][df['Sex'] == 'female'][df['Pclass'] == 1].value_counts()
female_firstclass.plot(kind = 'bar', label = 'Female First Class', color = 'deeppink', alpha = 0.5)
ax4.set_xticklabels(['Survived', 'Dead'], rotation = 0)
ax4.set_xlim(-1, len(female_firstclass))
ax4.set_ylim(0, 400)
ax4.set_title("Female Passengers by First class")
plt.legend(loc = 'best')
#Plot: 2
ax5 = fig.add_subplot(222) # .add_subplot(rcp): r = row, c = col, p = position
female_secondclass = df['Survived'][df['Sex'] == 'female'][df['Pclass'] == 2].value_counts()
female_secondclass.plot(kind = 'bar', label = 'Female Second Class', color = 'deeppink', alpha = 0.5)
ax5.set_xticklabels(['Survived', 'Dead'], rotation = 0)
ax5.set_xlim(-1, len(female_secondclass))
ax5.set_ylim(0, 400)
ax5.set_title("Female Passengers by Second class")
plt.legend(loc = 'best')           
#Plot:3
ax6 = fig.add_subplot(223) # .add_subplot(rcp): r = row, c = col, p = position
female_thirdclass = df['Survived'][df['Sex'] == 'female'][df['Pclass'] == 3].value_counts()
female_thirdclass.plot(kind = 'bar', label = 'Female third Class', color = 'deeppink', alpha = 0.5)
ax6.set_xticklabels(['Survived', 'Dead'], rotation = 0)
ax6.set_xlim(-1, len(female_thirdclass))
ax6.set_ylim(0, 400)
ax6.set_title("Female Passengers by Third class")
plt.legend(loc = 'best')            
print(female_firstclass)
print(female_secondclass)
print(female_thirdclass)
fig = plt.figure(figsize = (15, 12), dpi = 1600)

# Plot: 1
ax6 = fig.add_subplot(321) # .add_subplot(rcp): r = row, c = col, p = position
male_firstclass = df['Survived'][df['Sex'] == 'male'][df['Pclass'] == 1].value_counts()
male_firstclass.plot(kind = 'bar', label = 'male First Class', color = 'green', alpha = 0.5)

ax6.set_xlim(-1, len(male_firstclass))
ax6.set_ylim(0, 400)
ax6.set_title("Male Passengers by First class")

ax7 = fig.add_subplot(322) # .add_subplot(rcp): r = row, c = col, p = position
male_secondclass = df['Survived'][df['Sex'] == 'male'][df['Pclass'] == 2].value_counts()
male_secondclass.plot(kind = 'bar', label = 'male second Class', color = 'green', alpha = 0.5)

ax7.set_xlim(-1, len(male_secondclass))
ax7.set_ylim(0, 400)
ax7.set_title("Male Passengers by Second class")

ax8 = fig.add_subplot(323) # .add_subplot(rcp): r = row, c = col, p = position
male_thirdclass = df['Survived'][df['Sex'] == 'male'][df['Pclass'] == 3].value_counts()
male_thirdclass.plot(kind = 'bar', label = 'male third Class', color = 'green', alpha = 0.5)

ax8.set_xlim(-1, len(male_firstclass))
ax8.set_ylim(0, 400)
ax8.set_title("Male Passengers by Third class")

ax6 = fig.add_subplot(324) # .add_subplot(rcp): r = row, c = col, p = position
kidsmale_firstclass = df['Survived'][df['Sex'] == 'male'][df['filled_ages'] < 18][df['Pclass'] == 1].value_counts()
kidsmale_firstclass.plot(kind = 'bar', label = 'kids male First Class', color = 'blue', alpha = 0.5)

ax6.set_xlim(-1, len(male_firstclass))
ax6.set_ylim(0, 400)
ax6.set_title("Male kids Passengers by First class")

ax6 = fig.add_subplot(325) # .add_subplot(rcp): r = row, c = col, p = position
kidsmale_secondclass =df['Survived'][df['Sex'] == 'male'][df['filled_ages'] < 18][df['Pclass'] == 2].value_counts()
kidsmale_secondclass.plot(kind = 'bar', label = 'kids second Class', color = 'blue', alpha = 0.5)

ax6.set_xlim(-1, len(male_firstclass))
ax6.set_ylim(0, 400)
ax6.set_title("Male kids Passengers by second class")

ax9 = fig.add_subplot(326) # .add_subplot(rcp): r = row, c = col, p = position
kidsmale_thirdclass = df['Survived'][df['Sex'] == 'male'][df['filled_ages'] < 18][df['Pclass'] == 3].value_counts()
kidsmale_thirdclass.plot(kind = 'bar', label = 'kids male third Class', color = 'blue', alpha = 0.5)

ax9.set_xlim(-1, len(male_firstclass))
ax9.set_ylim(0, 400)
ax9.set_title("Male kids Passengers by third class")
print(kidsmale_firstclass)
print(kidsmale_secondclass)
print(kidsmale_thirdclass)
print(male_firstclass)
print(male_secondclass)
print(male_thirdclass)
kidsfmale_firstclass = df['Survived'][df['Sex'] == 'female'][df['Age'] < 18][df['Pclass'] == 1].value_counts()
kidsfmale_secondclass = df['Survived'][df['Sex'] == 'female'][df['Age'] < 18][df['Pclass'] == 2].value_counts()
kidsfmale_thirdclass = df['Survived'][df['Sex'] == 'female'][df['Age'] < 18][df['Pclass'] == 3].value_counts()
print(kidsfmale_firstclass)
print(kidsfmale_secondclass)
print(kidsfmale_thirdclass)
print(female_firstclass)
print(female_secondclass)
print(female_thirdclass)
Class1=df['Survived'][df['Pclass'] == 1].value_counts()
print(Class1)
print(342-136)
print(891-216)
import math
p1=float(136)/216
p2=float(206)/675
n1=216
n2=675
p = float((p1 * n1 + p2 * n2)) / (n1 + n2) 
SE = math.sqrt( p * ( 1 - p ) * ((float(1)/n1) + (float(1)/n2) ))
z = float(p1 - p2) / SE 
print("p:",p)
print("SE:",SE)
print("z:",z)
train_data = pd.read_csv('../input/train.csv')
test_data=test = pd.read_csv('../input/test.csv')


# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = df['Survived']
data = df.drop('Survived', axis = 1)

# Show the new dataset with 'Survived' removed
display(data.head())
#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train_data,test_data]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train_data['Age'] = train_data['Age'].fillna(0)
test_data['Age'] = test_data['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train_data[features].head(3)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# TODO: Initialize the three models
clf_A = tree.DecisionTreeClassifier(random_state=0)

clf_A.fit(train_data[features],train_data[target]) 


predictions = clf_A.predict(test_data[features])

predictions
submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions})
submission.head()
filename = 'submission_1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    
    # Ensure that the number of predictions matches number of outcomes
    if len(truth) == len(pred): 
        
        # Calculate and return the accuracy as a percent
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# Test the 'accuracy_score' function
predictions = pd.Series(np.ones(5, dtype = int))
print(accuracy_score(outcomes[:5], predictions))
print(accuracy_score(outcomes, predictions))

def predictions_1(data):
    """ Model with one feature: 
            - Predict a passenger survived if they are female. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            predictions.append(0)
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_1(data)
def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for _, passenger in data.iterrows():
        if passenger['Sex'] == 'female':
            if passenger['Age'] > 40 and passenger['Age'] < 60 and passenger['Pclass'] == 3:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if passenger['Age'] <= 10:
                predictions.append(1)
            elif passenger['Pclass'] == 1 and passenger['Age'] <= 40:
                predictions.append(1)
            else:
                predictions.append(0)
        
    
    # Return our predictions
    return pd.Series(predictions)

# Make the predictions
predictions = predictions_3(data)
predictions
print(accuracy_score(outcomes, predictions))
