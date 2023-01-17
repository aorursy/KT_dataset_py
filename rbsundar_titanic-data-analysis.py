import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 
traindf = pd.read_csv("../input/train.csv")
traindf.info()
traindf.describe()
testdf = pd.read_csv("../input/test.csv")
testdf.info()
testdf.describe()
m_traindf = traindf.copy()
m_testdf = testdf.copy()
title_list = ['SKIP', 'Mr.', 'Mrs.', 'Ms.', 'Miss.', 'Don.', 'Master.']              
#, 'Lady.', 'Sir.', 'Countess.', 'Rev.', 'Dr.', 'Major.', 'Capt.', 'Col.', 'Jonkheer.', 'Mlle.', 'Mme.']

# age - replace nulls with random values between +- Standard deviations
age_mean = m_traindf.Age.mean()
age_std = m_traindf.Age.std()
age_nulls = len(m_traindf[m_traindf.Age.isnull()])
age_vals = np.random.randint(age_mean - age_std, age_mean + age_std, age_nulls)
m_traindf.loc[m_traindf.Age.isnull(), 'Age'] = age_vals

# Embarked - replace nulls with oft-repeating values
embarked_mode = m_traindf.Embarked.mode()[0]
m_traindf.loc[m_traindf.Embarked.isnull(), 'Embarked'] = embarked_mode

# New features
m_traindf['SexCat'] = [0 if s == 'male' else 1 for s in m_traindf.Sex]
m_traindf['EmbarkedCat'] = [0 if e == 'C' else 1 if e == 'Q' else 2 for e in m_traindf.Embarked]
m_traindf['AgeBin'] = m_traindf.Age // 10
m_traindf['FamilySize'] = m_traindf.SibSp + m_traindf.Parch + 1
m_traindf['IsAlone'] = [1 if fs == 1 else 0 for fs in m_traindf.FamilySize]
m_traindf['FareBin'] = m_traindf.Fare // 30
m_traindf['HasCabin'] = [0 if type(c) is float else 1 for c in m_traindf.Cabin]
#Get the numbers alone from ticket
m_traindf['TicketBin'] = [sum([int(w) for w in tkt.split() if w.isdigit()]) for tkt in m_traindf.Ticket]
m_traindf.TicketBin = m_traindf.TicketBin // 14313 #create bins by dividing by the 25% value
m_traindf['Title'] = m_traindf.Name.apply(lambda v: sum([title_list.index(title) if title in v else 0 for title in title_list]))
m_traindf.info()
# age - replace nulls with random values between +- Standard deviations with training data's mean and std
age_nulls = len(m_testdf[m_testdf.Age.isnull()])
age_vals = np.random.randint(age_mean - age_std, age_mean + age_std, age_nulls)
m_testdf.loc[m_testdf.Age.isnull(), 'Age'] = age_vals

#Fill the null fare values with training mean and std
fare_mean = m_traindf.Fare.mean()
fare_std = m_traindf.Fare.std()
fare_nulls = len(m_testdf[m_testdf.Fare.isnull()])
fare_vals = np.random.randint(fare_mean - fare_std, fare_mean + fare_std, fare_nulls)
m_testdf.loc[m_testdf.Fare.isnull(), 'Fare'] = fare_vals

# Embarked - replace nulls with oft-repeating values from training data
m_testdf.loc[m_testdf.Embarked.isnull(), 'Embarked'] = embarked_mode

# New features
m_testdf['SexCat'] = [0 if s == 'male' else 1 for s in m_testdf.Sex]
m_testdf['EmbarkedCat'] = [0 if e == 'C' else 1 if e == 'Q' else 2 for e in m_testdf.Embarked]
m_testdf['AgeBin'] = m_testdf.Age // 10
m_testdf['FamilySize'] = m_testdf.SibSp + m_testdf.Parch + 1
m_testdf['IsAlone'] = [1 if fs == 1 else 0 for fs in m_testdf.FamilySize]
m_testdf['FareBin'] = m_testdf.Fare // 30
m_testdf['HasCabin'] = [0 if type(c) is float else 1 for c in m_testdf.Cabin]
#Get the numbers alone from ticket
m_testdf['TicketBin'] = [sum([int(w) for w in tkt.split() if w.isdigit()]) for tkt in m_testdf.Ticket]
m_testdf.TicketBin = m_testdf.TicketBin // 14313 #create bins by dividing by the 25% value of the training data
m_testdf['Title'] = m_testdf.Name.apply(lambda v: sum([title_list.index(title) if title in v else 0 for title in title_list]))
m_testdf.info()
f, axes = plt.subplots(10, 2, figsize=(10,40))
sns.countplot(x='Pclass', data=m_traindf, ax=axes[0, 0])
sns.countplot(x='Pclass', data=m_testdf, ax=axes[0, 1])

sns.countplot(x='Sex', data=m_traindf, ax=axes[1, 0])
sns.countplot(x='Sex', data=m_testdf, ax=axes[1, 1])

sns.countplot(x='AgeBin', data=m_traindf, ax=axes[2, 0])
sns.countplot(x='AgeBin', data=m_testdf, ax=axes[2, 1])

sns.countplot(x='FamilySize', data=m_traindf, ax=axes[3, 0])
sns.countplot(x='FamilySize', data=m_testdf, ax=axes[3, 1])

sns.countplot(x='IsAlone', data=m_traindf, ax=axes[4, 0])
sns.countplot(x='IsAlone', data=m_testdf, ax=axes[4, 1])

sns.countplot(x='FareBin', data=m_traindf, ax=axes[5, 0])
sns.countplot(x='FareBin', data=m_testdf, ax=axes[5, 1])

sns.countplot(x='Embarked', data=m_traindf, ax=axes[6, 0])
sns.countplot(x='Embarked', data=m_testdf, ax=axes[6, 1])

sns.countplot(x='HasCabin', data=m_traindf, ax=axes[7, 0])
sns.countplot(x='HasCabin', data=m_testdf, ax=axes[7, 1])

sns.countplot(x='TicketBin', data=m_traindf, ax=axes[8, 0])
sns.countplot(x='TicketBin', data=m_testdf, ax=axes[8, 1])

sns.countplot(x='Title', data=m_traindf, ax=axes[9, 0])
sns.countplot(x='Title', data=m_testdf, ax=axes[9, 1])

plt.show()
