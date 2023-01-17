import numpy as np

import pandas as pd

import re

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

from pylab import mpl  

mpl.rcParams['font.sans-serif'] = ['SimHei']  # Specify the font in the image
print('Reading Data...')

train_data = pd.read_csv("../input/titanic/train.csv", dtype={"Age": np.float64}, )

test_data = pd.read_csv("../input/titanic/train.csv", dtype={"Age": np.float64}, )
train_data.head() #dataframe格式
train_data.info()
train_data.describe()
%matplotlib inline 

import matplotlib.pyplot as plt

data_train = train_data



fig = plt.figure()

fig.set(alpha=0.2)  # Set chart color alpha parameter



plt.subplot2grid((3,7),(0,0))             # Separate several small pictures in one big picture

data_train.Survived.value_counts().plot(kind='bar')# Histogram

plt.title("Rescue situation ") # title

plt.ylabel("people")  



plt.subplot2grid((3,7),(0,3))

data_train.Pclass.value_counts().plot(kind="bar")

plt.ylabel("people")

plt.title("class distribution")



plt.subplot2grid((3,7),(0,6))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel("age")                         # Set the ordinate name

plt.grid(b=True, which='major', axis='y') 

plt.title("Rescue by age ")





plt.subplot2grid((3,7),(2,0), colspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel("age")# plots an axis lable

plt.ylabel("density") 

plt.title("Passenger age distribution by class")

#plt.legend(('First Class', '2 First Class', '3 First Class'),loc='best') # sets our legend for our graph.





plt.subplot2grid((3,7),(2,6))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title("Number of people on board at each boarding port")

plt.ylabel("people")  

plt.show()
fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df=pd.DataFrame({'Not rescued':Survived_0,'Rescued':Survived_1})

df.plot(kind='bar', stacked=True)

plt.title("Rescue status by passenger class")

plt.xlabel("Passenger class") 

plt.ylabel("Number of people") 

plt.show()
fig = plt.figure()

fig.set(alpha=0.2)  # 设定图表颜色alpha参数



Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df=pd.DataFrame({u'man':Survived_m, u'woman':Survived_f})

df.plot(kind='bar', stacked=True)

plt.title(u"Rescue by gender")

plt.xlabel(u"gender") 

plt.ylabel(u"people")

plt.show()
#然后我们再来看看各种舱级别情况下各性别的获救情况

fig=plt.figure()

fig.set(alpha=0.65) # 设置图像透明度，无所谓

plt.title(u"Rescue according to cabin class and gender")



ax1=fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().sort_index().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticks([0,1])

ax1.set_xticklabels([u"0", u"1"], rotation=0)

ax1.legend([u"Female/Premium"], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().sort_index().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels([u"0", u"1"], rotation=0)

plt.legend([u"Female/Lower Class"], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().sort_index().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels([u"0", u"1"], rotation=0)

plt.legend([u"Male / Premium Class"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().sort_index().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels([u"0", u"1"], rotation=0)

plt.legend([u"Male / Lower Class"], loc='best')



plt.show()
# Functions



def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
print('Cleaning Data...')



# When cleaning the data, the test set is also put into it, so that the format of the test set is the same as the training set when predicting the results later.

combined2 = pd.concat([train_data, test_data], axis=0)

combined2.Embarked.fillna('S', inplace=True)

combined2.Fare.fillna(np.median(combined2.Fare[combined2.Fare.notnull()]), inplace=True)

combined2['Title'] = combined2["Name"].apply(get_title)



# In order to facilitate the establishment of the model, we will map the str type data to the int type one by one.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona":10, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, "Ms": 2}

combined2["TitleCat"] = combined2.loc[:,'Title'].map(title_mapping)

combined2['CabinCat'] = pd.Categorical(combined2.Cabin.fillna('0').apply(lambda x: x[0])).codes

combined2.Cabin.fillna('0', inplace=True)

combined2['EmbarkedCat'] = pd.Categorical(combined2.Embarked).codes

combined2.drop(['Ticket'], axis=1, inplace=True)

print('Consolidating Data...')



full_data = pd.concat([combined2.drop(['Survived'],axis=1),

                       pd.get_dummies(combined2.Sex, prefix='Sex'),

                       combined2.Survived], axis=1)


print('Generating Features...')



full_data['FamilySize'] = full_data["SibSp"] + full_data["Parch"]

full_data['NameLength'] = full_data.Name.apply(lambda x: len(x))



import operator

family_id_mapping = {}

def get_family_id(row):

    last_name = row["Name"].split(",")[0]

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]

family_ids = full_data.apply(get_family_id, axis=1)

# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.

family_ids[full_data["FamilySize"] < 3] = -1

full_data["FamilyId"] = family_ids



#### Person Label

child_age = 14

def get_person(passenger):

    age, sex = passenger

    if (age < child_age):

        return 'child'

    elif (sex == 'female'):

        return 'female_adult'

    else:

        return 'male_adult'

full_data = pd.concat([full_data, pd.DataFrame(full_data[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)

dummies = pd.get_dummies(full_data['person'])

full_data = pd.concat([full_data,dummies],axis=1)



def process_surname(nm):

    return nm.split(',')[0].lower()

full_data['surname'] = full_data['Name'].apply(process_surname)



#### Persihing Females

perishing_female_surnames = list(set(full_data[(full_data.female_adult == 1.0) &

                                     (full_data.Survived == 0.0) &

                                     ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))

def perishing_mother_wife(passenger): 

    surname, Pclass, person = passenger

    return 1.0 if (surname in perishing_female_surnames) else 0.0

full_data['perishing_mother_wife'] = full_data[['surname', 'Pclass', 'person']].apply(perishing_mother_wife, axis=1)



#### Survivng Males

surviving_male_surnames = list(set(full_data[(full_data.male_adult == 1.0) &

                                     (full_data.Survived == 1.0) &

                                     ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))

def surviving_father_husband(passenger): 

    surname, Pclass, person = passenger

    return 1.0 if (surname in surviving_male_surnames) else 0.0

full_data['surviving_father_husband'] = full_data[['surname', 'Pclass', 'person']].apply(surviving_father_husband, axis=1)



classers = ['Fare','Parch','Pclass','SibSp','TitleCat','CabinCat','Sex_female','Sex_male', 'EmbarkedCat', 'FamilySize', 'NameLength', 'FamilyId']

age_et = ExtraTreesRegressor(n_estimators=200)

X_train = full_data.loc[full_data.Age.notnull(),classers]

Y_train = full_data.loc[full_data.Age.notnull(),['Age']]

X_test = full_data.loc[full_data.Age.isnull(),classers]

age_et.fit(X_train,np.ravel(Y_train))

age_preds = age_et.predict(X_test)

full_data.loc[full_data.Age.isnull(),['Age']] = age_preds
print('Building Model...')



#### Model Build - Random Forest (Categorical Features)

model_dummys = ['Age','male_adult', 'female_adult', 'child','perishing_mother_wife','surviving_father_husband','Fare','Parch','Pclass','SibSp','TitleCat','CabinCat','Sex_female','Sex_male', 'EmbarkedCat', 'FamilySize', 'NameLength', 'FamilyId']

model_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0:0.745,1:0.255})

X_data = full_data.iloc[:891,:]  # Remove the test set part from the data set

X_train = X_data.loc[:,model_dummys]

Y_data = full_data.iloc[:891,:]

Y_train = Y_data.loc[:,['Survived']]

X_t_data = full_data.iloc[891:,:]

X_test = X_t_data.loc[:,model_dummys]

model_rf.fit(X_train, np.ravel(Y_train))
print('Generating Predictions...')



model_results = model_rf.predict(X_test)

print('Processing Submission File...')





model_results = [str(int(x)) for x in model_results]

submission = pd.DataFrame()

submission['PassengerId'] = X_t_data.PassengerId

submission['Survived'] = model_results

submission.set_index(['PassengerId'],inplace=True, drop=True)

submission.head(3)

submission.to_csv('titanic_submission.csv')



print('Done.')
