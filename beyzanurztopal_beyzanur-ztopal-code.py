#necessary imports

!pip install pandas plotnine

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')

from plotnine import *

%matplotlib inline
#loading train & test data



train = pd.read_csv('../input/shelter-animal-outcomes/train.csv.gz')

test = pd.read_csv('../input/shelter-animal-outcomes/test.csv.gz')



train.head(3)
test.head(3)
train.describe()

print("OutcomeType:\n",train.OutcomeType.unique())

print("\nAnimalType:\n",train.AnimalType.unique())

print("\nSexuponOutcome:\n",train.SexuponOutcome.unique())

print("\nAgeuponOutcome:",train.AgeuponOutcome.unique())
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print("Null percentage of Name:",train["Name"].isna().sum()/train["Name"].shape[0])

print("Null percentage of OutcomeSubtype:",train["OutcomeSubtype"].isna().sum()/train["OutcomeSubtype"].shape[0])
#to work easier rename the column names



train.rename(columns={'AnimalID' :'ID','Name':'name', 'DateTime':'date','OutcomeType':'target','OutcomeSubtype':'subtype','AnimalType':'animal','SexuponOutcome':'sex','AgeuponOutcome':'age','Breed':'breed','Color':'color'}, inplace=True) 

test.rename(columns={'Name':'name', 'DateTime':'date','AnimalType':'animal','SexuponOutcome':'sex','AgeuponOutcome':'age','Breed':'breed','Color':'color'}, inplace=True) 
plt.figure(figsize=(7,5))

sns.countplot(x=train.target, order=['Adoption','Transfer','Return_to_owner','Euthanasia','Died'])
ggplot(train, aes(x='animal', fill='target')) + geom_bar(colour='black') + labs(y='count')
plt.figure(figsize=(10,6))

sns.countplot(x='sex',data=train,hue='target')
train_dog = train.loc[train.animal == 'Dog']

train_cat = train.loc[train.animal == 'Cat']



test_dog = test.loc[test.animal == 'Dog']

test_cat = test.loc[test.animal == 'Cat']
print("\nAgeuponOutcome:",train.age.unique())
#since age column contains different time units, convert them to month

def age_month(x):

  x = str(x)

  if x == 'nan':

    return 0

  age = int(x.split()[0])

  if x.find('year') > -1:

    return age * 12

  if x.find('month') > -1:

    return age

  if x.find('week') > -1:

    return age // 4

  if x.find('day') > -1:

    return age // 30

  else:

    return 0 



train_dog['age'] = train_dog.age.apply(age_month)

train_cat['age'] = train_cat.age.apply(age_month)

test_dog['age'] = test_dog.age.apply(age_month)

test_cat['age'] = test_cat.age.apply(age_month)





#age in less than 1 month

def age_less1month(x):

  if x >= 0 and x<= 1:

    return 'yes'

  else:

    return 'no' 



#age in between 2 years and 10 years

def age_2_10(x):

  if x >= 24 and x<= 120:

    return 'yes'

  else:

    return 'no' 



#age in greater than 10 years

def age_10plus(x):

  if x > 120:

    return 'yes'

  else:

    return 'no' 



train_dog['less_1month'] = train_dog.age

train_dog['less_1month'] = train_dog.less_1month.apply(age_less1month)

train_cat['less_1month'] = train_cat.age

train_cat['less_1month'] = train_cat.less_1month.apply(age_less1month)



test_cat['less_1month'] = test_cat.age

test_cat['less_1month'] = test_cat.less_1month.apply(age_less1month)

test_dog['less_1month'] = test_dog.age

test_dog['less_1month'] = test_dog.less_1month.apply(age_less1month)





train_cat['age_2_10'] = train_cat.age

train_cat['age_2_10'] = train_cat.age_2_10.apply(age_2_10)

train_dog['age_2_10'] = train_dog.age

train_dog['age_2_10'] = train_dog.age_2_10.apply(age_2_10)



test_cat['age_2_10'] = test_cat.age

test_cat['age_2_10'] = test_cat.age_2_10.apply(age_2_10)

test_dog['age_2_10'] = test_dog.age

test_dog['age_2_10'] = test_dog.age_2_10.apply(age_2_10)



train_cat['age_10plus'] = train_cat.age

train_cat['age_10plus'] = train_cat.age_10plus.apply(age_10plus)

train_dog['age_10plus'] = train_dog.age

train_dog['age_10plus'] = train_dog.age_10plus.apply(age_10plus)



test_cat['age_10plus'] = test_cat.age

test_cat['age_10plus'] = test_cat.age_10plus.apply(age_10plus)

test_dog['age_10plus'] = test_dog.age

test_dog['age_10plus'] = test_dog.age_10plus.apply(age_10plus)





#categorize dog ages wrt dog age scale

def age_dog(x):

  if x == 0:

    return 'unknown'

  elif (x>0 and x<7):

    return 'new born'

  elif (x>6 and x<25):

    return 'junior'

  elif (x>24 and x<37):

    return 'prime'

  elif (x>36 and x<73):

    return 'mature'

  elif (x>72 and x<121):

    return 'senior'

  elif x>120:

    return 'geriatric'





#categorize cat ages wrt cat age scale

def age_cat(x):

  if x == 0:

    return 'unknown'

  elif (x>0 and x<7):

    return 'new born'

  elif (x>6 and x<37):

    return 'junior'

  elif (x>36 and x<85):

    return 'prime'

  elif (x>84 and x<133):

    return 'mature'

  elif (x>132 and x<181):

    return 'senior'

  elif x>180:

    return 'geriatric'



train_dog['age'] = train_dog.age.apply(age_dog)

train_cat['age'] = train_cat.age.apply(age_cat)



test_dog['age'] = test_dog.age.apply(age_dog)

test_cat['age'] = test_cat.age.apply(age_cat)





sns.countplot(data=train_dog, x=train_dog.target, hue=train_dog.age,order=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
sns.countplot(data=train_cat, x=train_cat.target, hue=train_cat.age,order=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
#assign categorical scales to the numbers

age_list = {'new born':1, 'junior':2, 'prime':3, 'mature':4, 'senior':5, 'geriatric':6, 'unknown':7 }



train_dog['age'] = train_dog.age.map(age_list)

train_cat['age'] = train_cat.age.map(age_list)



test_dog['age'] = test_dog.age.map(age_list)

test_cat['age'] = test_cat.age.map(age_list)



train_cat.head(1)
train_dog.head(1)
train.breed.value_counts()
#agrressive dog breeds (appliying only dogs dataset)

def aggressive(breed):

		if breed.find("Pit Bull") != -1:

			return 1

		elif breed.find("Rottweiler") != -1:

			return 2

		elif breed.find("Husky") != -1:

			return 3

		elif breed.find("Shepherd") != -1:

			return 4

		elif breed.find("Malamute") != -1:

			return 5

		elif breed.find("Doberman") != -1:

			return 6

		elif breed.find("Chow") != -1:

			return 7

		elif breed.find("Dane") != -1:

			return 8

		elif breed.find("Boxer") != -1:

			return 9

		elif breed.find("Akita") != -1:

			return 10

		else:

			return 11

    

train_dog["aggresive"] = train_dog["breed"].apply(aggressive)

test_dog["aggresive"] = test_dog["breed"].apply(aggressive)





#allergic dog breeds (appliying only dogs dataset)

def allergic(breed):

		if breed.find("Akita") != -1:

			return 1

		elif breed.find("Malamute") != -1:

			return 2

		elif breed.find("Eskimo") != -1:

			return 3

		elif breed.find("Corgi") != -1:

			return 4

		elif breed.find("Chow") != -1:

			return 5

		elif breed.find("Shepherd") != -1:

			return 6

		elif breed.find("Pyrenees") != -1:

			return 7

		elif breed.find("Labrador") != -1:

			return 8

		elif breed.find("Retriever") != -1:

			return 9

		elif breed.find("Husky") != -1:

			return 10

		else:

			return 11

    

train_dog["allergic"] = train_dog["breed"].apply(allergic)

test_dog["allergic"] = test_dog["breed"].apply(allergic)





#hair group classification

def hair_group(breed):

		if breed.find("Shorthair") != -1:

			return 0

		elif breed.find("Longhair") != -1:

			return 1

		else:

			return 2



train_cat["hairgroup"] = train_cat["breed"].apply(hair_group)

train_dog["hairgroup"] = train_dog["breed"].apply(hair_group)



test_cat["hairgroup"] = test_cat["breed"].apply(hair_group)

test_dog["hairgroup"] = test_dog["breed"].apply(hair_group)







#breed group classification - mix breed / or not

def breed_group(breed_input):

		breed = str(breed_input)

		if (' ' in breed) == False:

			br =  breed 

		else:

			breed_list = breed.split()

			try:

				br = breed_list[2] 

			except:

				br = breed_list[1] 

		if (br == "Mix"):

			return 0

		else:

			return 1

		return 1



train_dog["mix"] = train_dog["breed"].apply(breed_group)

train_cat["mix"] = train_cat["breed"].apply(breed_group)



test_cat["mix"] = test_cat["breed"].apply(breed_group)

test_dog["mix"] = test_dog["breed"].apply(breed_group)







#top dog and cat breeds 

dog_max = ['Pit Bull','Chihuahua Shorthair','Labrador Retriever','German Shepherd','Australian Cattle Dog','Dachshund','Boxer','Miniature Poodle','Border Collie','Australian Shepherd','Rat Terrier','Siberian Husky','Yorkshire Terrier','Catahoula','Jack Russell Terrier','Miniature Schnauzer','Shih Tzu','Rottweiler','Chihuahua Longhair','Beagle','American Bulldog','Cairn Terrier','American Staffordshire Terrier','Staffordshire','Australian Kelpie','Great Pyrenees']

cat_max = ['Domestic Shorthair','Domestic Medium Hair','Domestic Longhair','Siamese']





def dog_breed(x):

		x = str(x)

		if x in dog_max:

			return x

		else:

			return 'other'            



train_dog['breed'] = train_dog.breed.apply(dog_breed)

test_dog['breed'] = test_dog.breed.apply(dog_breed)





def cat_breed(x):

		x = str(x)

		if x in cat_max:

			return x

		else:

			return 'other'



train_cat['breed'] = train_cat.breed.apply(cat_breed)

test_cat['breed'] = test_cat.breed.apply(cat_breed)



train_cat.head(1)
train_dog.head(1)
train.color.value_counts()
#extract color in terms or solid or not

def extract_color(df):

    df['IsSolidColor'] = 1

    mixed_entries = df['color'].str.contains(r'\/')

    df.loc[mixed_entries, 'IsSolidColor'] = 0

    df['color'] = df['color'].str.replace('\/.*', '')

    return df



train_cat = extract_color(train_cat)

train_dog = extract_color(train_dog)



test_cat = extract_color(test_cat)

test_dog = extract_color(test_dog)





#label encoding for cat colors

le_color = preprocessing.LabelEncoder()

train_cat.color = le_color.fit_transform(train_cat.color)

test_cat.color = le_color.fit_transform(test_cat.color)



#label encoding for cat colors

le_color = preprocessing.LabelEncoder()

train_dog.color = le_color.fit_transform(train_dog.color)

test_dog.color = le_color.fit_transform(test_dog.color)



train_cat.head(1) 
train_dog.head(1)
#filling null values

train_dog.sex.fillna('Unknown', inplace=True)



#categorize for infertility

def intact_group(sex):

		try:

			intact_type = sex.split()

		except:

			return 0

		if intact_type[0] == "Neutered" or intact_type[0] ==  "Spayed":		

			return 1

		elif intact_type[0] == "Intact":

			return 2

		else:

			return 0



train_cat["virginity"] = train_cat["sex"].apply(intact_group)

train_dog["virginity"] = train_dog["sex"].apply(intact_group)



test_cat["virginity"] = test_cat["sex"].apply(intact_group)

test_dog["virginity"] = test_dog["sex"].apply(intact_group)





#categorize for male/female

def sex_group(sexs):

		try:

			sex_type = sexs.split()

		except:

			return 0

		#categorize

		if sex_type[0] == "Unknown":

			return 0

		elif sex_type[1] == "Male":

			return 1

		elif sex_type[1] == "Female":

			return 2

		else:

			return 0

        

train_cat["sex_"] = train_cat["sex"].apply(sex_group)

train_dog["sex_"] = train_dog["sex"].apply(sex_group)



test_cat["sex_"] = test_cat["sex"].apply(sex_group)

test_dog["sex_"] = test_dog["sex"].apply(sex_group)



train_cat.head(1)
train_dog.head(1)
#categorize name: has name/or not

def name(x):

  x = str(x)

  if x == 'nan':

    return 0

  else:

    return 1



train_cat['name'] = train_cat.name.apply(name)

train_dog['name'] = train_dog.name.apply(name)



test_cat['name'] = test_cat.name.apply(name)

test_dog['name'] = test_dog.name.apply(name)



train_cat.head(1)
train_dog.head(1)
sns.countplot(x='name', data=train_cat, hue='target')
sns.countplot(x='name', data=train_dog, hue='target')
#converting target features to numbers

target_list = {'Adoption':1, 'Died':2, 'Euthanasia':3, 'Return_to_owner':4, 'Transfer':5}



train_dog['target'] = train_dog.target.map(target_list)

train_cat['target'] = train_cat.target.map(target_list)
#taking date units from datetime column for cats

train_cat.date = pd.to_datetime(train_cat.date)

train_cat["dayofweek"] = train_cat.date.dt.dayofweek

train_cat["month"] = train_cat.date.dt.month

train_cat["year"] = train_cat.date.dt.year

train_cat["hour"] = train_cat.date.dt.hour

train_cat["minute"] = train_cat.date.dt.minute



test_cat.date = pd.to_datetime(test_cat.date)

test_cat["dayofweek"] = test_cat.date.dt.dayofweek

test_cat["month"] = test_cat.date.dt.month

test_cat["year"] = test_cat.date.dt.year

test_cat["hour"] = test_cat.date.dt.hour

test_cat["minute"] = test_cat.date.dt.minute



#taking date units from datetime column for dogs

train_dog.date = pd.to_datetime(train_dog.date)

train_dog["dayofweek"] = train_dog.date.dt.dayofweek

train_dog["month"] = train_dog.date.dt.month

train_dog["year"] = train_dog.date.dt.year

train_dog["hour"] = train_dog.date.dt.hour

train_dog["minute"] = train_dog.date.dt.minute



test_dog.date = pd.to_datetime(test_dog.date)

test_dog["dayofweek"] = test_dog.date.dt.dayofweek

test_dog["month"] = test_dog.date.dt.month

test_dog["year"] = test_dog.date.dt.year

test_dog["hour"] = test_dog.date.dt.hour

test_dog["minute"] = test_dog.date.dt.minute



#mapping years to numbers

year_list = {2013 : 1, 2014 : 2, 2015 : 3, 2016:4}

train_cat['year'] = train_cat.year.map(year_list)

train_dog['year'] = train_dog.year.map(year_list)



test_cat['year'] = test_cat.year.map(year_list)

test_dog['year'] = test_dog.year.map(year_list)



train_cat.head(1)
train_dog.head(1)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

X_cat = train_cat.drop(['ID', 'date', 'target', 'subtype', 'animal','sex'], axis=1)

Y_cat = train_cat['target']



from sklearn.model_selection import train_test_split

X_cat = pd.get_dummies(X_cat)

X_train_cat, X_test_cat, Y_train_cat, Y_test_cat = train_test_split(X_cat, Y_cat, test_size = 0.33, random_state=0)





X_dog = train_dog.drop(['ID', 'date', 'target', 'subtype', 'animal','sex'], axis=1)

Y_dog = train_dog['target']



from sklearn.model_selection import train_test_split

X_dog = pd.get_dummies(X_dog)

X_train_dog, X_test_dog, Y_train_dog, Y_test_dog = train_test_split(X_dog, Y_dog, test_size = 0.33, random_state=0)





print("train cat shapes:",X_train_cat.shape, X_test_cat.shape, Y_train_cat.shape, Y_test_cat.shape)

print("train dog shapes:",X_train_dog.shape, X_test_dog.shape, Y_train_dog.shape, Y_test_dog.shape)
X_cat.head(3)
X_dog.head(3)
from sklearn.linear_model import LogisticRegression



clf_cat = LogisticRegression()

clf_cat.fit(X_train_cat, Y_train_cat)



y_pred_log_reg = clf_cat.predict(X_test_cat)



print(confusion_matrix(Y_test_cat, y_pred_log_reg))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_log_reg))





clf_dog = LogisticRegression()

clf_dog.fit(X_train_dog, Y_train_dog)



y_pred_log_reg = clf_dog.predict(X_test_dog)



print("\n",confusion_matrix(Y_test_dog, y_pred_log_reg))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_log_reg))
from sklearn.tree import DecisionTreeClassifier



clf_cat = DecisionTreeClassifier()

model = clf_cat.fit(X_train_cat, Y_train_cat)



y_pred_decision_tree = clf_cat.predict(X_test_cat)



print(confusion_matrix(Y_test_cat, y_pred_decision_tree))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_decision_tree))







clf_dog = DecisionTreeClassifier()

model = clf_dog.fit(X_train_dog, Y_train_dog)



y_pred_decision_tree = clf_dog.predict(X_test_dog)



print("\n",confusion_matrix(Y_test_dog, y_pred_decision_tree))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_decision_tree))
from sklearn.ensemble import RandomForestClassifier



forestOpt_cat = RandomForestClassifier(max_depth = 25, n_estimators = 300, min_samples_split = 2, min_samples_leaf = 1)

                                   

modelOpt_cat = forestOpt_cat.fit(X_train_cat, Y_train_cat)

y_pred_cat = modelOpt_cat.predict(X_test_cat)



print(confusion_matrix(Y_test_cat, y_pred_cat))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_cat))







forestOpt_dog = RandomForestClassifier(random_state = 1, max_depth = 15, n_estimators = 500, min_samples_split = 2, min_samples_leaf = 3)

                                   

modelOpt_dog = forestOpt_dog.fit(X_train_dog, Y_train_dog)

y_pred_dog = modelOpt_dog.predict(X_test_dog)



print("\n", confusion_matrix(Y_test_dog, y_pred_dog))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_dog))
from sklearn.ensemble import AdaBoostClassifier



clf_cat = AdaBoostClassifier(random_state=0, n_estimators=7, learning_rate=0.9)

clf_cat.fit(X_train_cat, Y_train_cat)



y_pred_adaboost_cat =clf_cat.predict(X_test_cat)



print(confusion_matrix(Y_test_cat, y_pred_adaboost_cat))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_adaboost_cat))







clf_dog = AdaBoostClassifier(random_state=0, n_estimators=7, learning_rate=0.9)

clf_dog.fit(X_train_dog, Y_train_dog)



y_pred_adaboost_dog =clf_dog.predict(X_test_dog)



print("\n",confusion_matrix(Y_test_dog, y_pred_adaboost_dog))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_adaboost_dog))
from sklearn.ensemble import GradientBoostingClassifier



clf_cat = GradientBoostingClassifier(random_state=0, n_estimators=10)

clf_cat.fit(X_train_cat, Y_train_cat)



y_pred_grad_cat = clf_cat.predict(X_test_cat)



print(confusion_matrix(Y_test_cat, y_pred_grad_cat))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_grad_cat))





clf_dog = GradientBoostingClassifier(random_state=0, n_estimators=10)

clf_dog.fit(X_train_dog, Y_train_dog)



y_pred_grad_dog = clf_dog.predict(X_test_dog)



print("\n",confusion_matrix(Y_test_dog, y_pred_grad_dog))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_grad_dog))
from xgboost import XGBClassifier



xg_cat = XGBClassifier(random_state=42)

model_xg_cat = xg_cat.fit(X_train_cat, Y_train_cat)



y_pred_xg_cat = xg_cat.predict(X_test_cat)





print(confusion_matrix(Y_test_cat, y_pred_xg_cat))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_xg_cat))





xg_dog = XGBClassifier(random_state=0)

model_xg_dog = xg_dog.fit(X_train_dog, Y_train_dog)



y_pred_dog = xg_dog.predict(X_test_dog)





print("\n",confusion_matrix(Y_test_dog, y_pred_dog))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_dog))
from sklearn.svm import SVC



clf_cat = SVC(random_state=0)

clf_cat.fit(X_train_cat, Y_train_cat)



y_pred_svm_cat = clf_cat.predict(X_test_cat)



print(confusion_matrix(Y_test_cat, y_pred_svm_cat))

print("\nAccuracy Cat: ", accuracy_score(Y_test_cat, y_pred_svm_cat))





clf_dog = SVC(random_state=0)

clf_dog.fit(X_train_dog, Y_train_dog)



y_pred_svm_dog = clf_dog.predict(X_test_dog)



print("\n",confusion_matrix(Y_test_dog, y_pred_svm_dog))

print("\nAccuracy Dog: ", accuracy_score(Y_test_dog, y_pred_svm_dog))
test_cat.head(3)
#taking ID columns for submission before dropping

sub_cat = pd.DataFrame()

sub_cat['ID'] = test_cat.ID



test_cat = test_cat.drop(['ID','date','animal','sex'], axis=1)

test_cat = pd.get_dummies(test_cat)



#taking predictions for xgboost

y_pred_cat = model_xg_cat.predict_proba(test_cat)



#adding predictions to datasests which contains ID values

sub_cat['Adoption'], sub_cat['Died'], sub_cat['Euthanasia'], sub_cat['Return_to_owner'], sub_cat['Transfer'] = y_pred_cat[:,0], y_pred_cat[:,1], y_pred_cat[:,2], y_pred_cat[:,3], y_pred_cat[:,4]

sub_cat.head()
#taking ID columns for submission before dropping

sub_dog = pd.DataFrame()

sub_dog['ID'] = test_dog.ID



#taking predictions for xgboost

test_dog = test_dog.drop(['ID','date','animal','sex'], axis=1)

test_dog = pd.get_dummies(test_dog)



#adding predictions to datasests which contains ID values

y_pred_dog = model_xg_dog.predict_proba(test_dog)

sub_dog['Adoption'], sub_dog['Died'], sub_dog['Euthanasia'], sub_dog['Return_to_owner'], sub_dog['Transfer'] = y_pred_dog[:,0], y_pred_dog[:,1], y_pred_dog[:,2], y_pred_dog[:,3], y_pred_dog[:,4]

sub_dog.head()
sub = pd.concat([sub_cat,sub_dog])

sub = sub.sort_values('ID')

sub

# sub.to_csv("submission_last.csv", index=False)