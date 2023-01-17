import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

def age_convert(li):	
	age_in_weeks = []
	for i in li :
		j = str(i).strip().split(" ")
		try :
			if j[1][0].lower() == 'w' :
				age_in_weeks.append(int(j[0]))
			elif j[1][0].lower() == 'd' :
				age_in_weeks.append(int(j[0])/7.0)
			elif j[1][0].lower() == 'm' :
				age_in_weeks.append(int(j[0])*4)
			else :
				age_in_weeks.append(int(j[0])*48)
		except :
			age_in_weeks.append(None)

	maximum_age = max(age_in_weeks) 
	age_in_weeks_norm = []
	for i in age_in_weeks :
		try :
			val = float(i)/maximum_age
			age_in_weeks_norm.append(val)
		except :
			age_in_weeks_norm.append(None)

	return age_in_weeks_norm

train=pd.read_csv('../input/train.csv', parse_dates = ['DateTime'])
test=pd.read_csv('../input/test.csv', parse_dates = ['DateTime'])

labeller = preprocessing.LabelEncoder()
outcome = labeller.fit_transform(train.OutcomeType)

animal_type = pd.get_dummies(train.AnimalType)
sex = pd.get_dummies(train.SexuponOutcome)
breed = pd.get_dummies(train.Breed)
colour = pd.get_dummies(train.Color)
hour = train.DateTime.dt.hour
hour = pd.get_dummies(hour)
age = age_convert(train['AgeuponOutcome'].tolist())
age = pd.DataFrame(age)

train_data = pd.concat([animal_type,sex,breed,colour,hour,age], axis=1)
train_data['outcome'] = outcome
 
animal_type = pd.get_dummies(test.AnimalType)
sex = pd.get_dummies(test.SexuponOutcome)
breed = pd.get_dummies(test.Breed)
colour = pd.get_dummies(test.Color)
hour = test.DateTime.dt.hour
hour = pd.get_dummies(hour)
age = age_convert(test['AgeuponOutcome'].tolist())
age = pd.DataFrame(age)

test_data = pd.concat([animal_type,sex,breed,colour,hour,age], axis=1)

features = ['Cat','Dog','Spayed Female','Neutered Male','Intact Female','Unknown','Intact Male']

#EVALUATION
# training, validation = train_test_split(train_data, train_size=.70)
# model = RandomForestClassifier(n_estimators = 50, n_jobs = -1)
# model.fit(training[features],training['outcome'])
# predicted = model.predict_proba(validation[features])
# print log_loss(validation['outcome'], predicted)

#SUBMISSION
model = RandomForestClassifier(n_estimators = 50, n_jobs = -1)
model.fit(train_data[features], train_data['outcome'])
predicted = model.predict_proba(test_data[features])
result=pd.DataFrame(predicted, columns = labeller.classes_)
result.index += 1
result.to_csv('RF_Result.csv', index = True, index_label = 'Id')