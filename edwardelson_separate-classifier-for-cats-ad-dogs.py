import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

#import data
shelter_train = pd.read_csv("../input/train.csv")
shelter_train_outcome = shelter_train["OutcomeType"]
shelter_test = pd.read_csv("../input/test.csv")
#########################################################################################################
#### Create separate dog and cat database
#########################################################################################################

dog_train = shelter_train[shelter_train["AnimalType"]=="Dog"]
dog_train = dog_train.reset_index()
dog_train.drop("AnimalType", axis=1, inplace=True)
dog_train.drop("index", axis=1, inplace=True)
dog_train_outcome = dog_train["OutcomeType"]
dog_train.drop("OutcomeType", axis=1, inplace=True)
dog_test = shelter_test[shelter_test["AnimalType"]=="Dog"]
dog_test = dog_test.reset_index()
dog_test.drop("AnimalType", axis=1, inplace=True)
dog_test.drop("index", axis=1, inplace=True)

cat_train = shelter_train[shelter_train["AnimalType"]=="Cat"]
cat_train = cat_train.reset_index()
cat_train.drop("AnimalType", axis=1, inplace=True)
cat_train.drop("index", axis=1, inplace=True)
cat_train_outcome = cat_train["OutcomeType"]
cat_train.drop("OutcomeType", axis=1, inplace=True)
cat_test = shelter_test[shelter_test["AnimalType"]=="Cat"]
cat_test = cat_test.reset_index()
cat_test.drop("AnimalType", axis=1, inplace=True)
cat_test.drop("index", axis=1, inplace=True)

#keep ID
dog_test_ID = dog_test["ID"].as_matrix()
dog_test_ID = np.array([dog_test_ID])
dog_test_ID = dog_test_ID.T
dog_test.drop("ID", axis=1, inplace=True)
cat_test_ID = cat_test["ID"].as_matrix()
cat_test_ID = np.array([cat_test_ID])
cat_test_ID = cat_test_ID.T
cat_test.drop("ID", axis=1, inplace=True)


#########################################################################################################
#### Hypothesis Plotting
#########################################################################################################
plot = True

if plot == True: 
	#plot correlation between each feature and OutcomeType
	axis = {
			"AgeuponOutcome",
			#"Breed",
			#"Color",
			#"SexuponOutcome",
			}
	count = 0
	for ax in axis:
		x = dog_train[ax]
		y = dog_train_outcome
		plt.figure(count)
		plt.title("Dog" + ax)
		sns.countplot(x=x, hue=y)
		count = count + 1

	for ax in axis:
		x = cat_train[ax]
		y = cat_train_outcome
		plt.figure(count)
		plt.title("Cat" + ax)
		sns.countplot(x=x, hue=y)
		count = count + 1

	x = shelter_train["AnimalType"]
	y = shelter_train_outcome
	plt.figure(count)
	plt.title("AnimalType")
	sns.countplot(x=x, hue=x)

	#plt.show()


#########################################################################################################
####Pre-Processing, mostly dropping and altering data
#########################################################################################################
def pre_processing(shelter_train, shelter_test, animal_type):
	#########################################################
	#### Drop ID, OutcomeType, OutcomeSubType
	#########################################################
	#ID: Drop
	shelter_train.drop("AnimalID", axis=1, inplace=True)
	# shelter_test.drop("ID", axis=1, inplace=True) -> Keep this for tagging later

	#OutcomeSubType: Drop
	shelter_train.drop("OutcomeSubtype", axis=1, inplace=True)

	#########################################################
	#### Fetch Year, Month, Day, Hour, Minute from DateTime
	#########################################################
	time_train = pd.to_datetime(shelter_train["DateTime"])
	time_test = pd.to_datetime(shelter_test["DateTime"])

	shelter_train["Year"] = time_train.dt.year
	shelter_test["Year"] = time_test.dt.year
	shelter_train["Month"] = time_train.dt.month
	shelter_test["Month"] = time_test.dt.month
	shelter_test["Day"] = time_test.dt.day
	shelter_train["Day"] = time_train.dt.day
	shelter_test["Hour"] = time_test.dt.hour
	shelter_train["Hour"] = time_train.dt.hour
	shelter_test["Minute"] = time_test.dt.minute
	shelter_train["Minute"] = time_train.dt.minute

	#drop DateTime
	shelter_train.drop("DateTime", axis=1, inplace=True)
	shelter_test.drop("DateTime", axis=1, inplace=True)

	#########################################################
	#### Convert SexuponOutcome to Virginity and Sex
	#########################################################

	# Virginity
	# fill in missing data with mode
	shelter_train["SexuponOutcome"].fillna("Spayed Female", inplace=True)
	shelter_test["SexuponOutcome"].fillna("Spayed Female", inplace=True)

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

	shelter_train["Virginity"] = shelter_train["SexuponOutcome"].apply(intact_group)
	shelter_test["Virginity"] = shelter_test["SexuponOutcome"].apply(intact_group)

	# Sex
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

	shelter_train["Sex"] = shelter_train["SexuponOutcome"].apply(sex_group)
	shelter_test["Sex"] = shelter_test["SexuponOutcome"].apply(sex_group)

	shelter_train.drop("SexuponOutcome", axis=1, inplace=True)
	shelter_test.drop("SexuponOutcome", axis=1, inplace=True)

	#########################################################
	#### Convert Name to has_name
	#########################################################
	def check_has_name(name):
		if type(name) is str:
			return 1
		else: #if name is NaN
			return 0

	#has_name: create new column for has_name
	shelter_train["has_name"] = shelter_train["Name"].apply(check_has_name)
	shelter_test["has_name"] = shelter_test["Name"].apply(check_has_name)
	#drop the name column
	shelter_train.drop("Name", axis=1, inplace=True)
	shelter_test.drop("Name", axis=1, inplace=True)

	#########################################################
	#### Convert Age to Age in days
	#########################################################
	#Age: fill missing data in Age, assume NaN = 1 year (modus value)
	shelter_train["AgeuponOutcome"].fillna("1 month", inplace=True)
	shelter_test["AgeuponOutcome"].fillna("1 month", inplace=True)
	#convert age to age group, author arbitrarily decides
	def age_group(age):
		try:
			age_list = age.split() #"2 days" -> ["2", "days"]
		except:
			return None
		ages = int(age_list[0])
		if(age_list[1].find("s")): #weeks->week, days->day
			age_list[1] = age_list[1].replace("s","")
		if age_list[1] == "day":
			return ages
		elif (age_list[1] == "week"):
			return ages*7
		elif (age_list[1] == "month"):
			return ages*30
		elif (age_list[1] == "year"):
			return ages*365

	#replace AgeuponOutcome with Age group
	shelter_train["AgeuponOutcome"] = shelter_train["AgeuponOutcome"].apply(age_group)
	shelter_test["AgeuponOutcome"] = shelter_test["AgeuponOutcome"].apply(age_group)
	#########################################################
	#### Convert Breed to Hair, Aggressiveness, Weight, BreedType
	#########################################################

	#hair group (for cats)
	def hair_group(breed):
		if breed.find("Shorthair") != -1:
			return 0
		elif breed.find("Longhair") != -1:
			return 1
		else:
			return 2

	shelter_train["Hairgroup"] = shelter_train["Breed"].apply(hair_group)
	shelter_test["Hairgroup"] = shelter_test["Breed"].apply(hair_group)

	#aggressiveness based on breed type. Most dangerous breeds:
	# Pitbull (55-65 lbs), Rottweiler (100-130 lbs), Husky-type (66 lbs), German
	# Shepherd (100 lbs) , Alaskan Malamute (100 lbs), Doberman pinscher (65-90lbs),
	# chow chow (70 lbs), Great Danes (200pounds), Boxer (70 lbs), Akita (45 kg)
	def aggressive(breed):
		if breed.find("Pit Bull") != -1:
			return 1
		elif breed.find("Rottweiler") != -1:
			return 2#1
		elif breed.find("Husky") != -1:
			return 3#1
		elif breed.find("Shepherd") != -1:
			return 4#1
		elif breed.find("Malamute") != -1:
			return 5#1
		elif breed.find("Doberman") != -1:
			return 6#1
		elif breed.find("Chow") != -1:
			return 7#1
		elif breed.find("Dane") != -1:
			return 8#1
		elif breed.find("Boxer") != -1:
			return 9#1
		elif breed.find("Akita") != -1:
			return 10#1
		else:
			return 11#2

	if (animal_type == "Dog"):
		shelter_train["Aggresiveness"] = shelter_train["Breed"].apply(aggressive)
		shelter_test["Aggresiveness"] = shelter_test["Breed"].apply(aggressive)

	#Most allergic breeds:
	#Akita, Alaskan Malamute, American Eskimo, Corgi, Chow-chow, German
	#Shepherd, Great Pyrenees, Labrador, Retriever, Husky
	def allergic(breed):
		if breed.find("Akita") != -1:
			return 1
		elif breed.find("Malamute") != -1:
			return 2#1
		elif breed.find("Eskimo") != -1:
			return 3#1
		elif breed.find("Corgi") != -1:
			return 4#1
		elif breed.find("Chow") != -1:
			return 5#1
		elif breed.find("Shepherd") != -1:
			return 6#1
		elif breed.find("Pyrenees") != -1:
			return 7#1
		elif breed.find("Labrador") != -1:
			return 8#1
		elif breed.find("Retriever") != -1:
			return 9#1
		elif breed.find("Husky") != -1:
			return 10#1
		else:
			return 11#2

	if (animal_type == "Dog"):
		shelter_train["Allergic"] = shelter_train["Breed"].apply(allergic)
		shelter_test["Allergic"] = shelter_test["Breed"].apply(allergic)

	#weight based on breed type. Most dangerous breeds:
	# Below 100 lbs: Pitbull (55-65 lbs), Husky-type (66 lbs), Doberman pinscher (65-90lbs), Boxer (70 lbs), Akita (45 kg), chow chow (70 lbs)
	# Above 100 lbs: Rottweiler (100-130 lbs), German Shepherd (100 lbs), Alaskan Malamute (100 lbs), Great Danes (200pounds), 
	def weight(breed):
		if breed.find("Pit Bull") != -1:
			return 1
		elif breed.find("Husky") != -1:
			return 1
		elif breed.find("Doberman") != -1:
			return 1
		elif breed.find("Boxer") != -1:
			return 1
		elif breed.find("Akita") != -1:
			return 1
		elif breed.find("Chow") != -1:
			return 1
		elif breed.find("Rottweiler") != -1:
			return 2
		elif breed.find("Shepherd") != -1:
			return 2
		elif breed.find("Malamute") != -1:
			return 2
		elif breed.find("Dane") != -1:
			return 2
		else:
			return 3

	if (animal_type == "Dog"):
		shelter_train["Weight"] = shelter_train["Breed"].apply(weight)
		shelter_test["Weight"] = shelter_test["Breed"].apply(weight)

	# #fetch breed type
	# def breed_group(breed_input):
	# 	breed = str(breed_input)
	# 	if (' ' in breed) == False:
	# 		return breed #only 1 word
	# 	breed_list = breed.split()
	# 	try:
	# 		return breed_list[2] #fetch last word, for 1 words breed
	# 	except:
	# 		return breed_list[1] #fetch last word, for 2 words breed
	# 	return breed

	def breed_group(breed_input):
		breed = str(breed_input)
		if (' ' in breed) == False:
			br =  breed #only 1 word
		else:
			breed_list = breed.split()
			try:
				br = breed_list[2] #fetch last word, for 1 words breed
			except:
				br = breed_list[1] #fetch last word, for 2 words breed
		if (br == "Mix"):
			return 0
		else:
			return 1
		return 1

	shelter_train["Breed"] = shelter_train["Breed"].apply(breed_group)
	shelter_test["Breed"] = shelter_test["Breed"].apply(breed_group)
	# ### convert each unique label to unique integers
	# intval, label = pd.factorize(shelter_train["Breed"], sort=True)
	# shelter_train["Breed"] = pd.DataFrame(intval)
	# del intval, label
	# intval, label = pd.factorize(shelter_test["Breed"], sort=True)
	# shelter_test["Breed"] = pd.DataFrame(intval)
	# del intval, label

	#########################################################
	#### Fetch First word of Color
	#########################################################

	# Color Intact
	def color_group(color):
		try:
			color_type = color.split()
		except:
			return "unknown"
		return str(color_type[0])

	shelter_train["Color"] = shelter_train["Color"].apply(color_group)
	shelter_test["Color"] = shelter_test["Color"].apply(color_group)
	#### convert each unique label to unique integers
	intval, label = pd.factorize(shelter_train["Color"], sort=True)
	shelter_train["Color"] = pd.DataFrame(intval)
	del intval, label
	intval, label = pd.factorize(shelter_test["Color"], sort=True)
	shelter_test["Color"] = pd.DataFrame(intval)
	del intval, label
	#Color: Drop
	# shelter_train.drop("Color", axis=1, inplace=True)
	# shelter_test.drop("Color", axis=1, inplace=True)

	print(shelter_train.head())
	return shelter_train, shelter_test

dog_train, dog_test = pre_processing(dog_train, dog_test, "Dog")
cat_train, cat_test = pre_processing(cat_train, cat_test, "Cat")

#########################################################################################################
#### Plot Pre-Processed Data
#########################################################################################################
plot = True

if plot == True: 
	#plot correlation between each feature and OutcomeType
	axis = {
			"AgeuponOutcome",
			"Hour",
			#"Minute",
			"Virginity",
			}
	count = 0
	for ax in axis:
		x = dog_train[ax]
		y = dog_train_outcome
		y = y.reset_index()
		y = y["OutcomeType"]
		plt.figure(count)
		plt.title("Dog" + ax)
		sns.countplot(x=x, hue=y)
		count = count + 1

	for ax in axis:
		x = cat_train[ax]
		y = cat_train_outcome
		y = y.reset_index()
		y = y["OutcomeType"]
		plt.figure(count)
		plt.title("Cat" + ax)
		sns.countplot(x=x, hue=y)
		count = count + 1

	plt.show()


#########################################################################################################
#### PCA
#########################################################################################################
# from sklearn.decomposition import PCA
# n = 12
# pca_train = PCA(n_components=n)
# pca_test = PCA(n_components=n)
# pca_train.fit(dog_train)
# pca_test.fit(dog_test)

# dog_train = pd.DataFrame(pca_train.transform(dog_train))
# dog_test = pd.DataFrame(pca_test.transform(dog_test))


#########################################################################################################
#### Split Cross-Validation Dataset
#########################################################################################################
from sklearn.cross_validation import train_test_split
dog_X_train, dog_X_val, dog_y_train, dog_y_val = train_test_split(dog_train, dog_train_outcome, test_size=0.3)
cat_X_train, cat_X_val, cat_y_train, cat_y_val = train_test_split(cat_train, cat_train_outcome, test_size=0.3)


#########################################################################################################
#### Prediction Model
#########################################################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss

classifiers = [
    # KNeighborsClassifier(100),
    # SVC(max_iter=1000, probability=True, kernel='rbf', degree=20),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(max_depth=3),
    # RandomForestClassifier(max_depth=5, n_estimators=500, max_features=1),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    # LogisticRegression(),
    GradientBoostingClassifier()
    # GradientBoostingClassifier(learning_rate=0.05, min_samples_split=50, max_depth=8)
    # GradientBoostingClassifier(learning_rate=0.005, n_estimators=3000, min_samples_split=600, min_samples_leaf=30, max_depth=12, subsample=0.85)
    ]

print("DOG")
for classifier in classifiers:
	dog_log = classifier 
	dog_log.fit(dog_X_train, dog_y_train)

	show_validation = True

	if (show_validation == True):
		dog_y_probs = dog_log.predict_proba(dog_X_val)
		dog_y_pred = dog_log.predict(dog_X_val)
		print(type(classifier))
		print("accuracy_score:", accuracy_score(dog_y_val, dog_y_pred))
		print("log_loss:", log_loss(dog_y_val, dog_y_probs))
	elif (show_validation == False):
		dog_y_probs = dog_log.predict_proba(dog_X_train)
		dog_y_pred = dog_log.predict(dog_X_train)
		print(type(classifier))
		print("accuracy_score:", accuracy_score(dog_y_train, dog_y_pred))
		print("log_loss:", log_loss(dog_y_train, dog_y_probs))

print("CAT")
for classifier in classifiers:
	cat_log = classifier
	cat_log.fit(cat_X_train, cat_y_train)

	show_validation = True
	# log knows how many classes are there idn y_train
	if (show_validation == True):
		cat_y_probs = cat_log.predict_proba(cat_X_val)
		cat_y_pred = cat_log.predict(cat_X_val)
		print(type(classifier))
		print("accuracy_score:", accuracy_score(cat_y_val, cat_y_pred))
		print("log_loss:", log_loss(cat_y_val, cat_y_probs))
	elif (show_validation == False):
		print(type(classifier))
		print("accuracy_score:", accuracy_score(cat_y_train, cat_y_pred))
		print("log_loss:", log_loss(cat_y_train, cat_y_probs))



#########################################################################################################
#### Model Fitting for Dog
#########################################################################################################
#fit dog data
dog_log.fit(dog_train, dog_train_outcome)
print(dog_log.classes_)
dog_y_probs = dog_log.predict_proba(dog_test)
dog_test_result = np.append(dog_test_ID, dog_y_probs, axis=1)

#plot feature importance

plt.figure(0)
plt.title("Feature Importance")
print(dog_log.feature_importances_)
print(dog_train.columns)
importance = dog_log.feature_importances_
sns.barplot(y=dog_train.columns, x=importance)
#########################################################################################################
#### Model Fitting for Cat
#########################################################################################################

#fit cat data
cat_log.fit(cat_train, cat_train_outcome)
print(cat_log.classes_)
cat_y_probs = cat_log.predict_proba(cat_test)
cat_test_result = np.append(cat_test_ID, cat_y_probs, axis=1)

#plot feature importance
plt.figure(1)
plt.title("Feature Importance")
print(cat_log.feature_importances_)
print(cat_train.columns)
importance = cat_log.feature_importances_
sns.barplot(y=cat_train.columns, x=importance)

#plt.show()

#########################################################################################################
#### Combine All Prediction
#########################################################################################################
y_probs = np.append(dog_test_result, cat_test_result, axis=0)
y_probs = y_probs[y_probs[:,0].argsort()]
y_probs = y_probs[:,1:]
print(y_probs)

results = pd.read_csv("../input/sample_submission.csv")

#each result has their corresponding probabilistic value
results["Adoption"] = y_probs[:,0]
results["Died"] = y_probs[:,1]
results["Euthanasia"] = y_probs[:,2]
results["Return_to_owner"] = y_probs[:,3]
results["Transfer"] = y_probs[:,4]

results.to_csv("split_animal.csv",index = False)


