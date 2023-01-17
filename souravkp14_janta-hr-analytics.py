import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("/kaggle/input/train.csv")

test = pd.read_csv("/kaggle/input/test.csv")



test_enrollee_id = test['enrollee_id']

n_train = train.shape[0]

n_test = test.shape[0]



data = pd.concat([train,test])

del data['enrollee_id']



print(data.shape)

print(n_train)

print(n_test)
data.head()
data.info()
count_classes = pd.value_counts(train['target'], sort=True)

count_classes.plot(kind='bar', rot=0)

plt.title('Job change class Distribution')

plt.xticks(range(2))

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
NoJobChange = len(train[train.target == 0])

JobChange = len(train[train.target == 1])

print("No. of Employees not looking for job change: {}".format(NoJobChange))

print("No. of Employees looking for job change: {}".format(JobChange))

print("Percentage of Employees not looking for job change: {:.2f}%".format((NoJobChange / (len(train.target))*100)))

print("Percentage of Employees looking for job change: {:.2f}%".format((JobChange / (len(train.target))*100)))
data.isnull().sum()
percent_missing = data.isnull().sum() * 100 / len(data)

missing_value_df = pd.DataFrame({'percent_missing': percent_missing})

missing_value_df
#Ratios in dataset





#gender

print("Gender\n")

male = len(data[data.gender == "Male"])

female = len(data[data.gender == "Female"])

other = len(data[data.gender == "Other"])



print("Ratio of Male: {}".format(male / 25894))

print("Ratio of Female: {}".format(female / 25894))

print("Ratio of Other: {}".format(other / 25894))



print("-"*50)



#major_discipline

print("Major discipline\n")

STEM = len(data[data.major_discipline == "STEM"])

Other = len(data[data.major_discipline == "Other"])

No_Major = len(data[data.major_discipline == "No Major"])

Business_Degree = len(data[data.major_discipline == "Business Degree"])

Arts = len(data[data.major_discipline == "Arts"])

Humanities = len(data[data.major_discipline == "Humanities"])





print("Ratio of STEM: {}".format(STEM / 28149))

print("Ratio of Other: {}".format(Other / 28149))

print("Ratio of No Major: {}".format(No_Major / 28149))

print("Ratio of Business Degree: {}".format(Business_Degree / 28149))

print("Ratio of Arts: {}".format(Arts / 28149))

print("Ratio of Humanities: {}".format(Humanities / 28149))





print("-"*50)





#company_size

print("Company size\n")

_100to500 = len(data[data.company_size == "100-500"])

_less10 = len(data[data.company_size == "<10"])

_50to99 = len(data[data.company_size == "50-99"])

_5000to9999 = len(data[data.company_size == "5000-9999"])

_10000plus = len(data[data.company_size == "10000+"])

_1000to4999 = len(data[data.company_size == "1000-4999"])

_10to49 = len(data[data.company_size == "10/49"])

_500to999 = len(data[data.company_size == "500-999"])



print("Ratio of 100-500: {}".format(_100to500 / 24550))

print("Ratio of <10: {}".format(_less10 / 24550))

print("Ratio of 50-99: {}".format(_50to99 / 24550))

print("Ratio of 5000-9999: {}".format(_5000to9999 / 24550))

print("Ratio of 10000+: {}".format(_10000plus / 24550))

print("Ratio of 1000-4999: {}".format(_1000to4999 / 24550))

print("Ratio of 10/49: {}".format(_10to49 / 24550))

print("Ratio of 500-999: {}".format(_500to999 / 24550))







print("-"*50)





#company_type

print("Company type\n")

Pvt_Ltd = len(data[data.company_type == "Pvt Ltd"])

Funded_Startup = len(data[data.company_type == "Funded Startup"])

Public_Sector = len(data[data.company_type == "Public Sector"])

Early_Stage_Startup = len(data[data.company_type == "Early Stage Startup"])

NGO = len(data[data.company_type == "NGO"])

Other = len(data[data.company_type == "Other"])





print("Ratio of Pvt Ltd: {}".format(Pvt_Ltd / 24011))

print("Ratio of Funded Startup: {}".format(Funded_Startup / 24011))

print("Ratio of Public Sector: {}".format(Public_Sector / 24011))

print("Ratio of Early Stage Startup: {}".format(Early_Stage_Startup / 24011))

print("Ratio of NGO: {}".format(NGO / 24011))

print("Ratio of Other: {}".format(Other / 24011))





print("-"*50)
#Filling null values



# gender

data['gender'] = data['gender'].fillna(pd.Series(np.random.choice(['Male', 'Female', 'Other'], 

                                                                    p=[0.90, 0.08, 0.02], size=len(data))))



#enrolled_university

data['enrolled_university'] = data['enrolled_university'].fillna((data['enrolled_university'].mode()[0]))



#education_level

data['education_level'] = data['education_level'].fillna((data['education_level'].mode()[0]))





#major_discipline

data['major_discipline'] = data['major_discipline'].fillna(pd.Series(np.random.choice(['STEM', 'Other', 'No Major', 

'Business Degree', 'Arts', 'Humanities'], p=[0.88, 0.02, 0.01, 0.02, 0.02, 0.05], size=len(data))))





#experience

data['experience'] = data['experience'].fillna((data['experience'].mode()[0]))





#company_size

data['company_size'] = data['company_size'].fillna(pd.Series(np.random.choice(['100-500', '<10', '50-99',

'5000-9999', '10000+', '1000-4999', '10/49', '500-999' ], p=[0.2, 0.1, 0.24, 0.04, 0.15, 0.10, 0.1, 0.07],

                                                                                size=len(data))))

#company_type

data['company_type'] = data['company_type'].fillna(pd.Series(np.random.choice(['Pvt Ltd', 'Funded Startup',

'Public Sector', 'Early Stage Startup', 'NGO', 'Other' ], p=[0.76, 0.08, 0.07, 0.04, 0.04, 0.01 ], size=len(data))))



#last_new_job

data['last_new_job'] = data['last_new_job'].fillna((data['last_new_job'].mode()[0]))

#Label encoding

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



data['relevent_experience'] = labelencoder.fit_transform(data['relevent_experience'])



data['enrolled_university'] = labelencoder.fit_transform(data['enrolled_university'])



data['education_level'] = labelencoder.fit_transform(data['education_level'])



data['city'] = labelencoder.fit_transform(data['city'])

print(data.last_new_job.unique())

print(data.experience.unique())
data.experience = data.experience.replace(">20", 21)

data.experience = data.experience.replace("<1", 0)



data.last_new_job = data.last_new_job.replace(">4", 5)

data.last_new_job = data.last_new_job.replace("never", 0)



data['experience'] = data['experience'].astype(str).astype(int)

data['last_new_job'] = data['last_new_job'].astype(str).astype(int)



data['training_hours'] = np.log(data['training_hours'])
#Since 'gender', 'major_discipline', 'company_size', 'company_type'  are categorical variables we'll turn them into dummy variables.



a = pd.get_dummies(data['gender'], prefix = "gender")

b = pd.get_dummies(data['major_discipline'], prefix = "major_discipline")

c = pd.get_dummies(data['company_size'], prefix = "company_size")

d = pd.get_dummies(data['company_type'], prefix = "company_type")





frames = [data, a, b, c, d]

data = pd.concat(frames, axis = 1)



data = data.drop(columns = ['gender', 'major_discipline', 'company_size', 'company_type' ])
cols_data = list(data.columns.values)

cols_data
#Rearrange column order



data = data[['city',

 'city_development_index',

 'education_level',

 'enrolled_university',

 'experience',

 'last_new_job',

 'relevent_experience',

 'training_hours',

 'gender_Female',

 'gender_Male',

 'gender_Other',

 'major_discipline_Arts',

 'major_discipline_Business Degree',

 'major_discipline_Humanities',

 'major_discipline_No Major',

 'major_discipline_Other',

 'major_discipline_STEM',

 'company_size_10/49',

 'company_size_100-500',

 'company_size_1000-4999',

 'company_size_10000+',

 'company_size_50-99',

 'company_size_500-999',

 'company_size_5000-9999',

 'company_size_<10',

 'company_type_Early Stage Startup',

 'company_type_Funded Startup',

 'company_type_NGO',

 'company_type_Other',

 'company_type_Public Sector',

 'company_type_Pvt Ltd',

 'target'

]]
data.head()
X_cluster = data.iloc[:, 0:-1].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X_cluster)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)

cluster = kmeans.fit_predict(X_cluster)
#train

data['cluster']= cluster

data.head()
train_data = data.iloc[:n_train]

test_data = data.iloc[n_train:]



del test_data['target']



print(train_data.shape)

print(test_data.shape)
y_train = train_data['target']

del train_data['target']

x_train = train_data



x_test = test_data
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics.scorer import make_scorer

from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.feature_selection import SelectKBest, chi2
model = ExtraTreeClassifier()

model.fit(x_train,y_train)

feature_imp = pd.DataFrame({'Feature' : x_train.columns, 'Score' : model.feature_importances_})

feature_imp.sort_values(by=['Score'], ascending=False)
# Initialise the Scaler 

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler() 

x_train = scaler.fit_transform(x_train)
from sklearn.utils import class_weight



class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

class_weights_dict = dict(enumerate(class_weights))

print(class_weights_dict)
dt = DecisionTreeClassifier(class_weight=class_weights_dict)



scores = cross_validate(dt, x_train, y_train, cv=2, scoring=make_scorer(roc_auc_score, average='weighted'))

print(scores['test_score'])

print(np.mean(scores['test_score']))
rf = RandomForestClassifier(n_estimators=200, class_weight=class_weights_dict)



scores = cross_validate(rf, x_train, y_train, cv=2, scoring=make_scorer(roc_auc_score, average='weighted'))

print(scores['test_score'])

print(np.mean(scores['test_score']))
gb = GradientBoostingClassifier(n_estimators=200)



scores = cross_validate(gb, x_train, y_train, cv=2, scoring=make_scorer(roc_auc_score, average='weighted'))

print(scores['test_score'])

print(np.mean(scores['test_score']))
lr = LogisticRegression(random_state=0, class_weight=class_weights_dict)



scores = cross_validate(lr, x_train, y_train, cv=2, scoring=make_scorer(roc_auc_score, average='weighted'))

print(scores['test_score'])

print(np.mean(scores['test_score']))
clf = lr
clf.fit(x_train, y_train)
result = pd.DataFrame(data=test['enrollee_id'])

result.head()
test.head()
test = test.drop(['enrollee_id'], axis=1)
x_test = scaler.transform(x_test)

pred = clf.predict(x_test)

print(pred)
test.shape
pred.shape
#Column addition



result['target']= pred

result.head(25)
result.to_csv("prediction.csv", index=False)