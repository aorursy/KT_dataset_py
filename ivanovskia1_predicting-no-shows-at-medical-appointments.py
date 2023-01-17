import pandas as pd

df = pd.read_csv("../input/KaggleV2-May-2016.csv")

df.describe()
df.shape
df.columns
no_show = df["No-show"].value_counts()

print(no_show)



Percent_no_show = no_show["Yes"]/ no_show.sum() * 100

print("Percent who didn't show up to their appointment:",Percent_no_show )
df['No-show'].replace("No", 0,inplace=True)

df['No-show'].replace("Yes", 1,inplace=True)

df["Gender"].value_counts()
Exploratory_Analysis = ['Gender','Hipertension','Alcoholism','Diabetes']

for r in Exploratory_Analysis :

    print(df.groupby(r)['No-show'].mean())

#df.groupby('Hipertension')['No-show'].mean()
#Convert to Categorical

df['Handcap'] = pd.Categorical(df['Handcap'])

#Convert to Dummy Variables

Handicap = pd.get_dummies(df['Handcap'], prefix = 'Handicap')

df = pd.concat([df, Handicap], axis=1)

## See how each type of handicap affects whether a patient will show up to an appointment 

handicaps = ["Handicap_1", "Handicap_2", "Handicap_3", "Handicap_4"]

for h in handicaps:

    print(df.groupby(h)['No-show'].mean())

    
df.groupby('SMS_received')['No-show'].mean()
min_age = df.Age.min()

print("Min Age:", min_age)

max_age = df.Age.max()

print("Max Age:", max_age)

df = df[(df.Age >= 0) & (df.Age <= 100)]
import numpy as np



# Converts the two variables to datetime variables

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])



# Create a variable called "AwaitingTime" by subtracting the date the patient made the appointment and the date of the appointment.

df['AwaitingTime'] = df["AppointmentDay"].sub(df["ScheduledDay"], axis=0)



# Convert the result "AwaitingTime" to number of days between appointment day and scheduled day. 

df["AwaitingTime"] = (df["AwaitingTime"] / np.timedelta64(1, 'D')).abs()

# Number of Appointments Missed by Patient

df['Num_App_Missed'] = df.groupby('PatientId')['No-show'].apply(lambda x: x.cumsum())

df.columns
df.drop(["PatientId",'AppointmentID',

         'ScheduledDay','Handcap','AppointmentDay','Neighbourhood'], axis=1, inplace = True)
X = df[['Gender', 'Diabetes','Hipertension', 'Scholarship', 'SMS_received',

        'Handicap_0','Handicap_1','Handicap_2','Handicap_3','Handicap_4', 'Num_App_Missed', 'Age', 'AwaitingTime']]



y = df["No-show"]
#This will create dummies for the remaining variables (Gender)

X_train1 = pd.get_dummies(X)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train1)

rescaledX2 = scaler.transform(X_train1)

# summarize transformed data

np.set_printoptions(precision=3)

print(rescaledX2[0:5,:])
from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split



# Create Training and Test Dataset with 75% Training and 25% Test

X_train, X_test, y_train, y_test = train_test_split(rescaledX2, y, test_size=0.25)



# Run Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



# Analyze results

print("Results:")

print("Accuracy", metrics.accuracy_score(y_test,y_pred))



# save confusion matrix and slice into four pieces

confusion = metrics.confusion_matrix(y_test, y_pred)

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]



#Specificity: When the actual value is negative, how often is the prediction correct?

print("Specificity:",TN / float(TN + FP))



#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?

print("False Positive Rate:",FP / float(TN + FP))



#Precision: When a positive value is predicted, how often is the prediction correct?

print("Precision:",metrics.precision_score(y_test, y_pred))



#Sensitivity:

print("Recall:",metrics.recall_score(y_test, y_pred))
# examine the class distribution of the testing set (using a Pandas Series method)

print("Class Distribution:", y_test.value_counts())

# calculate the percentage of ones

print("Percentage of Ones:", y_test.mean())



# calculate the percentage of zeros

print("Percentage of Zeros:", 1 - y_test.mean())



# calculate null accuracy (for binary classification problems coded as 0/1)

print("Null Accuracy:",max(y_test.mean(), 1 - y_test.mean()))
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import train_test_split



X_train2, X_test2, y_train2, y_test2 = train_test_split(rescaledX2, y, test_size=0.25, random_state = 42)



tree = DecisionTreeClassifier(max_depth=10, random_state=0)

tree.fit(X_train2, y_train2)

y_pred2 = tree.predict(X_test2)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train2, y_train2)))

print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test2, y_test2)))





# save confusion matrix and slice into four pieces

confusion = metrics.confusion_matrix(y_test2, y_pred2)

TP = confusion[1, 1]

TN = confusion[0, 0]

FP = confusion[0, 1]

FN = confusion[1, 0]



#Specificity: When the actual value is negative, how often is the prediction correct?

print("Specificity:",TN / float(TN + FP))



#False Positive Rate: When the actual value is negative, how often is the prediction incorrect?

print("False Positive Rate:",FP / float(TN + FP))



#Precision: When a positive value is predicted, how often is the prediction correct?

print("Precision:",metrics.precision_score(y_test2, y_pred2))



#Sensitivity:

print("Recall:",metrics.recall_score(y_test2, y_pred2))
import matplotlib.pyplot as plt

n_features = X_train1.shape[1]

plt.barh(range(n_features), tree.feature_importances_, )

plt.yticks(np.arange(n_features), X_train1)

plt.xlabel('Feature Importance')

plt.ylabel('Feature')

plt.show()
bins = [0, 20, 40, 60, 100]

age_groups = df.groupby(pd.cut(df.Age, bins))

age_groups["No-show"].mean()

bins = [0,1,7, 180]

wait_time = df.groupby(pd.cut(df.AwaitingTime, bins))

wait_time["No-show"].mean()