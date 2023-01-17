
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from os import listdir
import seaborn as sns
import matplotlib.pyplot as plt
# Finding out all the files that we can analyse and list them for loading and further exploration. 
arr = os.listdir('/kaggle/input/healthcare-analytics/Train')
print(arr)


#Loading all the files for further explorations.
train_df = pd.read_csv('/kaggle/input/healthcare-analytics/Train/Train.csv')
first_health_df = pd.read_csv('/kaggle/input/healthcare-analytics/Train/First_Health_Camp_Attended.csv')
second_health_df = pd.read_csv('/kaggle/input/healthcare-analytics/Train/Second_Health_Camp_Attended.csv')
third_health_df = pd.read_csv('/kaggle/input/healthcare-analytics/Train/Third_Health_Camp_Attended.csv')
health_camp_detail_df = pd.read_csv('/kaggle/input/healthcare-analytics/Train/Health_Camp_Detail.csv')
patient_profile_df = pd.read_csv('/kaggle/input/healthcare-analytics/Train/Patient_Profile.csv')
test_init = pd.read_csv('/kaggle/input/healthcare-analytics/Train/test.csv')

first_health_df.head()
second_health_df.sort_values(['Health Score'], ascending = True)
third_health_df.head()
third_health_df.shape
third_health_df.sort_values(['Number_of_stall_visited'], ascending = True)
positive_outcome_third = third_health_df[third_health_df['Number_of_stall_visited'] > 0]
positive_outcome_third
positive_1 = first_health_df.drop(["Donation", 'Health_Score', 'Unnamed: 4'], axis=1)

positive_2 = second_health_df.drop(['Health Score'], axis=1)

positive_3 = positive_outcome_third.drop(['Number_of_stall_visited', 'Last_Stall_Visited_Number'], axis = 1)
positive_outcomes = pd.concat([positive_1, positive_2, positive_3], axis = 0)
#create tuple column with (Patient_ID, Health_Camp_ID) - we see we have 20534 positive outcomes.
positive_outcomes['patient_camp'] = list(zip(positive_outcomes.Patient_ID, positive_outcomes.Health_Camp_ID))
positive_outcomes.head()
train_df['patient_camp_train'] = list(zip(train_df.Patient_ID, train_df.Health_Camp_ID))
train_df.head()
positive_list = list(positive_outcomes.patient_camp)
len(positive_list)
health_camp_detail_df.info
patient_profile_df.head()
patient_profile_df.info()
positive_outcomes.head()

# we have 20534 patients and camps touples that qualify as - positive outcomes:
len(positive_list)
positive_list[:3]
#accessing only the camp portion of a tuple:
positive_list[0][1]
# how many unique patients we have that have positive outcomes?
unique_patient_ID_with_positive = list(positive_outcomes.Patient_ID.unique())
unique_patient_ID_with_positive[:3]
#we have only 11069 patients with positive outcomes:
len(unique_patient_ID_with_positive)
#Creating a function that shows all camps with positive outcome attended by a patient:
def camps(patient):
    camps = []
    i = 0
    while i < 20533:
        if positive_list[i][0] == patient:
            camps.append(positive_list[i][1])
        i += 1
    return camps

#patient_camps_attended = { 'patient_id': patient_profile_df.Patient_ID,
#                          'camps': camps(patient_profile_df.Patient_ID)}
camps(0)
train_df.size

def outcome(patient, camp):
    if len(camps(patient)) == 0:
        return 0
    
    elif camp in camps(patient):
        return 1 
    else:
        return 0
    
outcome(485679, 6555)
outcome(485679, 6578)
train_df.head()
outcomes = []
for patient_camp_tuple in train_df.patient_camp_train:
    outcomes.append(outcome(patient_camp_tuple[0], patient_camp_tuple[1])) 
outcomes
df = pd.DataFrame(outcomes,columns=['Outcome'])
df.head()
type(df.Outcome)
type(train_df.Var1)
new = pd.concat((train_df, df), axis = 1)
new.head()
new.Outcome.unique()
new.Outcome[10014]
new.Outcome.sum()
train_df.head()
train_df.shape
test_init.shape
test_init.head()
new.head()
new.head()
### Preping and Spliting the data 
# Separate features and labels
X = new.drop(["Patient_ID", "Health_Camp_ID","Registration_Date", "Outcome", "patient_camp_train"], axis = 1)
y = df
#y = new.drop(["Patient_ID", "Health_Camp_ID","Registration_Date","patient_camp_train", "Var1", "Var2", "Var3", "Var4", "Var5"], axis = 1)
X.head()
X.describe()
from matplotlib import pyplot as plt
%matplotlib inline

features = ['Var1','Var2','Var3','Var4','Var5']
for col in features:
    new.boxplot(column=col, by='Outcome', figsize=(6,6))
    plt.title(col)
plt.show()
y.head()
from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

print ('Training cases: %d\nTest cases: %d' % (X_train.size, X_test.size))
# Train the model
from sklearn.linear_model import LogisticRegression

# Set regularization rate
reg = 0.01

# train a logistic regression model on the training set
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train.values.ravel())
                                                            
                                                        
print (model)
predictions = model.predict(X_test)
print('Predicted labels: ', predictions)
print('Actual labels:    ' ,y_test)
predictions.sum()
y_test.Outcome.sum()
from sklearn.metrics import accuracy_score

print('Accuracy: ', accuracy_score(y_test, predictions))
from sklearn.metrics import precision_score, recall_score

print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
from sklearn.metrics import confusion_matrix

# Print the confusion matrix
cm = confusion_matrix(y_test, predictions)
print (cm)
y_scores = model.predict_proba(X_test)
print(y_scores)
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
# Train the model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Define preprocessing for numeric columns (normalize them so they're on the same scale)
numeric_features = [0,1,2,3,4]
numeric_transformer = Pipeline(steps=[
    ('scaler', PowerTransformer())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', LogisticRegression(C=1/reg, solver="liblinear"))])


# fit the pipeline to train a logistic regression model on the training set
model_pipe = pipeline.fit(X_train, (y_train.values.ravel()))
print (model_pipe)
# Get predictions from test data
predictions_pipe = model_pipe.predict(X_test)

# Get evaluation metrics
cm = confusion_matrix(y_test, predictions_pipe)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions_pipe))
print("Overall Precision:",precision_score(y_test, predictions_pipe))
print("Overall Recall:",recall_score(y_test, predictions_pipe))
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# calculate ROC curve
y_scores = model_pipe.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', RandomForestClassifier(n_estimators=100))])

# fit the pipeline to train a random forest model on the training set
model = pipeline.fit(X_train, (y_train.values.ravel()))
print (model)
predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print ('Confusion Matrix:\n',cm, '\n')
print('Accuracy:', accuracy_score(y_test, predictions))
print("Overall Precision:",precision_score(y_test, predictions))
print("Overall Recall:",recall_score(y_test, predictions))
auc = roc_auc_score(y_test,y_scores[:,1])
print('\nAUC: ' + str(auc))

# calculate ROC curve
y_scores = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

test_init.head()
X_test_test = test_init.drop(["Patient_ID", 'Health_Camp_ID', "Registration_Date"], axis = 1)
X_test_test.head()
test_init['patient_camp_train'] = list(zip(test_init.Patient_ID, test_init.Health_Camp_ID))
predictions_test = model.predict(X_test_test)
predictions_test
outcomes_test = pd.DataFrame(predictions_test, columns = ["Outcome"])
outcomes_test
test_for_output = test_init.drop(["Registration_Date", "Var1", "Var2", "Var3", "Var4", "Var5", "patient_camp_train"], axis = 1)
test_for_output
output_df = pd.concat([test_for_output, outcomes_test], axis = 1)
output_df 
output_df.to_csv("Output.csv", index = False)
testing_output_csv = pd.read_csv('/kaggle/working/Output.csv')

testing_output_csv
