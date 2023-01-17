#Importing Necessary Libraries

#Preprocessing Libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score



#Feature Selection Libraries

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



# ML Libraries (Random Forest, Naive Bayes, SVM)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

 

# Evaluation Metrics

from yellowbrick.classifier import ClassificationReport

from sklearn import metrics
#Read Iris Dataset into DataFrame & Print sample dataset

df = pd.read_csv('../input/Iris.csv', error_bad_lines=False)

df.head()
#Check if NA Exist

df.info()



#If no NA Exist, proceed

#df = df.dropna()
#Drop unusable Attributes: ID

#Reason: ID is the unique identifier that directly linked to each instance, which makes it not meaningful for training

df = df.drop(['Id'], axis=1)

#Display Target Class

df['Species'].unique()
#Encode labels into categorical variables:

df['Species'] = pd.factorize(df["Species"])[0] 

Target = 'Species'

df['Species'].unique()
#Define Full Feature Set

Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print('Full Features: ', Features)
# Feature Selection using Recursive Feature Elimination

# Split Dataframe to target class and features

X_fs = df[Features]

Y_fs = df[Target]

  

# Feature Selection Model Fitting

model = LogisticRegression(solver='lbfgs', multi_class='auto')



#Mark the Number of Features to be selected, Adjust this Number to enhance the Model Performance

rfe = RFE(model, 3) 

fit = rfe.fit(X_fs, Y_fs)



print("Number of Features Selected : %s" % (fit.n_features_))

print("Feature Ranking             : %s" % (fit.ranking_))

print("Selected Features           : %s" % (fit.support_))



# At Current Point, the attributes is select manually based on Feature Selection Part. 

Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print("Selected Features           :", Features)

#Split dataset to Training Set & Test Set

x, y = train_test_split(df, 

                        test_size = 0.2, 

                        train_size = 0.8, 

                        random_state= 3)



x1 = x[Features]    #Features to train

x2 = x[Target]      #Target Class to train

y1 = y[Features]    #Features to test

y2 = y[Target]      #Target Class to test



print('Feature Set Used    : ', Features)

print('Target Class        : ', Target)

print('Training Set Size   : ', x.shape)

print('Test Set Size       : ', y.shape)
# Gaussian Naive Bayes

# Create Model with configuration

nb_model = GaussianNB() 



# Model Training

nb_model.fit(X=x1, y=x2)



# Prediction with Test Set

result= nb_model.predict(y[Features]) 
# Model Evaluation

ac_sc = accuracy_score(y2, result)

rc_sc = recall_score(y2, result, average="weighted")

pr_sc = precision_score(y2, result, average="weighted")

f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("========== Naive Bayes Results ==========")

print("Accuracy    : ", ac_sc)

print("Recall      : ", rc_sc)

print("Precision   : ", pr_sc)

print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
# Classification Report

# Instantiate the classification model and visualizer

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

visualizer = ClassificationReport(nb_model, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data
# Random Forest

# Create Model with configuration

rf_model = RandomForestClassifier(n_estimators=70, # Number of trees

                                  min_samples_split = 30,

                                  bootstrap = True, 

                                  max_depth = 50, 

                                  min_samples_leaf = 25)



# Model Training

rf_model.fit(X=x1,

             y=x2)



# Prediction

result = rf_model.predict(y[Features])
# Model Evaluation

ac_sc = accuracy_score(y2, result)

rc_sc = recall_score(y2, result, average="weighted")

pr_sc = precision_score(y2, result, average="weighted")

f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("========== Random Forest Results ==========")

print("Accuracy    : ", ac_sc)

print("Recall      : ", rc_sc)

print("Precision   : ", pr_sc)

print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
# Classification Report

# Instantiate the classification model and visualizer

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

visualizer = ClassificationReport(rf_model, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data
# Support Vector Machine

# Create Model with configuration

svm_model = SVC(kernel='linear')



# Model Training

svm_model.fit(X=x1, y=x2)  



# Prediction

result = svm_model.predict(y[Features])  
# Model Evaluation

ac_sc = accuracy_score(y2, result)

rc_sc = recall_score(y2, result, average="weighted")

pr_sc = precision_score(y2, result, average="weighted")

f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("============= SVM Results =============")

print("Accuracy    : ", ac_sc)

print("Recall      : ", rc_sc)

print("Precision   : ", pr_sc)

print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
# Classification Report

# Instantiate the classification model and visualizer

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

visualizer = ClassificationReport(svm_model, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the Test Set



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data
# Ensemble Voting Model

# Combine 3 Models to create an Ensemble Model



# Create Model with configuration

eclf1 = VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model), ('gnb', nb_model)], 

                         weights=[1,1,1],

                         flatten_transform=True)

eclf1 = eclf1.fit(X=x1, y=x2)   



# Prediction

result = eclf1.predict(y[Features])
# Model Evaluation

ac_sc = accuracy_score(y2, result)

rc_sc = recall_score(y2, result, average="weighted")

pr_sc = precision_score(y2, result, average="weighted")

f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("============= Ensemble Voting Results =============")

print("Accuracy    : ", ac_sc)

print("Recall      : ", rc_sc)

print("Precision   : ", pr_sc)

print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
# Classification Report

# Instantiate the classification model and visualizer

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

visualizer = ClassificationReport(eclf1, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data