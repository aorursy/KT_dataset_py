# Visualization Libraries

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



#Preprocessing Libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score



# ML Libraries

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier



# Evaluation Metrics

from yellowbrick.classifier import ClassificationReport

from sklearn import metrics
df = pd.concat([pd.read_csv('../input/Chicago_Crimes_2001_to_2004.csv', error_bad_lines=False), pd.read_csv('../input/Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)], ignore_index=True)

df = pd.concat([df, pd.read_csv('../input/Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)], ignore_index=True)

df = pd.concat([df, pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)], ignore_index=True)

df.head()
df.info()
# Preprocessing

# Remove NaN Value (As Dataset is huge, the NaN row could be neglectable)  

df = df.dropna()
# As the dataset is too huge is size, we would just subsampled a dataset for modelling as proof of concept

df = df.sample(n=100000)
# Remove irrelevant/not meaningfull attributes

df = df.drop(['Unnamed: 0'], axis=1)

df = df.drop(['ID'], axis=1)

df = df.drop(['Case Number'], axis=1) 



df.info()
# Splitting the Date to Day, Month, Year, Hour, Minute, Second

df['date2'] = pd.to_datetime(df['Date'])

df['Year'] = df['date2'].dt.year

df['Month'] = df['date2'].dt.month

df['Day'] = df['date2'].dt.day

df['Hour'] = df['date2'].dt.hour

df['Minute'] = df['date2'].dt.minute

df['Second'] = df['date2'].dt.second 

df = df.drop(['Date'], axis=1) 

df = df.drop(['date2'], axis=1) 

df = df.drop(['Updated On'], axis=1)

df.head()
# Convert Categorical Attributes to Numerical

df['Block'] = pd.factorize(df["Block"])[0]

df['IUCR'] = pd.factorize(df["IUCR"])[0]

df['Description'] = pd.factorize(df["Description"])[0]

df['Location Description'] = pd.factorize(df["Location Description"])[0]

df['FBI Code'] = pd.factorize(df["FBI Code"])[0]

df['Location'] = pd.factorize(df["Location"])[0] 
Target = 'Primary Type'

print('Target: ', Target)
# Plot Bar Chart visualize Primary Types

plt.figure(figsize=(14,10))

plt.title('Amount of Crimes by Primary Type')

plt.ylabel('Crime Type')

plt.xlabel('Amount of Crimes')



df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')



plt.show()
# At previous plot, we could see that the classes is quite imbalance

# Therefore, we are going to group several less occured Crime Type into 'Others' to reduce the Target Class amount



# First, we sum up the amount of Crime Type happened and select the last 13 classes

all_classes = df.groupby(['Primary Type'])['Block'].size().reset_index()

all_classes['Amt'] = all_classes['Block']

all_classes = all_classes.drop(['Block'], axis=1)

all_classes = all_classes.sort_values(['Amt'], ascending=[False])



unwanted_classes = all_classes.tail(13)

unwanted_classes
# After that, we replaced it with label 'OTHERS'

df.loc[df['Primary Type'].isin(unwanted_classes['Primary Type']), 'Primary Type'] = 'OTHERS'



# Plot Bar Chart visualize Primary Types

plt.figure(figsize=(14,10))

plt.title('Amount of Crimes by Primary Type')

plt.ylabel('Crime Type')

plt.xlabel('Amount of Crimes')



df.groupby([df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh')



plt.show()
# Now we are left with 14 Class as our predictive class

Classes = df['Primary Type'].unique()

Classes
#Encode target labels into categorical variables:

df['Primary Type'] = pd.factorize(df["Primary Type"])[0] 

df['Primary Type'].unique()
# Feature Selection using Filter Method 

# Split Dataframe to target class and features

X_fs = df.drop(['Primary Type'], axis=1)

Y_fs = df['Primary Type']



#Using Pearson Correlation

plt.figure(figsize=(20,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor['Primary Type'])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.2]

relevant_features
# At Current Point, the attributes is select manually based on Feature Selection Part. 

Features = ["IUCR", "Description", "FBI Code"]

print('Full Features: ', Features)
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

target_names = Classes

visualizer = ClassificationReport(rf_model, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data
# Neural Network

# Create Model with configuration 

nn_model = MLPClassifier(solver='adam', 

                         alpha=1e-5,

                         hidden_layer_sizes=(40,), 

                         random_state=1,

                         max_iter=1000                         

                        )



# Model Training

nn_model.fit(X=x1,

             y=x2)



# Prediction

result = nn_model.predict(y[Features]) 
# Model Evaluation

ac_sc = accuracy_score(y2, result)

rc_sc = recall_score(y2, result, average="weighted")

pr_sc = precision_score(y2, result, average="weighted")

f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("========== Neural Network Results ==========")

print("Accuracy    : ", ac_sc)

print("Recall      : ", rc_sc)

print("Precision   : ", pr_sc)

print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
# Classification Report

# Instantiate the classification model and visualizer

target_names = Classes

visualizer = ClassificationReport(nn_model, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data
# K-Nearest Neighbors

# Create Model with configuration 

knn_model = KNeighborsClassifier(n_neighbors=3)



# Model Training

knn_model.fit(X=x1,

             y=x2)



# Prediction

result = knn_model.predict(y[Features]) 
# Model Evaluation

ac_sc = accuracy_score(y2, result)

rc_sc = recall_score(y2, result, average="weighted")

pr_sc = precision_score(y2, result, average="weighted")

f1_sc = f1_score(y2, result, average='micro')

confusion_m = confusion_matrix(y2, result)



print("========== K-Nearest Neighbors Results ==========")

print("Accuracy    : ", ac_sc)

print("Recall      : ", rc_sc)

print("Precision   : ", pr_sc)

print("F1 Score    : ", f1_sc)

print("Confusion Matrix: ")

print(confusion_m)
# Classification Report

# Instantiate the classification model and visualizer

target_names = Classes

visualizer = ClassificationReport(knn_model, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data
# Ensemble Voting Model

# Combine 3 Models to create an Ensemble Model



# Create Model with configuration

eclf1 = VotingClassifier(estimators=[('knn', knn_model), ('rf', rf_model), ('nn', nn_model)], 

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

target_names = Classes

visualizer = ClassificationReport(eclf1, classes=target_names)

visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer

visualizer.score(y1, y2)       # Evaluate the model on the test data



print('================= Classification Report =================')

print('')

print(classification_report(y2, result, target_names=target_names))



g = visualizer.poof()             # Draw/show/poof the data