# 1. Import Libraries

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

from sklearn.naive_bayes import GaussianNB



# Evaluation Metrics

from yellowbrick.classifier import ClassificationReport

from sklearn import metrics
# Read dataset and display first 5 row

df = pd.read_csv('../input/train.csv', error_bad_lines=False) 

df.head(5)
# 2. Preliminary Analysis

df.info()
# 3. Data Preparation

# Remove irrelevant/not meaningfull attributes

df = df.drop(['Descript'], axis=1)

df = df.drop(['Resolution'], axis=1)



df.info()
# Splitting the Date to Day, Month, Year, Hour, Minute, Second

df['date2'] = pd.to_datetime(df['Dates'])

df['Year'] = df['date2'].dt.year

df['Month'] = df['date2'].dt.month

df['Day'] = df['date2'].dt.day

df['Hour'] = df['date2'].dt.hour

df['Minute'] = df['date2'].dt.minute

df['Second'] = df['date2'].dt.second 

df = df.drop(['Dates'], axis=1) 

df = df.drop(['date2'], axis=1)

df.head(5)
# Convert Categorical Attributes to Numerical

df['PdDistrict'] = pd.factorize(df["PdDistrict"])[0]

df['Address'] = pd.factorize(df["Address"])[0]

df['DayOfWeek'] = pd.factorize(df["DayOfWeek"])[0]

df['Year'] = pd.factorize(df["Year"])[0]

df['Month'] = pd.factorize(df["Month"])[0]

df['Day'] = pd.factorize(df["Day"])[0]

df['Hour'] = pd.factorize(df["Hour"])[0]

df['Minute'] = pd.factorize(df["Minute"])[0]

df['Second'] = pd.factorize(df["Second"])[0] 

df.head(5)
# Display targer class

Target = 'Category'

print('Target: ', Target)
# Plot Bar Chart visualize Crime Types

plt.figure(figsize=(14,10))

plt.title('Amount of Crimes by Category')

plt.ylabel('Crime Category')

plt.xlabel('Amount of Crimes')



df.groupby([df['Category']]).size().sort_values(ascending=True).plot(kind='barh')



plt.show()
# Display all unique classes

Classes = df['Category'].unique()

Classes
#Encode target labels into categorical variables:

df['Category'] = pd.factorize(df["Category"])[0] 

df['Category'].unique()
# 4. Feature Selection using Filter Method 

# Split Dataframe to target class and features

X_fs = df.drop(['Category'], axis=1)

Y_fs = df['Category']



#Using Pearson Correlation

plt.figure(figsize=(20,10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Correlation with output variable

cor_target = abs(cor['Category'])

#Selecting highly correlated features

relevant_features = cor_target[cor_target>0.02]

relevant_features
# At Current Point, the attributes is select manually based on Feature Selection Part. 

Features = ["Address","Year","Hour", "Minute" ]

print('Full Features: ', Features) 
# 5. Split dataset to Training Set & Test Set

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
# 6. Random Forest

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
# 7. Neural Network

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
# 8. Naive Bayes

# Create Model with configuration 

nb_model = GaussianNB() 



# Model Training

nb_model.fit(X=x1, 

             y=x2)



# Prediction

result = nb_model.predict(y[Features])
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