# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Import numpy, pandas, matpltlib.pyplot, sklearn modules and seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Uploading dataset from drive location using pandas library
accidents=pd.read_csv(r'/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
accidents.head()
data=accidents
data
accidents.info()
accidents.describe()
accidents.shape
accidents.columns
len(accidents)
# Convert Start_Time and End_Time to datetypes
accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'], errors='coerce')
accidents['End_Time'] = pd.to_datetime(accidents['End_Time'], errors='coerce')
# Extract year, month, day, hour and weekday
accidents['Year']=accidents['Start_Time'].dt.year
accidents['Month']=accidents['Start_Time'].dt.strftime('%b')
accidents['Day']=accidents['Start_Time'].dt.day
accidents['Hour']=accidents['Start_Time'].dt.hour
accidents['Weekday']=accidents['Start_Time'].dt.strftime('%a')
# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
accidents[td]=round((accidents['End_Time']-accidents['Start_Time'])/np.timedelta64(1,'m'))
accidents.info()
# Check if there is any negative time_duration values
accidents[td][accidents[td]<=0]
# Drop the rows with td<0

neg_outliers=accidents[td]<=0

# Set outliers to NAN
accidents[neg_outliers] = np.nan

# Drop rows with negative td
accidents.dropna(subset=[td],axis=0,inplace=True)
accidents.info()
# Double check to make sure no more negative td
accidents[td][accidents[td]<=0]
# Remove outliers for Time_Duration(min): n * standard_deviation (n=3), backfill with median

n=3

median = accidents[td].median()
std = accidents[td].std()
outliers = (accidents[td] - median).abs() > std*n

# Set outliers to NAN
accidents[outliers] = np.nan

# Fill NAN with median
accidents[td].fillna(median, inplace=True)

accidents.info()
# Print time_duration information
print('Max time to clear an accident: {} minutes or {} hours or {} days; Min to clear an accident td: {} minutes.'.format(accidents[td].max(),round(accidents[td].max()/60), round(accidents[td].max()/60/24), accidents[td].min()))
# Set the list of features to include in Machine Learning
feature_lst=['Source','TMC','Severity','Start_Lng','Start_Lat','Distance(mi)','Side','City','County','State','Timezone','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Hour','Weekday', 'Time_Duration(min)']
# Select the dataset to include only the selected features
accidents_sel=accidents[feature_lst].copy()
accidents_sel.info()
# Check missing values
accidents_sel.isnull().mean()
accidents_sel.dropna(subset=accidents_sel.columns[accidents_sel.isnull().mean()!=0], how='any', axis=0, inplace=True)
accidents_sel.shape
# Set state
state='PA'

# Select the state of Pennsylvania
accidents_statePA=accidents_sel.loc[accidents_sel.State==state].copy()
accidents_statePA.drop('State',axis=1, inplace=True)
accidents_statePA.info()
# Map of accidents, color code by county

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=accidents_statePA, hue='County', legend=False, s=20)
plt.show()
# Set county
county='Montgomery'

# Select the state of Pennsylvania
accidents_countyMO=accidents_statePA.loc[accidents_statePA.County==county].copy()
accidents_countyMO.drop('County',axis=1, inplace=True)
accidents_countyMO.info()
# Map of accidents, color code by city

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=accidents_countyMO, hue='City', legend=False, s=20)
plt.show()
# Set state
state='CA'

# Select the state of California
accidents_stateCA=accidents_sel.loc[accidents_sel.State==state].copy()
accidents_stateCA.drop('State',axis=1, inplace=True)
accidents_stateCA.info()
# Map of accidents, color code by county

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=accidents_stateCA, hue='County', legend=False, s=20)
plt.show()
# Set county
county='Sacramento'

# Select the state of Pennsylvania
accidents_countySA=accidents_stateCA.loc[accidents_stateCA.County==county].copy()
accidents_countySA.drop('County',axis=1, inplace=True)
accidents_countySA.info()
# Map of accidents, color code by city

sns.scatterplot(x='Start_Lng', y='Start_Lat', data=accidents_countySA, hue='City', legend=False, s=20)
plt.show()
# Generate dummies for categorical data
accidents_countyMO_dummy = pd.get_dummies(accidents_countyMO,drop_first=True)
accidents_countyMO_dummy.info()
from sklearn.model_selection import train_test_split
# Assign the data
accidents=accidents_countyMO_dummy

# Set the target for the prediction
target='Severity'


# Create arrays for the features and the response variable

# set X and y
y = accidents[target]
X = accidents.drop(target, axis=1)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
# List of classification algorithms
algo_lst=['Logistic Regression',' K-Nearest Neighbors','Decision Trees','Random Forest','Naive Bayes']

# Initialize an empty list for the accuracy for each algorithm
accuracy_lst=[]
# Import KNeighbors Classifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Import DecisionTree Classifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Import RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

#Import Navie Bayes Classifier
from sklearn.naive_bayes import GaussianNB

# Import SVM Classifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

# Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)


print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))
# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X_train,y_train)

# Predict the labels for the training data X
y_pred = knn.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)

print('[K-Nearest Neighbors (KNN)] knn.score: {:.3f}.'.format(knn.score(X_test, y_test)))
print('[K-Nearest Neighbors (KNN)] accuracy_score: {:.3f}.'.format(acc))
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, n_neighbor in enumerate(neighbors):
    
    # Setup a k-NN Classifier with n_neighbor
    knn = KNeighborsClassifier(n_neighbors=n_neighbor)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
# Decision tree algorithm

# Instantiate dt_entropy, set 'entropy' as the information criterion
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)


# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)

# Use dt_entropy to predict test set labels
y_pred= dt_entropy.predict(X_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)


# Print accuracy_entropy
print('[Decision Tree -- entropy] accuracy_score: {:.3f}.'.format(accuracy_entropy))
# Instantiate dt_gini, set 'gini' as the information criterion
dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)


# Fit dt_entropy to the training set
dt_gini.fit(X_train, y_train)

# Use dt_entropy to predict test set labels
y_pred= dt_gini.predict(X_test)

# Evaluate accuracy_entropy
accuracy_gini = accuracy_score(y_test, y_pred)

# Append to the accuracy list
acc=accuracy_gini
accuracy_lst.append(acc)

# Print accuracy_gini
print('[Decision Tree -- gini] accuracy_score: {:.3f}.'.format(accuracy_gini))
# Random Forest algorithm

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)


# Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))
feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k=20
sns.barplot(x=feature_imp[:20], y=feature_imp.index[:k])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# List top k important features
k=20
feature_imp.sort_values(ascending=False)[:k]
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.03
sfm = SelectFromModel(clf, threshold=0.03)

# Train the selector
sfm.fit(X_train, y_train)

feat_labels=X.columns

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])
# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)
# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

# View The Accuracy Of Our Full Feature Model
print('[Randon forest algorithm -- Full feature] accuracy_score: {:.3f}.'.format(accuracy_score(y_test, y_pred)))

# Apply The Full Featured Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

# View The Accuracy Of Our Limited Feature Model
print('[Randon forest algorithm -- Limited feature] accuracy_score: {:.3f}.'.format(accuracy_score(y_test, y_important_pred)))
#Create a Gaussian Classifier
model= GaussianNB()

#Train the model using the training sets y_pred=model.predict(X_test)
model.fit(X_train,y_train)

# Predicting the Model
y_pred = model.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)

# Model Accuracy, how often is the classifier correct?
print("[Navie Bayes algorithm] accuracy_score: {:.3f}.".format(acc))
# Make a plot of the accuracy scores for different algorithms

# Generate a list of ticks for y-axis
y_ticks=np.arange(len(algo_lst))

# Combine the list of algorithms and list of accuracy scores into a dataframe, sort the value based on accuracy score
accidents_acc=pd.DataFrame(list(zip(algo_lst, accuracy_lst)), columns=['Algorithm','Accuracy_Score']).sort_values(by=['Accuracy_Score'],ascending = True)

# Export to a file
# accidents_acc.to_csv('Accuracy_scores_algorithms_{}.csv'.format(state),index=False)


# Make a plot
ax=accidents_acc.plot.barh('Algorithm', 'Accuracy_Score', align='center',legend=False,color='0.5')


# Add the data label on to the plot
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+0.02, i.get_y()+0.2, str(round(i.get_width(),2)), fontsize=10)

# Set the limit, lables, ticks and title
plt.xlim(0,1.1)
plt.xlabel('Accuracy Score')
plt.yticks(y_ticks, accidents_acc['Algorithm'], rotation=0)
plt.title('[{}-{}] Which algorithm is better?'.format(state, county))

plt.show()
data["Severity"].value_counts()
# plot a histogram  
data['Severity'].hist(bins=10)
# Distribution of accidents according to their severity and when (Day/Night) they occur.
fig,ax = plt.subplots(1,2,figsize=(18,6))

# Pie chart, that shows the distribution of accidents by severity from 1 to 4.
sizes = data.groupby('Severity').size() # serie with severity as index and the number of row of each categorie as values.
sizes = sizes[[2,1,3,4]] # the serie is reordered so that the values are clearly visible in the pie.
labels = 'Severity 2','Severity 1', 'Severity 3', 'severity 4' 
explode = (0.1,0.1,0.1,0.1)
colors = ['red','purple','green','yellow']
ax[0].pie(sizes, explode=explode, labels= labels,colors=colors,autopct='%1.2f%%',shadow=True, startangle=0)
ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax[0].set_title('Accidents by Severity', size=20)

# Seaborn countplot of accident distributions.
sns.countplot(data=data,x='Severity', hue='Sunrise_Sunset',ax=ax[1])
ax[1].set_title('Accidents by Severity and Moment', size= 20)
ax[1].set_xlabel('Severity', size=18)
ax[1].set_ylabel('Number of Accidents', size=18)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Scatterplot', fontsize=22)
ax.plot(data['Severity'], data['Visibility(mi)'], 'ko')

# This scatter plot gives the severity of accidents with respect to the visibility.
fig=sns.heatmap(data[['TMC','Severity','Distance(mi)',
                    'Temperature(F)','Wind_Chill(F)','Humidity(%)',
                    'Pressure(in)','Visibility(mi)','Wind_Speed(mph)']].corr(),
                annot=True,cmap='RdBu',linewidths=0.2,annot_kws={'size':15})
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

#Heatmap is a two-dimensional graphical representation of data where the individual values that are contained in a matrix are represented as colors
# Or we can also say that these Heat maps display numeric tabular data where the cells are colored depending upon the contained value. 
#Heat maps are great for making trends in this kind of data more readily apparent, particularly when the data is ordered and there is clustering.
plt.figure(figsize =(10,5))
data.groupby(['Year']).size().sort_values(ascending=True).plot.bar()
data['time'] = pd.to_datetime(data.Start_Time, format='%Y-%m-%d %H:%M:%S')

plt.subplots(2,2,figsize=(15,10))
for s in np.arange(1,5):
    plt.subplot(2,2,s)
    plt.hist(pd.DatetimeIndex(data.loc[data["Severity"] == s]['time']).month, bins=[1,2,3,4,5,6,7,8,9,10,11,12,13], align='left', rwidth=0.8)
    plt.title("Accident Count by Month with Severity " + str(s), fontsize=14)
    plt.xlabel("Month", fontsize=16)
    plt.ylabel("Accident Count", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
plt.figure(figsize =(15,5))
data.groupby(['Month']).size().plot.bar()
data['DayOfWeek'] = data['time'].dt.dayofweek
plt.subplots(2,2,figsize=(15,10))
for s in np.arange(1,5):
    plt.subplot(2,2,s)
    plt.hist(data.loc[data["Severity"] == s]['DayOfWeek'], bins=[0,1,2,3,4,5,6,7], align='left', rwidth=0.8)
    plt.title("Accident Count by Day with Severity " + str(s), fontsize=16)
    plt.xlabel("Day", fontsize=16)
    plt.ylabel("Accident Count", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
for s in np.arange(1,5):
    plt.subplots(figsize=(12,5))
    data.loc[data["Severity"] == s]['Weather_Condition'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)
    plt.xlabel('Weather Condition',fontsize=16)
    plt.ylabel('Accident Count',fontsize=16)
    plt.title('20 of The Main Weather Conditions for Accidents of Severity ' + str(s),fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
for s in ["Fog","Light Rain","Rain","Heavy Rain","Snow"]:
    plt.subplots(1,2,figsize=(12,5))
    plt.suptitle('Accident Severity Under ' + s,fontsize=16)
    plt.subplot(1,2,1)
    data.loc[data["Weather_Condition"] == s]['Severity'].value_counts().plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)
    plt.xlabel('Severity',fontsize=16)
    plt.ylabel('Accident Count',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.subplot(1,2,2)
    data.loc[data["Weather_Condition"] == s]['Severity'].value_counts().plot.pie(autopct='%1.0f%%',fontsize=16)
factors = ['Temperature(F)','Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

for factor in factors:
    # remove some of the extreme values
    factorMin = data[factor].quantile(q=0.0001)
    factorMax = data[factor].quantile(q=0.9999)
    # print df["Severity"].groupby(pd.cut(df[factor], np.linspace(factorMin,factorMax,num=20))).count()
    plt.subplots(figsize=(15,5))
    for s in np.arange(1,5):
        data["Severity"].groupby(pd.cut(data[factor], np.linspace(factorMin,factorMax,num=20))).mean().plot()
        plt.title("Mean Severity as a Function of " + factor, fontsize=16)
        plt.xlabel(factor + " Range", fontsize=16)
        plt.ylabel("Mean Severity", fontsize=16)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=16)
for s in ['Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']:
    # check if infrastructure type is found in any record 
    if (data[s] == True).sum() > 0:
        plt.subplots(1,2,figsize=(12,5))
        plt.xticks(fontsize=14)
        plt.suptitle('Accident Severity Near ' + s,fontsize=16)
        plt.subplot(1,2,1)
        data.loc[data[s] == True]['Severity'].value_counts().plot.bar(width=0.5,color='y',edgecolor='k',align='center',linewidth=1)
        plt.xlabel('Severity',fontsize=16)
        plt.ylabel('Accident Count',fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.subplot(1,2,2)
        data.loc[data[s] == True]['Severity'].value_counts().plot.pie(autopct='%1.0f%%',fontsize=16)