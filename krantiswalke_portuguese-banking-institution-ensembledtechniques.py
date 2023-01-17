# To enable plotting graphs in Jupyter notebook
%matplotlib inline

# Importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# importing ploting libraries
import matplotlib.pyplot as plt   

#importing seaborn for statistical plots
import seaborn as sns

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

import numpy as np
#import os,sys
from scipy import stats

# calculate accuracy measures and confusion matrix
from sklearn import metrics
datapath = '../input'
my_data = pd.read_csv(datapath+'/bank-full.csv')
my_data.head(10)
my_data.shape
my_data.columns
my_data.dtypes
val=my_data.isnull().values.any()

if val==True:
    print("Missing values present : ", my_data.isnull().values.sum())
    my_data=my_data.dropna()
else:
    print("No missing values present")
#null values
my_data.isnull().values.any()
my_data.describe().T
my_data.info()
my_data.apply(lambda x: len(x.unique()))
print('Jobs:\n',my_data['job'].unique())
print('Marital:\n',my_data['marital'].unique())
print('Default:\n',my_data['default'].unique())
print('Education:\n',my_data['education'].unique())
print('Housing:\n',my_data['housing'].unique())
print('Loan:\n',my_data['loan'].unique())
print('Contact:\n',my_data['contact'].unique())
print('Month:\n',my_data['month'].unique())
print('Day:\n',my_data['day'].unique())
print('Campaign:\n',my_data['campaign'].unique())
#Find Mean
my_data.mean()
#Find Median
my_data.median()
#Find Standard Deviation
my_data.std()
my_data.skew(axis = 0, skipna = True) 
my_data.hist(figsize=(10,10),color="blueviolet",grid=False)
plt.show()
sns.pairplot(my_data.iloc[:,1:])
print('Min age: ', my_data['age'].max())
print('Max age: ', my_data['age'].min())
plt.figure(figsize = (30,12))
sns.countplot(x = 'age',  palette="rocket", data = my_data)
plt.xlabel("Age", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Age Distribution', fontsize=15)
sns.boxplot(x = 'age', data = my_data, orient = 'v')
plt.ylabel("Age", fontsize=15)
plt.title('Age Distribution', fontsize=15)
sns.distplot(my_data['age'])
plt.xlabel("Age", fontsize=15)
plt.ylabel('Occurence', fontsize=15)
plt.title('Age x Ocucurence', fontsize=15)
# Quartiles
print('1º Quartile: ', my_data['age'].quantile(q = 0.25))
print('2º Quartile: ', my_data['age'].quantile(q = 0.50))
print('3º Quartile: ', my_data['age'].quantile(q = 0.75))
print('4º Quartile: ', my_data['age'].quantile(q = 1.00))
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Ages above: ', my_data['age'].quantile(q = 0.75) + 
                      1.5*(my_data['age'].quantile(q = 0.75) - my_data['age'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', my_data[my_data['age'] > 70.5]['age'].count())
print('Number of clients: ', len(my_data))
#Outliers in %
print('Outliers are:', round(my_data[my_data['age'] > 70.5]['age'].count()*100/len(my_data),2), '%')
plt.figure(figsize = (30,12))
sns.countplot(x = 'job',data = my_data)
plt.xlabel("job", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Job Distribution', fontsize=20)
#plt.figure(figsize = (30,12))
sns.countplot(x = 'marital',data = my_data)
plt.xlabel("Marital", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Marital Distribution', fontsize=15)
sns.boxplot(x='marital',y='age',hue='Target',data=my_data)
#plt.figure(figsize = (30,12))
sns.countplot(x = 'education',data = my_data)
plt.xlabel("Education", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Education Distribution', fontsize=15)
sns.boxplot(x='education',y='age',hue='Target',data=my_data)
#plt.figure(figsize = (30,12))
sns.countplot(x = 'default',data = my_data)
plt.xlabel("Default", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Default Distribution', fontsize=15)
sns.boxplot(x='default',y='age',hue='Target',data=my_data)
print('Default:\n No credit in default:'     , my_data[my_data['default'] == 'no']     ['age'].count(),
              '\n Yes to credit in default:' , my_data[my_data['default'] == 'yes']    ['age'].count())
#plt.figure(figsize = (30,12))
sns.countplot(x = 'housing',data = my_data)
plt.xlabel("Housing", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Housing Distribution', fontsize=15)
print('Housing:\n No Housing:'     , my_data[my_data['housing'] == 'no']     ['age'].count(),
              '\n Yes Housing:' , my_data[my_data['housing'] == 'yes']    ['age'].count())
sns.boxplot(x='housing',y='age',hue='Target',data=my_data)
#plt.figure(figsize = (30,12))
sns.countplot(x = 'loan',data = my_data)
plt.xlabel("Loan", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Loan Distribution', fontsize=15)
print('Loan:\n No Personal loan:'     , my_data[my_data['loan'] == 'no']     ['age'].count(),
              '\n Yes Personal Loan:' , my_data[my_data['loan'] == 'yes']    ['age'].count())
sns.boxplot(x='loan',y='age',hue='Target',data=my_data)
#plt.figure(figsize = (30,12))
sns.countplot(x = 'contact',data = my_data)
plt.xlabel("Contact", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Contact Distribution', fontsize=15)
print('Contact:\n Unknown Contact:'     , my_data[my_data['contact'] == 'unknown']     ['age'].count(),
              '\n Cellular Contact:'   , my_data[my_data['contact'] == 'cellular']    ['age'].count(),
              '\n Telephone Contact:'  , my_data[my_data['contact'] == 'telephone']   ['age'].count())
#plt.figure(figsize = (30,12))
sns.countplot(x = 'month',data = my_data)
plt.xlabel("In which Month was a person contacted", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Monthly Distribution', fontsize=15)
sns.boxplot(x=my_data["day"])
sns.boxplot(x=my_data["duration"])
sns.distplot(my_data['duration'])
plt.xlabel("duration", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Duration distribution', fontsize=15)
# Quartiles
print('1º Quartile: ', my_data['duration'].quantile(q = 0.25))
print('2º Quartile: ', my_data['duration'].quantile(q = 0.50))
print('3º Quartile: ', my_data['duration'].quantile(q = 0.75))
print('4º Quartile: ', my_data['duration'].quantile(q = 1.00))
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Duration above: ', my_data['duration'].quantile(q = 0.75) + 
                      1.5*(my_data['duration'].quantile(q = 0.75) - my_data['duration'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', my_data[my_data['duration'] > 643.0]['duration'].count())
print('Number of clients: ', len(my_data))
#Outliers in %
print('Outliers are:', round(my_data[my_data['duration'] > 643.0]['duration'].count()*100/len(my_data),2), '%')
# Look, if the call duration is iqual to 0, then is obviously that this person didn't subscribed, 
# THIS LINES NEED TO BE DELETED LATER 
my_data[(my_data['duration'] == 0)]
my_data[my_data['duration'] == 0]['duration'].count()
plt.figure(figsize = (30,12))
sns.countplot(x = 'campaign', data = my_data)
plt.xlabel("Campaign", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Campaign Distribution', fontsize=15)
sns.boxplot(x = 'campaign', data = my_data, orient = 'v')
plt.ylabel("Campaign", fontsize=15)
plt.title('Campaign Distribution', fontsize=15)
sns.distplot(my_data['campaign'])
plt.xlabel("Campaign", fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Campaign distribution', fontsize=15)
# Quartiles
print('1º Quartile: ', my_data['campaign'].quantile(q = 0.25))
print('2º Quartile: ', my_data['campaign'].quantile(q = 0.50))
print('3º Quartile: ', my_data['campaign'].quantile(q = 0.75))
print('4º Quartile: ', my_data['campaign'].quantile(q = 1.00))
  # Interquartile range, IQR = Q3 - Q1
  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR 
  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR
    
print('Campaign above: ', my_data['campaign'].quantile(q = 0.75) + 
                      1.5*(my_data['campaign'].quantile(q = 0.75) - my_data['campaign'].quantile(q = 0.25)), 'are outliers')
print('Numerber of outliers: ', my_data[my_data['campaign'] > 6.0]['campaign'].count())
print('Number of clients: ', len(my_data))
#Outliers in %
print('Outliers are:', round(my_data[my_data['campaign'] > 6.0]['campaign'].count()*100/len(my_data),2), '%')
sns.boxplot(x='campaign',y='age',hue='Target',data=my_data)
sns.boxplot(x = 'pdays', data = my_data, orient = 'v')
plt.ylabel("pdays", fontsize=15)
plt.title('pdays Distribution', fontsize=15)
sns.boxplot(x = 'previous', data = my_data, orient = 'v')
plt.ylabel("Previous", fontsize=15)
plt.title('Previous', fontsize=15)
sns.countplot(x = 'poutcome', data = my_data, orient = 'v')
plt.ylabel("Poutcome", fontsize=15)
plt.title('Poutcome distribution', fontsize=15)
print('poutcome:\n Unknown poutcome:'     , my_data[my_data['poutcome'] == 'unknown']   ['age'].count(),
              '\n Failure in  poutcome:'  , my_data[my_data['poutcome'] == 'failure']   ['age'].count(),
              '\n Other poutcome:'        , my_data[my_data['poutcome'] == 'other']     ['age'].count(),
              '\n Success in poutcome:'   , my_data[my_data['poutcome'] == 'success']   ['age'].count())
sns.boxplot(x='poutcome',y='age',hue='Target',data=my_data)
my_data.boxplot(by = 'Target',  layout=(4,4), figsize=(20, 20))
sns.countplot(x = 'Target', data = my_data, orient = 'v')
plt.ylabel("Target", fontsize=15)
plt.title('Target distribution', fontsize=15)
#Let us look at the target column which is "Target"(yes/no).
my_data.groupby(["Target"]).count()
cor=my_data.corr()
cor
plt.subplots(figsize=(10,8))
sns.heatmap(cor,annot=True)


# Label encoder order in alphabetical
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
my_data['job']      = labelencoder_X.fit_transform(my_data['job']) 
my_data['marital']  = labelencoder_X.fit_transform(my_data['marital']) 
my_data['education']= labelencoder_X.fit_transform(my_data['education']) 
my_data['default']  = labelencoder_X.fit_transform(my_data['default']) 
my_data['housing']  = labelencoder_X.fit_transform(my_data['housing']) 
my_data['loan']     = labelencoder_X.fit_transform(my_data['loan']) 

my_data['contact']     = labelencoder_X.fit_transform(my_data['contact']) 
my_data['month']       = labelencoder_X.fit_transform(my_data['month']) 
#function to creat group of ages, this helps because we have 78 differente values here
def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
           
    return dataframe

age(my_data);
my_data.head()
print(my_data.shape)
my_data.head()
def duration(data):

    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data
duration(my_data);
my_data.head()
my_data.loc[(my_data['pdays'] == 999), 'pdays'] = 1
my_data.loc[(my_data['pdays'] > 0) & (my_data['pdays'] <= 10), 'pdays'] = 2
my_data.loc[(my_data['pdays'] > 10) & (my_data['pdays'] <= 20), 'pdays'] = 3
my_data.loc[(my_data['pdays'] > 20) & (my_data['pdays'] != 999), 'pdays'] = 4 
my_data.head()
my_data['poutcome'].replace(['unknown', 'failure','other', 'success'], [1,2,3,4], inplace  = True)
print(my_data.shape)
my_data.head()
Final_data=my_data
print(Final_data.shape)
Final_data.head()
Final_data.head()
#from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
X = Final_data.values[:,0:15]  ## Features
Y = Final_data.values[:,16]  ## Target.values[:,10]  ## Target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 7)
clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
NB=accuracy_score(Y_test, Y_pred, normalize = True) #Accuracy of Naive Bayes' Model
print('Accuracy_score:',NB)
print('Confusion_matrix of NB:')
print(metrics.confusion_matrix(Y_test,Y_pred))
final_data = Final_data[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']]
final_data.shape
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_std = pd.DataFrame(StandardScaler().fit_transform(final_data))
X_std.columns = final_data.columns
#split the dataset into training and test datasets
import numpy as np
from sklearn.model_selection import train_test_split

# Transform data into features and target
X = np.array(my_data.iloc[:,1:16]) 
y = np.array(my_data['Target'])

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)
print(y_train.shape)
# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

#Neighbors
neighbors = np.arange(0,25)

for k in neighbors:
    k_value = k+1
    knn = KNeighborsClassifier(n_neighbors = k_value)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    

myList = list(range(1,30))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))
ac_scores = []

# perform accuracy metrics for values from 1,3,5....19
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # predict the response
    y_pred = knn.predict(X_test)
    # evaluate accuracy
    scores = accuracy_score(y_test, y_pred)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
#Plot misclassification error vs k (with k value on X-axis) using matplotlib.
import matplotlib.pyplot as plt
# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
#Use k=23 as the final model for prediction
knn = KNeighborsClassifier(n_neighbors = 23)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_pred = knn.predict(X_test)

# evaluate accuracy
KNN=accuracy_score(y_test, y_pred)   #Accuracy of KNN model
print('Accuracy_score:',KNN)    
print('Confusion_matrix:')
print(metrics.confusion_matrix(y_test, y_pred))
array = my_data.values
X = array[:,0:16] # select all rows and first 16 columns which are the attributes
Y = array[:,16]   # select all rows and the 17th column which is the classification "yes", "no"
test_size = 0.30 # taking 70:30 training and test set
seed = 15  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed) # To set the random state
type(X_train)
# Fit the model on 30%
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
LR = model.score(X_test, y_test)
print('Accuracy:',LR)
print('confusion_matrix:')
print(metrics.confusion_matrix(y_test, y_predict))
A=LR  # Accuracy of Logistic regression model
# Decision tree in Python can take only numerical / categorical colums. It cannot take string / obeject types. 
# The following code loops through each column and checks if the column type is object then converts those columns
# into categorical with each distinct value becoming a category or code.

for feature in my_data.columns: # Loop through all columns in the dataframe
    if my_data[feature].dtype == 'object': # Only apply for columns with categorical strings
        my_data[feature] = pd.Categorical(my_data[feature]).codes # Replace strings with an integer
my_data.info()
train_char_label = ['No', 'Yes']
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer  #DT does not take strings as input for the model fit step....

# splitting data into training and test set for independent attributes
from sklearn.model_selection import train_test_split

X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=.30, random_state=1)

# splitting data into training and test set for independent attributes in the ratio of 70:30 
n=my_data['Target'].count()
train_set = my_data.head(int(round(n*0.7))) # Up to the last initial training set row
test_set = my_data.tail(int(round(n*0.3))) # Past the last initial training set row

# capture the target column ("Target") into separate vectors for training set and test set
train_labels = train_set.pop("Target")
test_labels = test_set.pop("Target")
# invoking the decision tree classifier function. Using 'entropy' method of finding the split columns. Other option 
# could be gini index.  Restricting the depth of the tree to 5 (no particular reason for selecting this)

#dt_model = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 5, random_state = 100)
                                  
dt_model = DecisionTreeClassifier(criterion = 'entropy' )
dt_model.fit(train_set, train_labels)
#Print the accuracy of the model & print the confusion matrix
dt_model.score(test_set , test_labels)
test_pred = dt_model.predict(test_set)
print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = train_set.columns))#Print the feature importance of the decision model
y_predict = dt_model.predict(test_set)
print(dt_model.score(train_set , train_labels))
print(dt_model.score(test_set , test_labels))
print(metrics.confusion_matrix(test_labels, y_predict))
reg_dt_model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 7)
reg_dt_model.fit(train_set, train_labels)
print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = train_set.columns))

y_predict = reg_dt_model.predict(test_set)
DTC=reg_dt_model.score(test_set , test_labels)
print(DTC)
print(metrics.confusion_matrix(test_labels, y_predict))
from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(base_estimator=dt_model, n_estimators=50)

#bgcl = BaggingClassifier(n_estimators=50)
bgcl = bgcl.fit(train_set, train_labels)

y_predict = bgcl.predict(test_set)

BGC=bgcl.score(test_set , test_labels)
print(BGC)

print(metrics.confusion_matrix(test_labels, y_predict))
from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(base_estimator=dt_model, n_estimators=10)
#abcl = AdaBoostClassifier( n_estimators=50)
abcl = abcl.fit(train_set, train_labels)

y_predict = abcl.predict(test_set)

ADE=abcl.score(test_set , test_labels)
print(ADE)

print(metrics.confusion_matrix(test_labels, y_predict))
from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 50)
gbcl = gbcl.fit(train_set, train_labels)
y_predict = gbcl.predict(test_set)
GBC=gbcl.score(test_set , test_labels)
print(GBC)
print(metrics.confusion_matrix(test_labels, y_predict))
from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 50)
rfcl = rfcl.fit(train_set, train_labels)
y_predict = rfcl.predict(test_set)
RFC=rfcl.score(test_set , test_labels)
print(RFC)
print(metrics.confusion_matrix(test_labels, y_predict))

models = pd.DataFrame({
                'Models': [ 'Gausian NB','K-Near Neighbors','Logistic Model', 'Decision Tree Classifier',
                            'Bagging Classifier ', 'Adaboost Ensemble ','GradientBoost Classifier ', 'Random Forest Classifier'],
                'Score':  [NB, KNN, LR, DTC, BGC, ADE, GBC, RFC]})

models.sort_values(by='Score', ascending=False)
