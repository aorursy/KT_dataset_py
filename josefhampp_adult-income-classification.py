#Start with importing system and operating system libraries
import sys
import os

#Suppress warnings, this will not affect the result
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#Check python version
print('Python version:',sys.version)

#Import pandas and numpy
import pandas as pd
import numpy as np

#Load the data set
data = pd.read_csv('../input/adult.csv', sep=',')
data.head()
import matplotlib.pyplot as plt

#Initialize an empty array to collect the sum of missing values per column
mvs = []

#Count the missing values
for x in data.columns:
    mvs.append(data[x].isin(["?"]).sum())

#Build the plot
fig, ax =  plt.subplots(figsize=(10,3))
index   = np.arange(data.shape[1])

ax.bar(index, mvs, alpha = 0.4, color = 'b')
ax.set_ylabel('Missing Values')
ax.set_xticks(index)
ax.set_xticklabels((data.columns))

fig.tight_layout()

plt.xticks(rotation=45)
plt.show()
#Only three features contain missing values
#To see if the missing values have an significant effect on the dataset
#Visualize the missing values compared with the complete dataset to see the effect

#Build the plot
yvalues = [data.shape[0], mvs[1], mvs[6], mvs[13]]

fig, ax = plt.subplots()
index   = np.arange(4)

ax.bar(index, yvalues, alpha = 0.4, color = 'b')
ax.set_ylabel('Data')
ax.set_xticks(index)
ax.set_xticklabels(('data set size','workclass','occupation','native.country'))

fig.tight_layout()

plt.xticks(rotation=45)
plt.show()
#Question 1: Age

#Set up a histogram
plt.hist(data.age, facecolor='green', alpha=0.5, bins=18, edgecolor='black')
plt.xlabel('Age')
plt.axvline(data.age.mean(), color='red', label='average age')
plt.axis([15, 95, 0, 3500])
plt.legend()
plt.show()
#Question 2: Gender

#Count male and female
m = 0
f = 0
for g in data.sex:
    if g == 'Male':
        m += 1
    if g == 'Female':
        f += 1

#Set up pie chart
colors = ['lightskyblue', 'lightcoral']
values = [m, f] 
labels = ['Male', 'Female'] 
plt.pie(values, labels=labels, colors=colors, shadow=True, startangle=90, autopct='%.2f')
plt.axis('equal')
plt.tight_layout()
plt.show()
#Question 3: Hours per Week

print('Mean:', data['hours.per.week'].mean())
#Set up a histogram
plt.hist(data['hours.per.week'], facecolor='green', alpha=0.5, bins=18, edgecolor='black')
plt.xlabel('Age')
plt.axvline(data['hours.per.week'].mean(), color='red', label='average hour per week')
plt.legend()
plt.show()
#Question 4: Native Country

#Count the countries 
f = data['native.country'].value_counts()

#Show the top 5
f.head()
#Question 5: Income

#Count how often people earn more and less than 50k
m = 0
l = 0
for i in data.income:
    if i == '<=50K':
        l += 1
    if i == '>50K':
        m += 1

#Set up pie chart
colors = ['lightskyblue', 'lightcoral']
values = [l, m] 
labels = ['<=50k', '>50k'] 
plt.pie(values, labels=labels, colors=colors, shadow=True, startangle=90, autopct='%.2f')
plt.axis('equal')
plt.tight_layout()
plt.show()
print('Number of rows with missing values:', data.shape[0],'\n')

#Save the number of size of the dataset before deleting the missing values
numRows = data.shape[0]

#Check if there are missing values in the dataset and delete the data if so
for x in data.columns:
    mv = data[x].isin(["?"]).sum()
    if mv > 0:
        data = data[data[x] != '?']

print('Number of rows without missing values:',data.shape[0])
print('We dropped of',numRows - data.shape[0],'rows')

#Plot a pie chart to visualize the result
colors = ['lightskyblue', 'lightcoral']
patches, texts = plt.pie([data.shape[0], numRows - data.shape[0]], colors=colors, shadow=True, startangle=90)
plt.legend(patches, ['Number of Rows', 'Number of dropped Rows'], loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
#Print the attributes of workclass an their occurrence.
f = data['workclass'].value_counts().reset_index()
f.columns = ['workclass', 'count']
print(f)
#Combine the mentioned attributes
data.workclass = data.workclass.replace({'Self-emp-not-inc': 'Self-emp',
                                        'Self-emp-inc': 'Self-emp',
                                        'Local-gov': 'Gov',
                                        'Federal-gov': 'Gov',
                                        'State-gov': 'Gov'})

#Count all distinct attributes of fnlwgt
f = data['fnlwgt'].value_counts().reset_index()
f.columns = ['fnlwgt', 'count']
print('Number of distinct attribute in column fnlwgt :',f.shape[0])
#Print the attributes of education and their occurrence.
f = data['education'].value_counts().reset_index()
f.columns = ['education', 'count']
print('\n',f)
#Combine the mentioned attributes
data.education = data.education.replace({'Preschool': 'No-school',
                                        '1st-4th': 'No-school',
                                        '5th-6th': 'No-school',
                                        '7th-8th': 'No-school',
                                        '9th': 'No-school',
                                        '10th': 'No-school',
                                        '11th': 'No-school',
                                        '12th': 'No-school',
                                        'Some-college': 'College',
                                        'Assoc-voc': 'College',
                                        'Assoc-acdm': 'College'})

#Print the attributes of education an their occurrence
f = data['marital.status'].value_counts().reset_index()
f.columns = ['marital.status', 'count']
print('\n',f)
#Combine the mentioned attributes
data['marital.status'].replace(['Married-civ-spouse'], 'Married', inplace=True)
data['marital.status'].replace('Never-married', 'Not-married', inplace=True)
data['marital.status'].replace(['Divorced'], 'Separated', inplace=True)
data['marital.status'].replace(['Separated'], 'Separated', inplace=True)
data['marital.status'].replace(['Married-spouse-absent'], 'Not-married', inplace=True)
data['marital.status'].replace(['Married-AF-spouse'], 'Married', inplace=True)

#Show the result
f = data['marital.status'].value_counts().reset_index()
f.columns = ['marital.status', 'count']
print('\n',f)
#Print the attributes of occupation an their occurrence
f = data['occupation'].value_counts().reset_index()
f.columns = ['occupation', 'count']
print('\n',f)

#Print the attributes of relationship an their occurrence
f = data['relationship'].value_counts().reset_index()
f.columns = ['relationship', 'count']
print('\n',f)

#Print the attributes of race an their occurrence
f = data['race'].value_counts().reset_index()
f.columns = ['race', 'count']
print('\n',f)

#Print the attributes of sex an their occurrence
f = data['sex'].value_counts().reset_index()
f.columns = ['sex', 'count']
print('\n',f)
#Print the attributes of native.country an their occurrence
f = data['native.country'].value_counts().reset_index()
f.columns = ['native.country', 'count']
print('\n',f)
data['native.country'].replace(['United-States'], 'N-America', inplace=True)
data['native.country'].replace(['Mexico'], 'N-America', inplace=True)
data['native.country'].replace(['Philippines'], 'Asia', inplace=True)
data['native.country'].replace(['Germany'], 'Europe', inplace=True)
data['native.country'].replace(['Puerto-Rico'], 'N-America', inplace=True)
data['native.country'].replace(['Canada'], 'N-America', inplace=True)
data['native.country'].replace(['India'], 'Asia', inplace=True)
data['native.country'].replace(['El-Salvador'], 'MS-America', inplace=True)
data['native.country'].replace(['Cuba'], 'MS-America', inplace=True)
data['native.country'].replace(['England'], 'Europe', inplace=True)
data['native.country'].replace(['Jamaica'], 'MS-America', inplace=True)
data['native.country'].replace(['Italy'], 'Europe', inplace=True)

data['native.country'].replace(['China'], 'Asia', inplace=True)
data['native.country'].replace(['Dominican-Republic'], 'MS-America', inplace=True)
data['native.country'].replace(['Vietnam'], 'Asia', inplace=True)
data['native.country'].replace(['Guatemala'], 'MS-America', inplace=True)
data['native.country'].replace(['Japan'], 'Asia', inplace=True)
data['native.country'].replace(['Columbia'], 'MS-America', inplace=True)
data['native.country'].replace(['Poland'], 'Europe', inplace=True)
data['native.country'].replace(['Taiwan'], 'Asia', inplace=True)
data['native.country'].replace(['Haiti'], 'MS-America', inplace=True)
data['native.country'].replace(['Iran'], 'Asia', inplace=True)
data['native.country'].replace(['Portugal'], 'Europe', inplace=True)
data['native.country'].replace(['Nicaragua'], 'MS-America', inplace=True)

data['native.country'].replace(['Peru'], 'MS-America', inplace=True)
data['native.country'].replace(['Greece'], 'Europe', inplace=True)
data['native.country'].replace(['Ecuador'], 'MS-America', inplace=True)
data['native.country'].replace(['France'], 'Europe', inplace=True)
data['native.country'].replace(['Ireland'], 'Europe', inplace=True)
data['native.country'].replace(['Hong'], 'Asia', inplace=True)
data['native.country'].replace(['Trinadad&Tobago'], 'MS-America', inplace=True)
data['native.country'].replace(['Cambodia'], 'Asia', inplace=True)
data['native.country'].replace(['Laos'], 'Asia', inplace=True)
data['native.country'].replace(['Thailand'], 'Asia', inplace=True)
data['native.country'].replace(['Yugoslavia'], 'Europe', inplace=True)
data['native.country'].replace(['Outlying-US(Guam-USVI-etc)'], 'N-America', inplace=True)

data['native.country'].replace(['Hungary'], 'Europe', inplace=True)
data['native.country'].replace(['Honduras'], 'MS-America', inplace=True)
data['native.country'].replace(['Scotland'], 'Europe', inplace=True)
data['native.country'].replace(['Holand-Netherlands'], 'Europe', inplace=True)

#Show the result
f = data['native.country'].value_counts().reset_index()
f.columns = ['native.country', 'count']
print('\n',f)
print('dataset size')
print('# of rows:', data.shape[0])
print('# of columns:', data.shape[1])
data.head()
#Drop the columns not needed
data.drop(columns=['education.num', 'fnlwgt', 'relationship'], axis=1, inplace=True)
data.head()
#Import OneHotEncoder, LabelEncoder & MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

#Define how to process which feature
columns_to_label_encode   = ['income']
columns_to_scale          = ['age', 'capital.gain', 'capital.loss', 'hours.per.week']

#Instantiate encoder/scaler
ohe    = OneHotEncoder(sparse=False)
le     = LabelEncoder()
mms    = MinMaxScaler()

#To one hot encode the string values, they need to be in a numeric format,
#To do so, we first label encode those features  
w = np.reshape(le.fit_transform(data['workclass']), (30162, 1))
e = np.reshape(le.fit_transform(data['education']), (30162, 1))
m = np.reshape(le.fit_transform(data['marital.status']), (30162, 1))
o = np.reshape(le.fit_transform(data['occupation']), (30162, 1))
r = np.reshape(le.fit_transform(data['race']), (30162, 1))
s = np.reshape(le.fit_transform(data['sex']), (30162, 1))
n = np.reshape(le.fit_transform(data['native.country']), (30162, 1))

#Concatenate the label encoded features
wemorsn = np.concatenate([w, e, m, o, r, s, n], axis=1)

#Scale and encode separate columns
one_hot_encoded_columns = ohe.fit_transform(wemorsn)
label_encoded_columns   = np.reshape(le.fit_transform(data[columns_to_label_encode]), (30162, 1))
min_max_scaled_columns  = mms.fit_transform(data[columns_to_scale])

#Concatenate again
processed_data = np.concatenate([min_max_scaled_columns, one_hot_encoded_columns, label_encoded_columns], axis=1)

#Turn processed data into DataFrame typ
pd_df = pd.DataFrame(processed_data, index=data.index)

pd_df.head(10)
from sklearn.model_selection import train_test_split

#Split prediction variable and features
X = pd_df.values[:, :-1]
y = pd_df.values[:, -1]

#Split test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import cross_val_score

#Import the classification models
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#Initialize the models
models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LR', LogisticRegression()))
models.append(('RF', RandomForestClassifier()))
models.append(('LSVM', SVC(kernel='linear')))
models.append(('SSVM', SVC(kernel='sigmoid')))
models.append(('RSVM', SVC(kernel='rbf')))

scores = []
names = []

#Set up charts to visualize the results
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(9, 9))
xAxis = 0
yAxis = 0

axes[4, 0].set_title('Mean Accuracy')

print('Mean accuracy on test data:')

#Train and test each model - save the results
for name, model in models:
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_test, y_test, cv=7)
    scores.append(score)
    names.append(name)
    axes[xAxis, yAxis].set_title(name)
    axes[xAxis, yAxis].plot(['1','2','3','4','5','6','7'], score, color='C'+str(len(names)-1))
    axes[xAxis, yAxis].set_xlabel("Validation")
    axes[xAxis, yAxis].set_ylabel("Accuracy")
    axes[4, 0].bar(name, score.mean(), alpha=0.4, color='C'+str(len(names)-1))
    axes[4, 0].set_ylim(0.79, 0.85)
    
    if len(names)%2 == 1:
        yAxis += 1
    else:
        xAxis +=1
        yAxis -=1
    print(name, model.score(X_test, y_test))

#Remove empty chart
axes[4, 1].remove()

fig.tight_layout()
plt.show()
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Construct pipelines
pipe1 = Pipeline((('LR', LogisticRegression()),))
pipe2 = Pipeline((('SVC', SVC()),))
pipe3 = Pipeline((('RF', RandomForestClassifier()),))

#Define parameters for each pipeline
para1 = {
    'LR__penalty' : ['l1', 'l2'],
    'LR__C': [0.01, 0.1, 1.0, 10]
}
para2 = {
    'SVC__C': [0.1, 1.0, 10],
    'SVC__kernel': ['linear', 'sigmoid', 'rbf'],
    'SVC__gamma': [0.01, 0.1, 1.0]
}
para3 = {
    'RF__n_estimators': [5, 10, 50],
    'RF__max_features': ['auto', 'sqrt', 'log2'],
    'RF__max_depth': [1, 5, 10],
    'RF__min_samples_split': [2, 5, 10],
    'RF__min_samples_leaf': [1, 2, 5]
}


paras = [ para1, para2, para3]
pipes = [ pipe1, pipe2, pipe3]


for i in range(len(pipes)):
    print('GridSearch for model', i)
    grid = GridSearchCV(pipes[i], paras[i], verbose=1, refit=False, n_jobs=-1)
    grid = grid.fit(X_train, y_train)
    print('Finished GridSearch\n')
    print('Best score:', grid.best_score_)
    print(grid.best_params_, '\n\n')
#Initialize the model with the evaluated parameters
RF  = RandomForestClassifier(max_depth=10, max_features='auto', min_samples_leaf=2, min_samples_split=10, n_estimators=10)

#Train the models on the whole data set
RF.fit(X, y)

print('Classification model is ready')