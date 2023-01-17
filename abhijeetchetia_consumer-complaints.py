# Importing modules needed for processing



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.options.mode.chained_assignment = None
train = "../input/edureka/Edureka_Consumer_Complaints_train.csv"

test = "../input/edureka/Edureka_Consumer_Complaints_test.csv"
# train = str(input("Enter the address and the name of the train file:-\n"))

# test = str(input("Enter the address of the name of the test file:-\n"))
# Reading dataset from the system



train_data = pd.read_csv(train, parse_dates=['Date received', 'Date sent to company'])

test_data = pd.read_csv(test, parse_dates=['Date received', 'Date sent to company'])



main_train_data = train_data # Saving the main train data in different variable just in case we need in future

main_test_data = test_data # Saving the main test data in different variable just in case we need in future
train_data.head(5) # Looking the raw train data
test_data.tail(5) # Looking the test data
test_data.shape
# Looking for total missing values in each columns the train data

train_miss = train_data.isnull().sum()

train_miss.to_frame().transpose()
# Looking for total missing values in the test data

test_miss = test_data.isnull().sum()

test_miss.to_frame().transpose()
'''

Converting the strings in "Consumer complaint narrative" to lower case. There are various reasons this is done like the user

is frustrated so he/she might have written the complaints in block letters or maybe it is his/hers preferred writting style. 

But, in most cases they serves no purpose while operating.

'''



train_data['Consumer complaint narrative'] = train_data['Consumer complaint narrative'].str.lower()

test_data['Consumer complaint narrative'] = test_data['Consumer complaint narrative'].str.lower()



# Seperating categorical values



train_cat_data = train_data.select_dtypes(include='object')

test_cat_data = test_data.select_dtypes(include='object')
# Here we can see how many unique values are there in the categorical dataset

# This code will perform operation for the training dataset



train_cat_data.nunique().to_frame().transpose()
# Here we can see how many unique values are there in the categorical dataset

# This code will perform operation for the testing dataset



test_cat_data.nunique().to_frame().transpose()
# Python code to display the most frequent categorical values in training data



train_cat_data.mode()
# Python code to display the most frequent categorical values in test data



test_cat_data.mode()
train_cat_data['Issue'].value_counts().head(10)
index = train_cat_data['Issue'].value_counts().index

values = train_cat_data['Issue'].value_counts().values

plt.figure(figsize=(15, 8))

ax = plt.axes()

ax.tick_params(width='1')

ax.set_facecolor("#7cb7e22f")

plt.bar(index, values, label='Issues', color='#12a4d9')

plt.title('Issues raised by customers\n', fontsize=15, fontstyle='italic')

plt.xlabel('\nIssues', fontsize=13)

plt.ylabel('Frequency\n', fontsize=13)

plt.xticks(rotation=90)

plt.legend()

plt.show()
index = train_cat_data['Issue'].value_counts().index[0:10]

values = train_cat_data['Issue'].value_counts().values[0:10]

plt.figure(figsize=(10, 10))

ax = plt.axes()

ax.set_facecolor('#f5f0e1')

plt.barh(index, values, label='Issues', color='#ffc13b')

plt.yticks(fontstyle='italic')

plt.ylabel('Issues\n', fontsize=13)

plt.xticks(rotation=90, fontsize=12)

plt.xlabel('\nFrequency', fontsize=13)

plt.title('Number of individual issues\n', fontsize=15, fontstyle='italic')

plt.legend()

plt.show()
train_cat_data['Product'].value_counts().to_frame().transpose()
product_data = train_cat_data[train_cat_data['Product'] == 'Mortgage']
issues = product_data['Issue'].value_counts().index

freq = product_data['Issue'].value_counts().values

issue_counts = pd.DataFrame({'Issues': issues, 'Frequency': freq})

issue_counts
ax = plt.axes()

ax.set_facecolor('#e1dd72')

plt.bar(train_cat_data.groupby(['Product']).size().index, 

        train_cat_data.groupby(['Product']).size().values, label='Products', color='#1b6535')

plt.xlabel('\nProduct', fontsize=12)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel('Frequency\n', fontsize=12)

plt.title('Bar graph of products with highest complaints\n', fontsize=15, fontstyle='italic')

plt.legend()

plt.show()
company = train_cat_data['Company'].value_counts().index

num_of_complaints = train_cat_data['Company'].value_counts().values

company_df = pd.DataFrame({'Company': company, 'Number of complaints': num_of_complaints})

company_df.head(10)
train_cat_data['Submitted via'].value_counts().to_frame()

# Here we can see there are 6 methods by which the complaints have been submitted which can be easily visualized in a Pie-chart
slices = train_cat_data['Submitted via'].value_counts().values

medium = train_cat_data['Submitted via'].value_counts().index

colours = ['#daf2dc', '#ffcce7', '#81b7d2', '#4d5198', '#e75874', '#5c3c92']

explode = (0.06, 0, 0.4, 0, 0.09, 0.5)

plt.pie(slices, labels=medium, explode=explode, colors=colours, shadow=True, autopct='%.1f%%', radius=3)

plt.legend()

plt.show()
state = train_cat_data.groupby(['State']).size().index

count = train_cat_data.groupby(['State']).size().values

plt.figure(figsize=(20, 40))

ax = plt.axes()

ax.set_facecolor('#e5e5dc')

plt.barh(state, count, color='#26495c')

plt.title('Figure to show geographical distributions\n', fontsize=15)

plt.xlabel('\nCounts', fontsize=15)

plt.ylabel('States\n', fontsize=15)

plt.show()
pd.set_option('display.max_columns', None)

state_count = pd.DataFrame(dict(States=state, Counts=count)).transpose()

state_count
# Let's make a new dataset for this operation

time_data = pd.DataFrame({})

time_data['Date'] = train_data['Date received']

time_data['Month Name'] = train_data['Date received'].dt.month_name()

time_data['Month Number'] = train_data['Date received'].dt.month

time_data['Day Name'] = train_data['Date received'].dt.day_name()

time_data['Day Number'] = train_data['Date received'].dt.day

time_data['Year'] = train_data['Date received'].dt.year

#time_data['Month/Year'] = time_data['Month Number'].map(str) + '-' + time_data['Year'].map(str)

#time_data['Month/Year'] = pd.to_datetime(time_data['Month/Year']).dt.to_period('M')

time_data
month_counts = time_data.groupby(['Month Number']).size()

months = month_counts.index

counts = month_counts.values

plt.figure(figsize=(12, 6))

ax = plt.axes()

ax.set_facecolor('#aed6dc')

plt.bar(months, counts, label='Complaints', color='#6883bc')

plt.grid(color='#141414')

plt.title('Checking rise and fall of complaints with respect to months\n', fontsize=15)

plt.xticks(months, fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('\nMonths', fontsize=15)

plt.ylabel('Complaints\n', fontsize=15)

plt.legend(fontsize=12)

plt.show()
format_date = time_data.groupby(['Date']).size().to_frame('Counts')

format_date

#month_years = pd.Series(format_months.index)

#complaint_counts = pd.Series(format_months.values)

#temp_month_df = pd.DataFrame(dict(Month_group = month_years, Complaints = complaint_counts))
fm = format_date['Counts'].resample('W').mean()
plt.figure(figsize=(20,10))

ax = plt.axes()

ax.set_facecolor('#32c8f312')

fm.plot(color='#1868ae')

plt.grid()

plt.grid(which='minor', color='#d72631', alpha=0.2)

plt.grid(which='major', color='#3a6b35', alpha=0.8)

plt.xlabel('\nTime line ->', fontsize='15')

plt.ylabel('Rate of complaints\n', fontsize='15')

plt.show()
diff = month_counts.diff()

diff # Finding the difference between number of complaints with respect to month
day_counts = time_data.groupby(['Day Name']).size()

# Making a temporary dataframe

day_df = pd.DataFrame()

day_df['Day'] = day_counts.index

day_df['Counts'] = day_counts.values

day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

day_df['Day'] = pd.Categorical(day_df['Day'], categories=day_name, ordered=True)

day_df = day_df.sort_values('Day')

day_df
plt.figure(figsize=(10, 4))

plt.barh(day_df['Day'], day_df['Counts'], label='Complaints', color='#316879')

plt.title('Complaints with respect to day of the week\n', fontsize=15, fontstyle='italic')

plt.xlabel('\nDay', fontsize=13)

plt.xticks(rotation=90, fontsize=12)

plt.ylabel('Complaints\n', fontsize=15)

plt.yticks(fontsize=12, fontstyle='italic')

plt.legend(fontsize=12)

plt.show()
response_counts = train_cat_data['Company response to consumer'].value_counts()

response = response_counts.index

counts = response_counts.values

plt.figure(figsize=(15, 5))

ax = plt.axes()

plt.barh(response, counts, color='#d2601a') # Color name is Orange

plt.grid(color='#1d3c45')# Color name is Pine Green

ax.set_facecolor('#fff1e1') # Color name is Light Peach

plt.xlabel('\nCounts', fontsize=12)

plt.ylabel('Responses\n', fontsize=12)

plt.yticks(rotation=20, fontsize='13')

plt.title('Count of responses to complaints\n', fontsize=17)

plt.show()
# Making a temporary dataset

response = pd.DataFrame()

response['Timely Response'] = train_cat_data['Timely response?']

response['Dispute'] = train_cat_data['Consumer disputed?']

response['States'] = train_cat_data['State']

response
# Check how many timely responses were observed

t_res = response['Timely Response'].value_counts()

t_res
# Check how many time customer disputed

c_dis = response['Dispute'].value_counts()

c_dis
sns.countplot(x='Timely Response', data=response, palette='hls')

plt.show()
sns.countplot(x='Dispute', data=response, palette='hls')

plt.show()
# Checking how many time customer did not dispute after timely response



nd_check1 = len(response[(response['Timely Response'] == 'Yes') & (response['Dispute'] == 'No')])

nd_check1
# Checking how many times customer disputed after timely response



nd_check2 = len(response[(response['Timely Response'] == 'Yes') & (response['Dispute'] == 'Yes')])

nd_check2
# Checking how many times customer disputed after no timely response



nd_check3 = len(response[(response['Timely Response'] == 'No') & (response['Dispute'] == 'Yes')])

nd_check3
# Checking how many times customer did not dispute after no timely response



nd_check4 = len(response[(response['Timely Response'] == 'No') & (response['Dispute'] == 'No')])

nd_check4
(nd_check1 / (nd_check1 + nd_check2)) * 100
(nd_check4 / (nd_check3 + nd_check4)) * 100
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split as tts
# Building a dataset by taking only the important columns we need for the operation.



raw_text_df = train_cat_data[['Product', 'Consumer complaint narrative']]

raw_text_df.head(5)
# Removing missing values

text_df = raw_text_df.dropna(axis=0).reset_index()

text_df.head(5)
len(text_df)
# Create a new column 'category_id' with encoded categories 

text_df['Product_id'] = text_df['Product'].factorize()[0]

product_id_df = text_df[['Product', 'Product_id']].drop_duplicates()

product_id_df
len(product_id_df)
text_df.head(20)
X_data = text_df['Consumer complaint narrative']

y_data = text_df['Product']

X_train, X_test, y_train, y_test = tts(X_data, y_data, test_size=0.25, random_state=5)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)

tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)
model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
model.score(tfidf_vectorizer_vectors, y_train)
new_complaint = """The bank obtained the property through a foreclosure. To the best of my knowledge any efforts to date to sell the property have been by an online auction process. This process has not allowed for interior inspection prior to the auction bid and could be one of the reasons why the minimum amount required by the Bank has not resulted in a sale. The home has remained vacant since XXXX, XXXX. The Bank is not maintaining the property. As President of the XXXX XXXX in which this property is located and as a home owner who lives XXXX doors from this house, I 'm particularly concerned about the vacant status and lack of maintenance of this home. A letter was sent to JP Morgan Chase XXXX/XXXX/XXXX requesting they work with the XXXX XXXX to retain a local Real Estate Agent who could market the property for them. A reply was requested by XXXX/XXXX/XXXX. To date ( XXXX/XXXX/XXXX ) we have received no response. 

"""

print(model.predict(fitted_vectorizer.transform([new_complaint])))
new_complaint = """Impersonated an attorney or official"""

print(model.predict(fitted_vectorizer.transform([new_complaint])))
new_complaint = """Not given enough info to verify debt"""

print(model.predict(fitted_vectorizer.transform([new_complaint])))
new_complaint = "PayPal is randomly sending PayPal account holders debit cards without permission ( unless permission is buried in fine print ). Recipients of the card do n't know what it is, why we are receiving it, if it was fraudulently requested, etc. -- then we have to spend 20 minutes holding for PayPal customer service only to find out that PayPal randomly sends the cards to account holders whether or not they request it. It is absolutely ridiculous and PayPal should know better."

print(model.predict(fitted_vectorizer.transform([new_complaint])))
# Let us have a quick look at the train data

main_train_data.head(2)
main_train_data.isnull().sum().to_frame().transpose()
processed_data = main_train_data
fill_columns = ['Sub-product', 'Sub-issue', 'Company public response', 'Consumer consent provided?', 'State', 'ZIP code']

processed_data[fill_columns] = processed_data[fill_columns].fillna('Not Available', axis=1)
processed_data.head(5)
processed_data = processed_data.drop(['Tags', 'Consumer complaint narrative', 'Complaint ID'], axis=1)
processed_data.head(2)
# processed_data['Day received'] = processed_data['Date received'].dt.day

processed_data['Day received name'] = processed_data['Date received'].dt.day_name()

processed_data['Week received'] = processed_data['Date received'].dt.week

processed_data['Year received'] = processed_data['Date received'].dt.year

# processed_data['Day sent to company'] = processed_data['Date sent to company'].dt.day

processed_data['Day name sent to company'] = processed_data['Date sent to company'].dt.day_name()

processed_data['Week sent to company'] = processed_data['Date sent to company'].dt.week

processed_data['Year sent to company'] = processed_data['Date sent to company'].dt.year

processed_data['Time interval'] = (processed_data['Date sent to company'] - processed_data['Date received']).dt.days
# Dropping Date Columns

processed_data2 = processed_data.drop(['Date received', 'Date sent to company'], axis=1)
processed_data2.dtypes.to_frame().transpose()
# we will use LabelEncoder as lb as stated above

object_cols = processed_data2.select_dtypes(include='object').columns.tolist()

object_cols # Filtering out the columns which have categorical data
objects = ['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company public response', 'Company', 'State',

           'ZIP code', 'Consumer consent provided?', 'Submitted via', 'Company response to consumer',

           'Timely response?', 'Day received name', 'Day name sent to company']

dispute = 'Consumer disputed?'
objects
# We will drop the Consumer complaint narrative column

# First we will drop the unwanted columns

processed_test_data = test_data.drop(['Tags', 'Consumer complaint narrative', 'Complaint ID'], axis=1)

# Filling missing values with string "Not Available"

fill_columns = ['Sub-product', 'Sub-issue', 'Company public response', 'Consumer consent provided?', 'State', 'ZIP code']

processed_test_data[fill_columns] = processed_data[fill_columns].fillna('Not Available', axis=1)
# processed_test_data['Day received'] = processed_test_data['Date received'].dt.day

processed_test_data['Day received name'] = processed_test_data['Date received'].dt.day_name()

processed_test_data['Week received'] = processed_test_data['Date received'].dt.week

processed_test_data['Year received'] = processed_test_data['Date received'].dt.year

# processed_test_data['Day sent to company'] = processed_test_data['Date sent to company'].dt.day

processed_test_data['Day name sent to company'] = processed_test_data['Date sent to company'].dt.day_name()

processed_test_data['Week sent to company'] = processed_test_data['Date sent to company'].dt.week

processed_test_data['Year sent to company'] = processed_test_data['Date sent to company'].dt.year

processed_test_data['Time interval'] = (processed_test_data['Date sent to company'] - processed_test_data['Date received']).dt.days
# Dropping Date Columns

processed_test_data2 = processed_test_data.drop(['Date received', 'Date sent to company'], axis=1)
from sklearn.preprocessing import LabelEncoder

column_label = LabelEncoder()

for column in objects:

    processed_data2[column] = column_label.fit_transform(processed_data2[column])

    processed_test_data2[column] = column_label.fit_transform(processed_test_data2[column])

processed_data2[dispute] = column_label.fit_transform(processed_data2[dispute])
processed_data2
final_data = processed_data2
y = final_data['Consumer disputed?']

X = final_data.drop('Consumer disputed?', axis=1)
y.value_counts()
# Using Smote process

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_new, y_new = sm.fit_sample(X, y)
print("Original X shape:- ", X.shape, "\nNew X shape:- ", X_new.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report
kf = KFold(n_splits=5, shuffle=True)
predictors = X_new.columns.tolist()

predictors
# Comparing single model using different features via. KFold

# First we will check for the whole dataset before splitting



X1 = X_new[['Product', 'Issue', 'Company', 'State', 'Submitted via', 'Timely response?', 

                  'Timely response?', 'Time interval']].values

X2 = X_new[['Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company', 'State', 'Submitted via',

                  'Company response to consumer', 'Timely response?', 'Time interval']].values

X3 = X_new[predictors].values

y = y_new.values
'''# Let us do cross validation for the decision tree



def score_model(X, y, kf):

    accuracy_scores = []

    precision_scores = []

    recall_scores = []

    f1_scores = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        dt = DecisionTreeClassifier()

        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))

        precision_scores.append(precision_score(y_test, y_pred))

        recall_scores.append(recall_score(y_test, y_pred))

        f1_scores.append(f1_score(y_test, y_pred))

    print("Accuracy: ", np.mean(accuracy_scores))

    print("Precision: ", np.mean(precision_scores))

    print("Recall: ", np.mean(recall_scores))

    print("F1 Score: ", np.mean(f1_scores), "\n")



print("Decision tree for split X1: ")

score_model(X1, y, kf)

print("Decision tree for split X2: ")

score_model(X2, y, kf)

print("Decision tree for split X3: ")

score_model(X3, y, kf)

'''
'''

# Let us do cross validation for the random forest



def score_model(X, y, kf):

    accuracy_scores = []

    precision_scores = []

    recall_scores = []

    f1_scores = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        rfc = RandomForestClassifier()

        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))

        precision_scores.append(precision_score(y_test, y_pred))

        recall_scores.append(recall_score(y_test, y_pred))

        f1_scores.append(f1_score(y_test, y_pred))

    print("Accuracy: ", np.mean(accuracy_scores))

    print("Precision: ", np.mean(precision_scores))

    print("Recall: ", np.mean(recall_scores))

    print("F1 Score: ", np.mean(f1_scores), "\n")



print("Random forest for split X1: ")

score_model(X1, y, kf)

print("Random forest for split X2: ")

score_model(X2, y, kf)

print("Random forest for split X3: ")

score_model(X3, y, kf)

'''
model_X_train, model_X_test, model_y_train, model_y_test = tts(X_new, y_new, test_size=0.25, random_state=2)
rfc_model = RandomForestClassifier(n_estimators=100, random_state=10)

rfc_model.fit(model_X_train, model_y_train)
rfc_model.score(model_X_train, model_y_train)
y_pred = rfc_model.predict(model_X_test)
print("Accuracy Score:- ", accuracy_score(model_y_test, y_pred))

print("R2 Score- ", r2_score(model_y_test, y_pred))

print("Precision:- ", precision_score(model_y_test, y_pred))

print("Recall:- ", recall_score(model_y_test, y_pred))

print("F1 score:- ", f1_score(model_y_test, y_pred))

print("\nConfusion Matrix:- \n", confusion_matrix(model_y_test, y_pred))
ft_imp = pd.Series(rfc_model.feature_importances_, index=[predictors]).sort_values(ascending=False)
ft_imp
final_X_data = X_new.drop(['Timely response?'], axis=1)

final_y_data = y_new
final_X_data.columns.tolist()
from sklearn.model_selection import GridSearchCV # Importing grid search
'''

estimators = [10, 30, 100, 150, 200]

param_grid = dict(n_estimators=estimators)

param_grid # Selecting parameters

'''
# rfc_temp_model = RandomForestClassifier(random_state=10)

# grid = GridSearchCV(rfc_temp_model, param_grid, cv=5)
'''

grid_X_data = final_X_data.sample(10000, replace=True, random_state=2)

grid_y_data = final_y_data.sample(10000, replace=True, random_state=2)

'''
# grid.fit(grid_X_data, grid_y_data)
'''

print("Best Parameter: ", grid.best_params_)

scores = grid.cv_results_['mean_test_score']

print(scores)

'''
'''

# Elbow Graph

plt.figure(figsize=(10, 4))

plt.plot(estimators, scores)

plt.xlabel("\nn_estimators")

plt.ylabel("accuracy\n")

plt.xlim(100, 300)

plt.grid()

plt.grid(which='minor', color='#d72631', alpha=0.2)

plt.title("Elbow graph for Grid Search CV")

plt.show()

'''
final_X_train, final_X_test, final_y_train, final_y_test = tts(final_X_data, final_y_data, test_size=0.2, random_state=2)
final_model = RandomForestClassifier(n_estimators=170, random_state=10)

final_model.fit(final_X_train, final_y_train)
final_model.score(final_X_data, final_y_data)
y_pred = final_model.predict(final_X_test)
print("Accuracy:- ", accuracy_score(final_y_test, y_pred))

print("Precision:- ", precision_score(final_y_test, y_pred))

print("F1 score:- ",  f1_score(final_y_test, y_pred))

print("Recall:- ", recall_score(final_y_test, y_pred))

print("Confusion matrix:-\n", confusion_matrix(final_y_test, y_pred))
final_test_data = processed_test_data2.drop('Timely response?', axis=1)

final_test_data
final_y_pred = final_model.predict(final_test_data)
final_y_pred.shape
pd.Series(final_y_pred).to_csv('Final predictions.csv', index=False, header=None)