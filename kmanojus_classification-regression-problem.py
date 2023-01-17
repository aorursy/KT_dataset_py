
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def removing_specialchar(str):
    return str.replace('?',0)
    
# Importing the dataset
app_train = pd.read_csv('../input/application_train.csv')
app_test=pd.read_csv('../input/application_test.csv')
app_train = app_train.apply(removing_specialchar,axis=1)
app_train.head(10)
app_train.isnull().sum()
print('Training data shape: ', app_train.shape)
app_train.head()
print('Testing data shape: ', app_test.shape)
app_test.head()
app_train['TARGET'].value_counts()
app_train['TARGET'].astype(int).plot.hist();

def missing_values_table(df):
    mis_val = df.isnull().sum()
        
        # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        
    return mis_val_table_ren_columns
    
missing_values = missing_values_table(app_train)
missing_values.head(20)        

#to get unique count of type of data types.
app_train.dtypes.value_counts()
#app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# to get distinct value of categorical variable
app_train.loc[:,app_train.dtypes=='object'].apply(pd.Series.nunique, axis = 0)

##data Analysis/data Distribution
plt.title("Credit Amount of Loan Applicant")
plt.show(app_train['AMT_CREDIT'].plot.hist(bins=100))
plt.title("Income Amt")
plt.show(app_train['AMT_INCOME_TOTAL'].plot.hist(bins=50))
plt.title("Price AMT")
plt.show(app_train['AMT_GOODS_PRICE'].plot.hist(bins=50))
plt.title("Type of Suite")
plt.show(app_train.groupby('NAME_TYPE_SUITE').size().plot(kind='bar'))
plt.show(app_train.groupby('TARGET').size().plot(kind='pie'))
app_train.groupby('NAME_FAMILY_STATUS').size().plot(kind='pie')
plt.title("Income Type")
app_train.groupby('NAME_INCOME_TYPE').size().plot(kind='pie')
plt.title("Making Own Car")
app_train.groupby('FLAG_OWN_CAR').size().plot(kind='pie')
plt.title("Making Own Realty")
app_train.groupby('FLAG_OWN_REALTY').size().plot(kind='pie')
plt.title("Loan Type")
app_train.groupby('NAME_CONTRACT_TYPE').size().plot(kind='bar')
plt.title("Occupation of Applicant")
app_train.groupby('OCCUPATION_TYPE').size().plot(kind='bar')
plt.title("Type of Organization who applied for Loan")
app_train.groupby('ORGANIZATION_TYPE').size().plot(kind='pie')
print('Dimension of data')
print(app_train.shape)
print('Peek at the data')
print(app_train.head(20))
print('Statistical Summary')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

#Label encoder & One hot encoder to get missing value replaced for categorical value
# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

train_labels = app_train['TARGET']
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
app_train['TARGET'] = train_labels
#Finding the anamalies
#AGE Column - Converting them into years
(app_train['DAYS_BIRTH']/-365).describe()
#Employement Days
app_train['DAYS_EMPLOYED'].describe()
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');

anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
#finding the correlation between variable with matching Target variable 1 or 0
import os
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
correlations = app_train.corr()['TARGET'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])  

plt.style.use('fivethirtyeight')
# Plot the distribution of ages in years
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');

plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages'); 

age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)   

age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups       

plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');

#finding the correlation between variable with matching Target variable 1 or 0
import os
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
correlations = app_train.corr()['TARGET'].sort_values()
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])  

plt.style.use('fivethirtyeight')
# Plot the distribution of ages in years
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');

plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages'); 

age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)   

age_groups  = age_data.groupby('YEARS_BINNED').mean()
age_groups       

plt.figure(figsize = (8, 8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');

ext_data = app_train[['TARGET', 'SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH','NAME_EDUCATION_TYPE_Higher education','CODE_GENDER_F','NAME_INCOME_TYPE_Pensioner','DAYS_EMPLOYED','ORGANIZATION_TYPE_XNA','FLOORSMAX_AVG','FLOORSMAX_MEDI','FLOORSMAX_MODE','EMERGENCYSTATE_MODE_No','HOUSETYPE_MODE_block of flats','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH']]
ext_data1 = app_train[['TARGET', 'SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data1.corr()
ext_data_corrs   

ext_test = app_test[['SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH','NAME_EDUCATION_TYPE_Higher education','CODE_GENDER_F','NAME_INCOME_TYPE_Pensioner','DAYS_EMPLOYED','ORGANIZATION_TYPE_XNA','FLOORSMAX_AVG','FLOORSMAX_MEDI','FLOORSMAX_MODE','EMERGENCYSTATE_MODE_No','HOUSETYPE_MODE_block of flats','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH']]

y = ext_data['TARGET']
train = ext_data.drop('TARGET',1)   

plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');   

plt.figure(figsize = (10, 12))

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)    

poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset = pd.read_csv('../input/application_train.csv')
y = app_train['TARGET']
feature_names = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH','NAME_EDUCATION_TYPE_Higher education','CODE_GENDER_F','NAME_INCOME_TYPE_Pensioner','ORGANIZATION_TYPE_XNA','FLOORSMAX_AVG','FLOORSMAX_MEDI','EMERGENCYSTATE_MODE_No','HOUSETYPE_MODE_block of flats','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_EMPLOYED','AMT_ANNUITY','AMT_INCOME_TOTAL','DAYS_REGISTRATION','DAYS_LAST_PHONE_CHANGE','DAYS_ID_PUBLISH']]
X = feature_names
#X=app_train


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func=chi2, k=4)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, y)
print("Num Features: -->",fit.n_features_)
print("Selected Features: %s",fit.support_)
print("Feature Ranking: %s",fit.ranking_)
for idx,i in enumerate(fit.ranking_):
    if i==1:
        print (i,idx)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
train = imputer.fit_transform(train)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

ext_test = app_test[['SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH','NAME_EDUCATION_TYPE_Higher education','CODE_GENDER_F','NAME_INCOME_TYPE_Pensioner','DAYS_EMPLOYED','ORGANIZATION_TYPE_XNA','FLOORSMAX_AVG','FLOORSMAX_MEDI','FLOORSMAX_MODE','EMERGENCYSTATE_MODE_No','HOUSETYPE_MODE_block of flats','AMT_GOODS_PRICE','REGION_POPULATION_RELATIVE','DAYS_BIRTH']]
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
train = imputer.fit_transform(train)
imputer = Imputer(strategy = 'median')
test = imputer.fit_transform(ext_test)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Logistic regression output :->')
print("\n")
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))

y_pred = logreg.predict(X_test)
log_reg_pred = logreg.predict_proba(X_test)[:, 1]
submit = ext_test[['SK_ID_CURR']]
#submit['TARGET'] = log_reg_pred*100
#submit['TARGET_1']= np.where(log_reg_pred > .5,1,0)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("Confusion metrics of Logistic regression- :", cm)
print("\n")
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print("\n")
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\n")
print("Confusion metrics of Decision Tree- :", cm)
print("\n")

print('KNN regression output :->')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

print("\n")
y_pred = knn.predict(X_test)
print("\n")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print("\n")

print('LinearDiscriminat Analysis regression output :->')
print("\n")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'
     .format(lda.score(X_test, y_test)))

y_pred = lda.predict(X_test)

# Making the Confusion Matrix
print("\n")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\n")
print(cm)
print("\n")

print('Support Vector Classification output :->')
print("\n")
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

y_pred = svm.predict(X_test)
print("\n")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\n")
print(cm)
print("\n")
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

print('KNN regression output :->')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

print("\n")
y_pred = knn.predict(X_test)
print("\n")
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print("\n")

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
#import keras
from keras.models import Sequential

from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 20))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#model.compile(loss='mean_squared_error', optimizer='adam')

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 500, nb_epoch = 500)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(test)
#log_reg_pred = logreg.predict_proba(test)[:, 1]
#submit = ext_test[['SK_ID_CURR']]
#submit['TARGET'] = log_reg_pred*100
#submit['TARGET_1']= np.where(log_reg_pred > .5,1,0)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)





        
   







