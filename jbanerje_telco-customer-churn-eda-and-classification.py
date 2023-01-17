#importing the libraries

#Data Processing Libraries
import numpy as np
import pandas as pd

#Data Vizualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Pretty display for notebooks
%matplotlib inline

# Machine Learning Library
from sklearn.preprocessing import LabelEncoder # Encode Categorical Variable to Numerical Variable
from sklearn.preprocessing import Imputer # Imputer Class to replace missing values
from sklearn.metrics import confusion_matrix # Library for model evaluation
from sklearn.metrics import accuracy_score # Library for model evaluation
from sklearn.model_selection import train_test_split # Library to split datset into test and train

from sklearn.linear_model  import LogisticRegression # Logistic Regression Classifier
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent Classifier
from sklearn.tree import DecisionTreeClassifier # Decision Tree Classifier
from sklearn.ensemble  import RandomForestClassifier # Random Forest Classifier
from sklearn.neighbors import KNeighborsClassifier # K Nearest neighbors Classifier
from sklearn.naive_bayes import GaussianNB #Naive Bayes Classifier
from sklearn.svm import SVC #Support vector Machine Classifier
from sklearn.ensemble import AdaBoostClassifier # Ada Boost Classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

#Ignoring the warnings
import warnings
warnings.filterwarnings('ignore')
# Read .csv file from location and load into pandas DataFrame
datset_churn = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
# Keeping a backup of original datset.Always a good practice
datset_churn_copy = datset_churn.copy()
datset_churn.shape  # output = (rows, columns)
# Getting the column names
datset_churn.columns.values
# Renaming the 3 columns.
datset_churn = datset_churn.rename(columns={'customerID' : 'CustomerID' , 'gender': 'Gender', 'tenure':'Tenure'})
print(datset_churn.columns.values)
datset_churn.info()
datset_churn.head()
datset_churn['TotalChargesNum']=pd.to_numeric(datset_churn['TotalCharges'])
#Identifying the rows containing missing data
missing_value_row = list(datset_churn[datset_churn['TotalCharges'] == " "].index)
print('Missing Value Rows-->', missing_value_row , '\nTotal rows-->', len(missing_value_row))
# Replacing the spaces with 0
for missing_row in missing_value_row :
    datset_churn['TotalCharges'][missing_row] = 0
# Let's try to convert it back to Numeric
datset_churn['TotalCharges']=pd.to_numeric(datset_churn['TotalCharges'])
datset_churn.info()
datset_churn.head() # This will print first 5 rows in pandas dataset.
datset_churn.describe(include=['O'])
#Creating the list of columns
datset_churn_column = list(datset_churn.columns)

#Removing numerical columns & CustomerID
datset_churn_column.remove('CustomerID')
datset_churn_column.remove('SeniorCitizen')
datset_churn_column.remove('Tenure')
datset_churn_column.remove('MonthlyCharges')
datset_churn_column.remove('TotalCharges')

# Printing Unique values in each categorical column
for col in datset_churn_column:
    print(col, "-", datset_churn[col].unique())
datset_churn.describe()
print("Assess missing values in dataset")
total = datset_churn.isnull().sum().sort_values(ascending=False)
percent = (datset_churn.isnull().sum()/datset_churn.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)
datset_churn[['MonthlyCharges','Tenure','TotalCharges']].head()
#Identifying the rows containing 0 value in Total Charges
zero_value_row = list(datset_churn[datset_churn['TotalCharges'] == 0].index)
print('0 Value Rows-->', missing_value_row , '\nTotal rows-->', len(missing_value_row))
# Replacing the spaces with 0
for zero_row in zero_value_row :
    datset_churn['TotalCharges'][zero_row] = datset_churn['Tenure'][zero_row] * datset_churn['MonthlyCharges'][zero_row]
#Validating the data
for zero_row in zero_value_row :
    print( datset_churn['MonthlyCharges'][zero_row],datset_churn['Tenure'][zero_row],datset_churn['TotalCharges'][zero_row])
# Getting the list of all columns
columns_hist = list(datset_churn.columns)

#Removing the Numerical Variables
columns_hist.remove('CustomerID')
columns_hist.remove('SeniorCitizen')
columns_hist.remove('Tenure')
columns_hist.remove('MonthlyCharges')
columns_hist.remove('TotalCharges')

#Creating Column into 4X4 matrix to display 16 bar charts in 4X4 form:
columns_hist_nparray = np.array(columns_hist)
columns_hist_nparray = np.reshape(columns_hist_nparray, (4,4)) # reshaping the columns into 4X4 matrix
# Plotting the bar chart
rows = 4 ; columns = 4
f, axes = plt.subplots(rows, columns, figsize=(20, 20))
print('Univariate Analysis of each categorical Variables')
for row in range(rows):
    for column in range(columns):
        sns.countplot(datset_churn[columns_hist_nparray[row][column]], palette = "Set1", ax = axes[row, column])
print('Univariate Analysis of each numerical Variables')
f, axes = plt.subplots(2, 3, figsize=(20,10))
#Charting the histogram
datset_churn["Tenure"].plot.hist(color='DarkBlue', alpha=0.7, bins=50, title='Tenure',ax=axes[0, 0])
datset_churn["MonthlyCharges"].plot.hist(color='DarkBlue', alpha=0.7, bins=50, title='MonthlyCharges',ax=axes[0, 1])
datset_churn["TotalCharges"].plot.hist(color='DarkBlue', alpha=0.7, bins=50, title='TotalCharges',ax=axes[0, 2])

#Charting the density plot
sns.distplot( datset_churn["Tenure"] , kde=True, rug=False, color="skyblue", ax=axes[1, 0])
sns.distplot( datset_churn["MonthlyCharges"] , kde=True, rug=False, color="olive", ax=axes[1, 1])
sns.distplot( datset_churn["TotalCharges"] , kde=True, rug=False, color="gold", ax=axes[1, 2])
sns.countplot(datset_churn['SeniorCitizen'], palette = "Set1")
f, axes = plt.subplots(1, 3, figsize=(15,5))
sns.boxplot(x=datset_churn["Tenure"], orient="v", color="olive",ax=axes[0])
sns.boxplot(x=datset_churn["MonthlyCharges"], orient="v", color="gold",ax=axes[1])
sns.boxplot(x=datset_churn["TotalCharges"] , orient="v", color="skyblue",ax=axes[2])
# Converting the categorical variable to numerical variable
datset_churn['Churn_Num'] = datset_churn['Churn'].map( {'Yes': 1, 'No': 0} ).astype(int)
# Validating the mappaing
datset_churn[['Churn','Churn_Num']].head()
# Plotting Tenure Column with Churn
# Churn_num indicates customer who left the company. 0 indicates customer who stayed.
fighist = sns.FacetGrid(datset_churn, col='Churn_Num')
fighist.map(plt.hist, 'Tenure', bins=20) 
# Plotting MonthlyCharges Column with Churn
# Churn_num indicates customer who left the company. 0 indicates customer who stayed.
fighist = sns.FacetGrid(datset_churn, col='Churn_Num')
fighist.map(plt.hist, 'MonthlyCharges', bins=20)
# Plotting TotalCharges Column with Churn
# Churn_num indicates customer who left the company. 0 indicates customer who stayed.
fighist = sns.FacetGrid(datset_churn, col='Churn_Num')
fighist.map(plt.hist, 'TotalCharges', bins=20)
col_list = columns_hist
col_list.remove('Churn')
for col in col_list:
    if col == 'PaymentMethod':
        aspect_ratio = 2.0
    else:
        aspect_ratio = 0.8
        
    plot_cat_data = sns.catplot(x=col, col='Churn_Num', data = datset_churn, kind='count', height=4, aspect=aspect_ratio)
# Creating tenure band and co-relation with Churn
datset_churn['TenureRange'] = pd.cut(datset_churn['Tenure'], 5)
datset_churn[['TenureRange', 'Churn_Num']].groupby(['TenureRange'], as_index=False).mean().sort_values(by='TenureRange', ascending=True)

# Replacing Age band with ordinals based on these bands
datset_churn.loc[ datset_churn['Tenure'] <= 8, 'TenureCat'] = 0
datset_churn.loc[(datset_churn['Tenure'] > 8) & (datset_churn['Tenure'] <= 15), 'TenureCat'] = 1
datset_churn.loc[(datset_churn['Tenure'] > 15) & (datset_churn['Tenure'] <= 30), 'TenureCat'] = 2
datset_churn.loc[(datset_churn['Tenure'] > 30) & (datset_churn['Tenure'] <= 45 ), 'TenureCat'] = 3
datset_churn.loc[(datset_churn['Tenure'] > 45) & (datset_churn['Tenure'] <= 60 ), 'TenureCat'] = 4
datset_churn.loc[ datset_churn['Tenure'] > 60, 'TenureCat'] = 5

datset_churn[['Tenure','TenureRange','TenureCat']].head(10)
# Creating MonthlyCharges Band and co-relation with Churn
datset_churn['MonthlyChargesRange'] = pd.cut(datset_churn['MonthlyCharges'], 5)
datset_churn[['MonthlyChargesRange', 'Churn_Num']].groupby(['MonthlyChargesRange'], as_index=False).mean().sort_values(by='MonthlyChargesRange', ascending=True)

# Replacing Age band with ordinals based on these bands
datset_churn.loc[ datset_churn['MonthlyCharges'] <= 20, 'MonthlyChargesCat'] = 0
datset_churn.loc[(datset_churn['MonthlyCharges'] > 20) & (datset_churn['MonthlyCharges'] <= 40), 'MonthlyChargesCat'] = 1
datset_churn.loc[(datset_churn['MonthlyCharges'] > 40) & (datset_churn['MonthlyCharges'] <= 60), 'MonthlyChargesCat'] = 2
datset_churn.loc[(datset_churn['MonthlyCharges'] > 60) & (datset_churn['MonthlyCharges'] <= 80 ), 'MonthlyChargesCat'] = 3
datset_churn.loc[(datset_churn['MonthlyCharges'] > 80) & (datset_churn['MonthlyCharges'] <= 100 ), 'MonthlyChargesCat'] = 4
datset_churn.loc[ datset_churn['MonthlyCharges'] > 100, 'MonthlyChargesCat'] = 5

#Checking the categories
datset_churn[['MonthlyCharges','MonthlyChargesRange','MonthlyChargesCat']].head(10)
#Creating a new column for family. If a customer has dependant or Partner, I am considering it as family .
list_family = []
for rows in range(len(datset_churn['Partner'])):
    if ((datset_churn['Partner'][rows] == 'No') and (datset_churn['Dependents'][rows] == 'No')):
        list_family.append('No')
    else:
        list_family.append('Yes')
datset_churn['Family'] = list_family
#print(datset_churn[['Partner', 'Dependents', 'Family' ]].head(10))

#Creating a new column for Online Services (Online Security & Online Backup) . If a customer has Online Security or Online Backup services
#then , I am considering it as "Yes" else "No"
list_online_services = []
for rows_os in range(len(datset_churn['OnlineSecurity'])):
    if ((datset_churn['OnlineSecurity'][rows_os] == 'No') and (datset_churn['OnlineBackup'][rows_os] == 'No')):
        list_online_services.append('No')
    else:
        list_online_services.append('Yes')
datset_churn['OnlineServices'] = list_online_services

#print(datset_churn[['OnlineSecurity', 'OnlineBackup', 'OnlineServices' ]].head(10))
 
#Creating a new column for Streaming Services (StreamingTV & StreamingMovies) . If a customer has StreamingTV or StreamingMovies
#then , I am considering it as "Yes" else "No"
list_streaming_services = []
for rows_stv in range(len(datset_churn['StreamingTV'])):
    if ((datset_churn['StreamingTV'][rows_stv] == 'No') and (datset_churn['StreamingMovies'][rows_stv] == 'No')):
        list_streaming_services.append('No')
    else:
        list_streaming_services.append('Yes')
datset_churn['StreamingServices'] = list_streaming_services

#print(datset_churn[['StreamingTV', 'StreamingMovies', 'StreamingServices' ]].head(10))

plot_cat_data = sns.catplot(x='Family', col='Churn_Num', data = datset_churn, kind='count', height=4, aspect=0.8)
plot_cat_data = sns.catplot(x='OnlineServices', col='Churn_Num', data = datset_churn, kind='count', height=4, aspect=0.8)
plot_cat_data = sns.catplot(x='StreamingServices', col='Churn_Num', data = datset_churn, kind='count', height=4, aspect=0.8)
datset_churn.info()
#Converting Gender column to numeric value
#datset_churn['Gender'].unique() # Print unique values in the column
datset_churn['Gender_Num'] = datset_churn['Gender'].map( {'Female': 1, 'Male': 0} ).astype(int) #Map Categorical to Numerical Values
datset_churn[['Gender','Gender_Num']].head(2) # Test the mapping
# For Partner & Dependant , we created Family Column . Converting Family column to numeric value
#datset_churn['Family'].unique() # Print unique values in the column
datset_churn['Family_Num'] = datset_churn['Family'].map( {'Yes': 1, 'No': 0} ).astype(int) #Map Categorical to Numerical Values
datset_churn[['Family','Family_Num']].head(2) # Test the mapping
datset_churn['PhoneService_Num'] = datset_churn['PhoneService'].map( {'Yes': 1, 'No': 0} ).astype(int)
datset_churn['MultipleLines_Num'] = datset_churn['MultipleLines'].map( {'No': 0, 'Yes': 1, 'No phone service':2} ).astype(int)
datset_churn['InternetService_Num'] = datset_churn['InternetService'].map( {'DSL': 0, 'Fiber optic': 1, 'No':2} ).astype(int)
datset_churn['OnlineServices_Num'] = datset_churn['OnlineServices'].map( {'Yes': 1, 'No': 0} ).astype(int)

datset_churn['DeviceProtection_Num'] = datset_churn['DeviceProtection'].map( {'No': 0, 'Yes': 1, 'No internet service':2} ).astype(int)
datset_churn['StreamingServices_Num'] = datset_churn['StreamingServices'].map( {'Yes': 1, 'No': 0} ).astype(int)
datset_churn['TechSupport_Num'] = datset_churn['TechSupport'].map( {'No': 0, 'Yes': 1, 'No internet service':2} ).astype(int)
datset_churn['Contract_Num'] = datset_churn['Contract'].map( {'Month-to-month': 0, 'One year': 1, 'Two year': 2} ).astype(int)
datset_churn['PaperlessBilling_Num'] = datset_churn['PaperlessBilling'].map( {'Yes': 1, 'No': 0} ).astype(int)
datset_churn['PaymentMethod_Num'] = datset_churn['PaymentMethod'].map( {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2 , 'Credit card (automatic)' : 3} ).astype(int)
datset_churn.info()
# Take a copy of dataset
datset_churn_copy = datset_churn.copy()
#Dropping the Categorical columns and keeping their equivalent numeric column
columns_to_drop = ['Gender', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn', 'Family', 'OnlineServices', 'StreamingServices']
datset_churn = datset_churn.drop(columns_to_drop, axis=1)

#Re-arranging the columns as per origial dataset
datset_churn = datset_churn[['CustomerID', 'Gender_Num', 'SeniorCitizen', 'Family_Num', 'TenureCat', 'PhoneService_Num', 'MultipleLines_Num', 'InternetService_Num', 'OnlineServices_Num', 'DeviceProtection_Num', 'TechSupport_Num', 'StreamingServices_Num', 'Contract_Num', 'PaperlessBilling_Num', 'PaymentMethod_Num', 'MonthlyChargesCat', 'Churn_Num']]
datset_churn = datset_churn.rename(columns={'Gender_Num' : 'Gender', 
                             'Family_Num' : 'Family',
                             'PhoneService_Num' : 'PhoneService',
                             'MultipleLines_Num': 'MultipleLines', 
                             'InternetService_Num' : 'InternetService', 
                             'OnlineServices_Num' : 'OnlineServices', 
                             'DeviceProtection_Num' : 'DeviceProtection',
                             'TechSupport_Num' : 'TechSupport', 
                             'StreamingServices_Num' : 'StreamingServices', 
                             'Contract_Num' : 'Contract', 
                             'PaperlessBilling_Num' : 'PaperlessBilling', 
                             'PaymentMethod_Num' : 'PaymentMethod', 
                             'MonthlyCharges' : 'MonthlyCharges', 
                             'Churn_Num' :  'Churn' })
datset_churn.info()
datset_churn.head(10) # Taking a quick look into the new data
X = datset_churn.iloc[:,1:16].values # Feature Variable
y = datset_churn.iloc[:,16].values # Target Variable

#Dividing data into test & train splitting 70% data for training anf 30% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))
#Creating function for Confusion Matrix , Precsion, Recall and F1 Score
def plot_confusion_matrix(classifier, y_test, y_pred_test):
    cm = confusion_matrix(y_test, y_pred_test)
    
    print("\n",classifier,"\n")
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap='RdBu')
    classNames = ['Churn-No','Churn-Yes']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]), 
                     horizontalalignment='center', color='White')
    
    plt.show()
        
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = 2*recall*precision/(recall+precision)

    print('Recall={0:0.3f}'.format(recall),'\nPrecision={0:0.3f}'.format(precision))
    print('F1={0:0.3f}'.format(F1))
    return;
from sklearn.metrics import average_precision_score, precision_recall_curve
def plot_prec_rec_curve(classifier, y_test, y_pred_score):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_score)
    average_precision = average_precision_score(y_test, y_pred_score)

    print('Average precision-recall score: {0:0.3f}'.format(
          average_precision))

    plt.plot(recall, precision, label='area = %0.3f' % average_precision, color="green")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="best")
    plt.show()
# Making a list of all classifiers
classifier_model = [LogisticRegression(),KNeighborsClassifier(),GaussianNB(),SVC(),DecisionTreeClassifier(),RandomForestClassifier(), SGDClassifier(), AdaBoostClassifier()]

# Creating empty list to store the performance details
classifier_model_list= []
classifier_accuracy_test = []
classifier_accuracy_train = []
f1score = []
precisionscore = []
recallscore = []
avg_pre_rec_score = []
cv_score = []

for classifier_list in classifier_model:
    classifier = classifier_list
 
    # Fitting the training set into classification model
    classifier.fit(X_train,y_train)
    
    # Predicting the output on test datset
    y_pred_test = classifier.predict(X_test)    
    score_test = accuracy_score(y_test, y_pred_test)
    
    # Predicting the output on training datset
    y_pred_train = classifier.predict(X_train) 
    score_train = accuracy_score(y_train, y_pred_train)
    
    # Cross Validation Score on training test
    scores = cross_val_score(classifier, X_train,y_train, cv=10)
    cv_score.append(scores.mean())
    
    #Keeping the model and accuracy score into a list
    classifier_model_list.append(classifier_list.__class__.__name__)
    classifier_accuracy_test.append(round(score_test,4))
    classifier_accuracy_train.append(round(score_train,4))
    
    #Precision, Recall and F1 score
    f1score.append(f1_score(y_test, y_pred_test))
    precisionscore.append(precision_score(y_test, y_pred_test))
    recallscore.append(recall_score(y_test, y_pred_test))
    
    #Calculating Average Precision Recall Score
    try:
        y_pred_score = classifier.decision_function(X_test)
    except:
        y_pred_score = classifier.predict_proba(X_test)[:,1]
    
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_pred_score)
    avg_pre_rec_score.append(average_precision)
    
    
    #Confusion Matrix
    plot_confusion_matrix(classifier_list.__class__.__name__, y_test, y_pred_test)
    plot_prec_rec_curve(classifier_list.__class__.__name__, y_test, y_pred_score)
#Creating pandas dataframe with Model and corresponding accuracy
#accuracy_df = pd.DataFrame({'Model':classifier_model_list , 'Test Accuracy':classifier_accuracy_test, 'Train Accuracy' :classifier_accuracy_train , 'Precision':precisionscore, 'Recall':recallscore ,'F1 Score':f1score},index=None)
accuracy_df = pd.DataFrame({'Model':classifier_model_list , 'Cross Val Score':cv_score, 'Test Accuracy' :classifier_accuracy_train , 'Precision':precisionscore, 'Recall':recallscore ,'Avg Precision Recall':avg_pre_rec_score ,'F1 Score':f1score})

# Calculating Average Accuracy = (Test + Train)/2
accuracy_df['Average_Accuracy'] =  (accuracy_df['Cross Val Score'] + accuracy_df['Test Accuracy'] )/ 2

#Arranging the Columns
print("\n*------------------------------    CLASSIFICATION MODEL PERFORMANCE EVALUATION      ---------------------*\n")
accuracy_df = accuracy_df[['Model','Cross Val Score', 'Test Accuracy', 'Average_Accuracy','Precision', 'Recall','Avg Precision Recall','F1 Score']]  # This will arrange the columns in the order we want

#Sorting the Columns based on Average Accuracy
accuracy_df.sort_values('Average_Accuracy', axis=0, ascending=False, inplace=True) # Sorting the data with highest accuracy in the top
accuracy_df
#accuracy_df.transpose()
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.linear_model  import LogisticRegression # Logistic Regression Classifier

#Logistic Regression Classifier
clf = LogisticRegression()

#Hyperparameters
parameters = {'C':np.logspace(0, 4, 10), 
              'penalty' : ['l1', 'l2']
             }

# Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score,beta=0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# View best hyperparameters
#print(grid_srchfit.best_params_)

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print ("Unoptimized model\n------")
print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print ("\nOptimized Model\n------")
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print (best_clf)
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score

# TODO: Initialize the classifier
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# TODO: Create the parameters list you wish to tune
parameters = {'n_estimators':[50, 120], 
              'learning_rate':[0.1, 0.5, 1.],
              'base_estimator__min_samples_split' : np.arange(2, 8, 2),
              'base_estimator__max_depth' : np.arange(1, 4, 1)
             }

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score,beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train,y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print ("Unoptimized model\n------")
print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print ("\nOptimized Model\n------")
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print (best_clf)
# Feature Importance for Adaboost
from sklearn.feature_selection import RFE
features = list(datset_churn.columns[1:16])

# Feature Importance for AdaBoostClassifier
adboost_cls = AdaBoostClassifier()
adboost_cls .fit(X_train, y_train)
feature_imp_adboost = np.round(adboost_cls.feature_importances_, 5)

feature_imp_df = pd.DataFrame({'Features' :features, 'Adaboost_Score': feature_imp_adboost})
feature_imp_df.sort_values('Adaboost_Score', axis=0, ascending=False, inplace=True)
print(feature_imp_df)
dataset_churn_new = datset_churn[['MonthlyChargesCat', 'TenureCat', 'Contract', 'InternetService', 'MultipleLines', 'PaymentMethod', 'Churn']]
X_new = dataset_churn_new.iloc[:,:-1].values # Feature Variable
y_new = dataset_churn_new.iloc[:,-1].values # Target Variable

#Dividing data into test & train splitting 80% data for training and 20% for test
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new , y_new, test_size=0.20)
print('There are {} samples in the training set and {} samples in the test set'.format(X_train_new.shape[0], X_test_new.shape[0]))
#Adaboost Classifier , filled the hyperparameter from the Grid Search
classifier = AdaBoostClassifier(algorithm='SAMME.R',
          base_estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=1,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=0.5, n_estimators=120, random_state=None)
 
# Fitting the training set into classification model
classifier.fit(X_train_new, y_train_new)
    
# Predicting the output on test datset
y_pred_new = classifier.predict(X_test_new)    

try:
    y_pred_new_score = classifier.decision_function(X_test_new)
except:
    y_pred_new_score = classifier.predict_proba(X_test_new)[:,1]
    
#Confusion Matrix and Precision Recall Curve
plot_confusion_matrix('Adaboost Classifier', y_test_new, y_pred_new)
plot_prec_rec_curve('Adaboost Classifier', y_test_new, y_pred_new_score)
#Logistic Regression , filled the hyperparameter from the Grid Search
classifier_logreg = LogisticRegression(C=2.7825594022071245, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
 
# Fitting the training set into classification model
classifier_logreg.fit(X_train_new, y_train_new)
    
# Predicting the output on test datset
y_pred_new = classifier_logreg.predict(X_test_new)    

try:
    y_pred_new_score = classifier_logreg.decision_function(X_test_new)
except:
    y_pred_new_score = classifier_logreg.predict_proba(X_test_new)[:,1]
    
#Confusion Matrix and Precision Recall Curve
plot_confusion_matrix('Logistic Regression', y_test_new, y_pred_new)
plot_prec_rec_curve('Logistic Regression', y_test_new, y_pred_new_score)