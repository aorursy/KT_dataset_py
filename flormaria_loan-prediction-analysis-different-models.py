import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/LoanStats3a-2007-2011.csv', low_memory=False)
df = df.drop( ['url','mths_since_last_delinq', 'mths_since_last_record','next_pymnt_d','pymnt_plan', 'desc','initial_list_status', 'out_prncp',

                         'out_prncp_inv','total_rec_int', 'total_rec_late_fee', 'recoveries','next_pymnt_d','earliest_cr_line','pub_rec',

              'inq_last_6mths','open_acc','last_pymnt_d','last_credit_pull_d','issue_d','revol_bal','total_rec_prncp',

              'collection_recovery_fee','issue_d','title','total_pymnt_inv','emp_title'], axis = 1)
df.head()
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()
sns.catplot(x='home_ownership', col='loan_status', kind='count', data=df);
sns.boxplot(x='loan_amnt', y='loan_status', data=df)

plt.title("Loan Status distribution as function of Loan Amount")

plt.show()
sns.catplot(x='emp_length', col='grade', kind='count', data=df);
sns.boxplot(x='loan_amnt', y='emp_length', data=df)

plt.title("Loan Status distribution as function of Loan Amount")

plt.show()
sns.catplot(data = df, x = "emp_length", y = "loan_amnt", hue = "loan_status")
#Fill missing data with mode

def mode_fillna(dataset, column_name):

    dataset[column_name].fillna( dataset[column_name].mode().iloc[0],inplace = True)

    return dataset



#Fill missing data in the emp_title column with the word Others

def emp_title_fillna(dataset):

    dataset['emp_title'].fillna('Others',inplace = True)

    return dataset



#Fill missing data with 0 for the columns:

#revol_util, last_pymnt_amnt, emp_length, total_credit_used, years_experience, delinq_2yrs,total_acc

#We decided to fill with cero because if we fill with the mean or mode we can affect the analysis

def column_fillna(dataset, column_name):

    dataset[column_name] = dataset[column_name].fillna(0)

    return dataset;



#import uuid 

#def generate_id(dataset):

#    dataset['id'] = uuid.uuid1()

#    return dataset



def drop_features(dataset,column_name):

    return dataset.drop([column_name], axis = 1)



#transform the column in months

def get_month_load(dataset):

    dataset = dataset.rename(index=str, columns={"term": "months_pay_loan"})

    new = dataset["months_pay_loan"].str.split(" ", n = 2, expand = True) 

    dataset["months_pay_loan"]= new[1]   

    return dataset



#transform the column in number

def get_value(dataset, actual_column, new_column, separator):

    dataset = dataset.rename(index=str, columns={actual_column: new_column})

    new = dataset[new_column].str.split(separator, n = 1, expand = True) 

    dataset[new_column]= new[0] 

    return dataset



#transform the column in number

def get_years_experience(dataset):

    dataset = dataset.rename(index=str, columns={"emp_length": "years_experience"})

    dataset['years_experience'] = dataset['years_experience'].str.replace('\+ years','') 

    dataset['years_experience'] = dataset['years_experience'].str.replace('< 1 year','0')

    dataset['years_experience'] = dataset['years_experience'].str.replace('years','')

    dataset['years_experience'] = dataset['years_experience'].str.replace('year','')

    return dataset



#transform the column according to description in approved or not aproved

def get_loan_status(dataset):

    dataset['loan_status'] = dataset['loan_status'].str.replace('Does not meet the credit policy. Status:Charged Off','not_approved') 

    dataset['loan_status'] = dataset['loan_status'].str.replace('Does not meet the credit policy. Status:Fully Paid','not_approved')

    dataset['loan_status'] = dataset['loan_status'].str.replace('Charged Off','approved')

    dataset['loan_status'] = dataset['loan_status'].str.replace('Fully Paid','approved')

    return dataset



# transform the column verified status

def get_verification_status(dataset):

    dataset['verification_status'] = dataset['verification_status'].str.replace('Source Verified','Verified') 

    return dataset



# validate nulls in all the columns

def validate(dataset):

    cols = dataset.head()

    for col in cols:

        if dataset[col].isnull().values.any() : return col

    return True

#Rename columns

def rename(dataset, column_name, new_column_name):

    dataset = dataset.rename(index=str, columns={column_name:new_column_name})

    return dataset
df = get_verification_status(df)

#df = get_loan_status(df)

df = mode_fillna(df,'grade')

df = mode_fillna(df,'loan_status')

df = column_fillna(df, 'revol_util')

df = column_fillna(df, 'last_pymnt_amnt')

df = column_fillna(df, 'emp_length')

df = df.set_index('id')

df = get_month_load(df)

df = get_value(df,'int_rate','interest_rate_loan','%')



df = get_value(df,'revol_util','total_credit_used','%')

df = column_fillna(df, 'total_credit_used')  

df = get_years_experience(df)

df = column_fillna(df, 'years_experience')

df = column_fillna(df, 'annual_inc') 

df = rename(df, 'annual_inc', 'annual_income')

df = column_fillna(df, 'delinq_2yrs') 

df = rename(df, 'delinq_2yrs', 'delinquency_others_loan')

df = column_fillna(df, 'total_acc') 

df = rename(df, 'total_acc', 'num_credit_lines')

df = rename(df, 'dti', 'amount_pay_month_another_obligations')

df = rename(df, 'installment', 'monthly_payment')
validate(df)
df.head()
df2 = df.loc[df['loan_status'] == 'Fully Paid'].sample(frac = .0582)

df3 = df.loc[df['loan_status'] == 'Charged Off'].sample(frac = .134)

df4 = df.loc[df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off']

df5 = df.loc[df['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid']

df = pd.concat([df2, df3,df4,df5])

df.head()

df = get_loan_status(df)
#Get hot enconding for the categorical columns: grade, sub_grade, home_ownership, verification_status

def enmarked_hot_encode(df, column_name):

    df_hot=pd.get_dummies(df[column_name])

    df_hot = pd.DataFrame(df_hot)

    df= df.join(df_hot)

    df= df.drop([column_name], axis = 1)

    return df 



#Get binary enconding for the categorical column loan_status

def binary_encode(df):

    df['get_approved'] = df['loan_status']

    df['get_approved'] = pd.get_dummies(df['get_approved'])

   # df['get_approved'] = pd.get_dummies(df['loan_status'])

    df= df.drop(['loan_status'], axis = 1)

    return df



#Get hot enconding for the categorical columns:purpose, zip_code, addr_state

from sklearn.preprocessing import LabelEncoder

def hot_encode(df, column_name):

    le_emp = LabelEncoder()

    df[column_name] = le_emp.fit_transform(df[column_name])

    return df



# Normalize the data

from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):

    cols_to_norm = ['loan_amnt','funded_amnt','funded_amnt_inv','annual_income','monthly_payment','zip_code','addr_state']

    df[cols_to_norm] = MinMaxScaler().fit_transform(df[cols_to_norm])

    return df
df = enmarked_hot_encode(df,'grade')

df = enmarked_hot_encode(df,'sub_grade')

df = enmarked_hot_encode(df,'home_ownership')

df = enmarked_hot_encode(df,'verification_status')

df = hot_encode(df, 'purpose')

df = hot_encode(df, 'zip_code')

df = hot_encode(df, 'addr_state')

df = binary_encode(df)

df = normalize_data(df)
df.head()
import matplotlib.pyplot as plt2

plt2.hist(df.get_approved)
def get_split(df):

    X = df.loc[:, 'loan_amnt':"Verified"]

    from sklearn.model_selection import train_test_split 

    #def hot_encode(df):

    label = LabelEncoder().fit_transform(df.get_approved)

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.20, random_state=2)

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = get_split(df)
def predicted_metrics(predicted):

    print('======================================================================================')

    print('======================================================================================')

    print('')

    print('* Predicted value: ')

    print(predicted)

    print('')

    print('* Accuracy Score: ')

    print(accuracy_score(y_test, predicted))

    print('')

    print('* Confusion matrix: ')

    print (confusion_matrix(y_test, predicted))

    print('')

    print('* Classification Report: ')

    print(classification_report(y_test, predicted))
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



def KNN_Model(n_neighbors, data_train, label, data_test, y_test):

    # Create KNN classifier

    model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=2, n_neighbors=n_neighbors, p=8,

           weights='uniform')

    # Train the model using the training sets, Fit the classifier to the data

    model.fit(data_train,label)

    #Predict Output

    predicted= model.predict(data_test) 

    predicted_metrics(predicted)

    print(predicted)

    return accuracy_score(y_test, predicted)
sc_knn1 = KNN_Model(1, X_train, y_train, X_test, y_test)

sc_knn3 = KNN_Model(3, X_train, y_train, X_test, y_test)

sc_knn5 = KNN_Model(5, X_train, y_train, X_test, y_test)



sc_knn7 = KNN_Model(7, X_train, y_train, X_test, y_test)

sc_knn9 = KNN_Model(14, X_train, y_train, X_test, y_test)

sc_knn11 = KNN_Model(25,X_train, y_train, X_test, y_test)



from sklearn.model_selection import GridSearchCV



def get_best_score(model):

    

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_





knn = KNeighborsClassifier()

leaf_range = list(range(3, 15, 1))

k_range = list(range(1, 15, 1))

weight_options = ['uniform', 'distance']

n_jobs_list = [2]

param_grid = dict(leaf_size=leaf_range, n_neighbors=k_range, weights=weight_options, n_jobs=n_jobs_list)

print(param_grid)



knn_grid = GridSearchCV(knn, param_grid, cv=10, verbose=1, scoring='accuracy')

knn_grid.fit(X_train, y_train)



sc_knn = get_best_score(knn_grid)
list_scores = [sc_knn1, sc_knn3, sc_knn5, sc_knn7, sc_knn9, sc_knn11]

list_scores_label=['1', '3', '5', '7', '9', '11'] 
# allow plots to appear within the notebook

%matplotlib inline



# plot the relationship between K and testing accuracy

# plt.plot(x_axis, y_axis)

plt.plot(list_scores_label, list_scores, color='green', marker='o')

plt.xlabel('Value of K for KNN')

plt.ylabel('Testing Accuracy')

plt.show()
# Training 

# import the class



from sklearn.linear_model import LogisticRegression



def LogisticRegression_Model(X_train, X_test, y_train):



    # instantiate the model (using the default parameters)

    # STEP 2: train the model on the training set

    logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False)

    logreg.fit(X_train, y_train)

    

    # STEP 3: make predictions on the testing set

    pred_logreg = logreg.predict(X_test)



    # compare actual response values (y_test) with predicted response values (y_pred)

    return pred_logreg

    #return accuracy_score(y_test, y_pred)
pred_logreg = LogisticRegression_Model(X_train, X_test, y_train)

accuracy_logreg = accuracy_score(y_test, pred_logreg)

print(accuracy_logreg)
confusion_matrix_logreg = confusion_matrix(y_test, pred_logreg)

print (confusion_matrix_logreg)
classification_report_logreg = classification_report(y_test, pred_logreg)

print(classification_report_logreg)
# BaggingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn import tree

def BaggingClassifier_Model(X_train, X_test, y_train):

    model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))

    model.fit(X_train, y_train)

    #print(model.score(X_test,y_test))

    prediction = model.predict(X_test)

    return prediction
prediction_bagging = BaggingClassifier_Model(X_train, X_test, y_train)

accuracy_score_bagging = accuracy_score(y_test, prediction_bagging)

print(accuracy_score_bagging)   
confusion_matrix_bagging = confusion_matrix(y_test, prediction_bagging)

print (confusion_matrix_bagging)
classification_report_bagging = classification_report(y_test, prediction_bagging)

print(classification_report_bagging)
# BaggingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn import tree

def BaggingClassifier_Model(X_train, X_test, y_train):

    model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    return prediction
from sklearn.ensemble import AdaBoostClassifier

def AdaBoostClassifier_Model(X_train, X_test, y_train):

    model = AdaBoostClassifier(random_state=1)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    model.score(X_test,y_test)

    

    return prediction
prediction_adaBoost = AdaBoostClassifier_Model(X_train, X_test, y_train)

accuracy_score_adaBoost = accuracy_score(y_test, prediction_adaBoost)

print(accuracy_score_adaBoost)  
confusion_matrix_adaBoost = confusion_matrix(y_test, prediction_adaBoost)

print (confusion_matrix_adaBoost)
classification_report_adaBoost = classification_report(y_test, prediction_adaBoost)

print(classification_report_adaBoost)
from sklearn.naive_bayes import GaussianNB



def GaussianNB_Model(X_train, X_test, y_train):

    gnb=GaussianNB()

    gnb.fit(X_train,y_train)

    pred_gnb = gnb.predict(X_test)

    return pred_gnb
prediction_GaussianNB = GaussianNB_Model(X_train, X_test, y_train)

accuracy_score_GaussianNB = accuracy_score(y_test, prediction_GaussianNB)

print(accuracy_score_GaussianNB)  
confusion_matrix_GaussianNB = confusion_matrix(y_test, prediction_GaussianNB)

print (confusion_matrix_GaussianNB)
classification_report_GaussianNB = classification_report(y_test, prediction_GaussianNB)

print(classification_report_GaussianNB)
from sklearn.tree import DecisionTreeClassifier

def DecisionTreeClassifier_Model(X_train, X_test, y_train):

    dtree = DecisionTreeClassifier()

    dtree.fit(X_train,y_train)

    pred_dtree = dtree.predict(X_test)

    return pred_dtree
prediction_dtree = DecisionTreeClassifier_Model(X_train, X_test, y_train)

accuracy_score_dtree = accuracy_score(y_test, prediction_dtree)

print(accuracy_score_dtree)  
confusion_matrix_dtree = confusion_matrix(y_test, prediction_dtree)

print (confusion_matrix_dtree)
classification_report_dtree = classification_report(y_test, prediction_dtree)

print(classification_report_dtree)
from sklearn.tree import DecisionTreeClassifier

def DecisionTreeClassifier_Model_Deep(X_train, X_test, y_train):

    dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)

    dtree_2.fit(X_train,y_train)

    pred_dtree_2 = dtree_2.predict(X_test)

    return pred_dtree_2
prediction_dtree_deep = DecisionTreeClassifier_Model_Deep(X_train, X_test, y_train)

accuracy_score_dtree_deep = accuracy_score(y_test, prediction_dtree_deep)

print(accuracy_score_dtree_deep) 
confusion_matrix_dtree_deep = confusion_matrix(y_test, prediction_dtree_deep)

print (confusion_matrix_dtree_deep)
classification_report_dtree_deep = classification_report(y_test, prediction_dtree_deep)

print(classification_report_dtree_deep)
from sklearn.ensemble import RandomForestClassifier



def RandomForestClassifier_model(X_train, X_test, y_train):

    rfc = RandomForestClassifier(max_depth=6, max_features=7)

    rfc.fit(X_train, y_train)

    pred_rfc = rfc.predict(X_test)

    return pred_rfc
prediction_random_forest = RandomForestClassifier_model(X_train, X_test, y_train)

accuracy_score_random_forest = accuracy_score(y_test, prediction_random_forest)

print(accuracy_score_random_forest)  
confusion_matrix_random_forest = confusion_matrix(y_test, prediction_random_forest)

print (confusion_matrix_random_forest)
classification_report_random_forest = classification_report(y_test, prediction_random_forest)

print(classification_report_random_forest)
from sklearn.svm import SVC



def svm_Model(X_train, X_test, y_train):

    svc = SVC(gamma = 0.01, C = 100)#, probability=True)

    svc.fit(X_train, y_train)

    pred_svc = svc.predict(X_test)

    return pred_svc
pred_svc = svm_Model(X_train, X_test, y_train)

accuracy_pred_svc = accuracy_score(y_test, pred_svc)

print(accuracy_pred_svc)  
confusion_matrix_svc = confusion_matrix(y_test, pred_svc)

print (confusion_matrix_svc)
classification_report_svc = classification_report(y_test, pred_svc)

print(classification_report_svc)
print(confusion_matrix(y_test, pred_svc))

print(classification_report(y_test, pred_svc))

print(accuracy_score(y_test, pred_svc))
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline



pipe_svc = Pipeline([('std_scl', StandardScaler()), 

                    ('pca', PCA(n_components=10)),

                    ('svc', SVC(random_state=1))])



pipe_svc.fit(X_train, y_train)



print('Test Accuracy: %.3f' % pipe_svc.score(X_test, y_test))
from sklearn.model_selection import validation_curve



param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=pipe_svc,

                                             X=X_train,

                                             y=y_train,

                                             param_name='svc__C',

                                             param_range=param_range,

                                             cv=10)



# Mean value of accuracy against training data

train_mean = np.mean(train_scores, axis=1)



# Standard deviation of training accuracy per number of training samples

train_std = np.std(train_scores, axis=1)



# Same as above for test data

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



# Plot training accuracies 

plt.plot(param_range, train_mean, color='red', marker='o', label='Training Accuracy')

# Plot the variance of training accuracies

plt.fill_between(param_range,

                train_mean + train_std,

                train_mean - train_std,

                alpha=0.15, color='red')



# Plot for test data as training data

plt.plot(param_range, test_mean, color='blue', linestyle='--', marker='s', 

        label='Test Accuracy')

plt.fill_between(param_range,

                test_mean + test_std,

                test_mean - test_std,

                alpha=0.15, color='blue')



plt.xscale('log')

plt.xlabel('Regularization parameter C')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
from sklearn.model_selection import learning_curve

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



def learning_curve_plot(clf,title):

    

    train_sizes,train_scores,test_scores = learning_curve(clf,X_train,y_train,random_state = 42,cv = 5)



    plt.figure()

    plt.title(title)

    

    ylim = (0.1, 1.01)

    if ylim is not None:

        plt.ylim(*ylim)

        

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                train_scores_mean + train_scores_std, alpha=0.1,

                color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

        label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

        label="Cross-validation score")



    plt.legend(loc="best")

    plt.show()
learning_curve_plot(LogisticRegression(),'Learning Curve of Logistic Regression')
learning_curve_plot(RandomForestClassifier(),'Learning Curve of Random Forest')
learning_curve_plot(DecisionTreeClassifier(),'Learning Curve of Decision Tree')
accuracy_pred_auto_sklearn = 0.8171064604185623

list_accuracy = [sc_knn9, accuracy_logreg, accuracy_score_adaBoost, accuracy_score_bagging,accuracy_score_GaussianNB, accuracy_score_dtree, 

               accuracy_score_dtree_deep, accuracy_score_random_forest, accuracy_pred_auto_sklearn]



list_accuracy_label=['KNN','Logistic Regresion' , 'AdaBoost', 'Bagging','GaussianNB', 'DecisionTree', 'DecisionTreeDeep',

                     'RandomForest', 'Auto Sklearn', ] 



#accuracy_pred_svc





# allow plots to appear within the notebook

%matplotlib inline



# plot the relationship between K and testing accuracy



plt.plot(list_accuracy_label, list_accuracy, color='green', marker='o')

plt.xlabel('Models')

plt.xticks(list_accuracy_label, rotation=40) 

plt.ylabel('Accuracy Models')



plt.show()
list_accuracy