import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

from sklearn import model_selection,linear_model, metrics



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cr_data = pd.read_csv("/kaggle/input/credit-risk-dataset/credit_risk_dataset.csv")

shape = cr_data.shape

print("There are {} rows and {} features.".format(shape[0], shape[1]))

print(cr_data.dtypes)

cr_data
# we will shorten the last 2 feature names and address the null values

cr_data = cr_data.rename(columns = {"cb_person_default_on_file":"default_hist", "cb_person_cred_hist_length": "cr_hist_len"})

cr_data.isnull().sum()
# percentage of null values from loan int rate col

cr_data.loan_int_rate.isnull().sum() / cr_data.shape[0]
plt.hist(cr_data['person_emp_length'])

plt.xlabel("Employment Length")

plt.ylabel("Frequency")

plt.title("Freq vs Employment Length")

plt.show()



plt.hist(cr_data['loan_int_rate'])

plt.xlabel("Interest Rate")

plt.ylabel("Frequency")

plt.title("Freq vs Interest Rate")
emp_len_null = cr_data[cr_data['person_emp_length'].isnull()].index

int_rate_null = cr_data[cr_data['loan_int_rate'].isnull()].index



cr_data['person_emp_length'].fillna((cr_data['person_emp_length'].median()), inplace=True)

cr_data['loan_int_rate'].fillna((cr_data['loan_int_rate'].median()), inplace = True)
# check distribution of age and interest rate





colors = ["blue","red"]

plt.scatter(cr_data['person_age'], cr_data['loan_int_rate'],

            c = cr_data['loan_status'],

            cmap = mpl.colors.ListedColormap(colors), alpha=0.5)

plt.xlabel("Person Age")

plt.ylabel("Loan Interest Rate")

plt.title("Interest Rate vs Age")

# Clean 1

cr_clean1 = cr_data[cr_data['person_age']<=100]



cr_data[cr_data['person_age']>100]
pd.crosstab(cr_clean1['default_hist'], cr_clean1['loan_grade'])
# note 0 is non default and 1 is default

default_hist_status_tab = pd.crosstab(cr_clean1['default_hist'], cr_clean1['loan_status'])

default_hist_status_tab
total1 = default_hist_status_tab.iloc[0].sum()

defaulted1 = default_hist_status_tab.iloc[0,1]



total2 = default_hist_status_tab.iloc[1].sum()

defaulted2 = default_hist_status_tab.iloc[1,1]



first_default = round(defaulted1 / total1 * 100, 2)

second_default = round(defaulted2 / total2 * 100, 2)



print("Despite the measures taken, {}% of clients defaulted for the first time.".format(first_default))

print("And {}% of clients who had previously defaulted, defaulted again.".format(second_default))
pd.crosstab(cr_clean1['default_hist'], cr_clean1['loan_intent'], 

            values = cr_clean1['loan_int_rate'], aggfunc = 'median')
cr_clean1
# one hot encoding categorical variables

num_col = cr_clean1.select_dtypes(exclude = 'object')

char_col = cr_clean1.select_dtypes(include = 'object')



encoded_char_col = pd.get_dummies(char_col)



cr_clean2 = pd.concat([num_col, encoded_char_col], axis=1)

cr_clean2
# Split Train and Test Sets

Y = cr_clean2['loan_status']

X = cr_clean2.drop('loan_status',axis=1)

 





x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, random_state=2020, test_size=.30)



#Start of Classification Logistics Regression



log_clf = linear_model.LogisticRegression()



log_clf.fit(x_train, np.ravel(y_train))
col_effect = pd.DataFrame()

col_effect['col_names'] = X.columns

col_effect['col_coef'] = log_clf.coef_[0]

col_effect
int_val = float(log_clf.intercept_)

print('The overall probablity of non default is {:.3%}'.format(int_val))

# first column is the logistic regression value

# second column is the predicted probability of default == 1

predict_log = pd.DataFrame(log_clf.predict_proba(x_test)[:,1], columns=['prob_default'])



pred_df = pd.concat([y_test.reset_index(drop=True), predict_log],axis=1)

pred_df
# check the accuracy

initial_accuracy = round(log_clf.score(x_test,  y_test),2)

print("The initial accuracy is {}".format(initial_accuracy))
thresh = np.linspace(0,1,21)

thresh
metrics.recall_score(pred_df.iloc[:,0],y_test, labels = [0,1])
def find_opt_thresh(predict,thr =thresh, y_true = y_test):

    data = predict

    

    def_recalls = []

    nondef_recalls = []

    accs =[]



    

    for threshold in thr:

        # predicted values for each threshold

        data['loan_status'] = data['prob_default'].apply(lambda x: 1 if x > threshold else 0 )

        

        accs.append(metrics.accuracy_score(y_true, data['loan_status']))

        

        stats = metrics.precision_recall_fscore_support(y_true, data['loan_status'])

        

        def_recalls.append(stats[1][1])

        nondef_recalls.append(stats[1][0])

        

        

    return accs, def_recalls, nondef_recalls



accs, def_recalls, nondef_recalls= find_opt_thresh(pred_df)
plt.plot(thresh,def_recalls)

plt.plot(thresh,nondef_recalls)

plt.plot(thresh,accs)

plt.xlabel("Probability Threshold")

plt.xticks(thresh, rotation = 'vertical')

plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])

#plt.axvline(x=0.45, color='pink')

plt.show()

max_accuracy_index = accs.index(max(accs))



print('The maximum accuracy is {:.0%}.'.format(accs[max_accuracy_index]))

print('Therefore we should have a threshold of {:.0%}.'.format(thresh[max_accuracy_index]))
cr_clean2
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=[0, 1])

data_rescaled = scaler.fit_transform(cr_clean2)



#Fitting the PCA algorithm with our Data

pca = PCA().fit(data_rescaled)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Hotel Booking Dataset Explained Variance')

plt.show()
# normalize data

from sklearn import preprocessing

from sklearn.decomposition import PCA



pie = cr_clean2.drop('loan_status',axis=1)



data_scaled = pd.DataFrame(preprocessing.scale(pie),columns = pie.columns) 



# PCA

pca = PCA(n_components=14)

pca_val = pca.fit_transform(data_scaled)

pca_dataset = pd.DataFrame(pca_val)
x_train, x_test, y_train, y_test = model_selection.train_test_split(pca_dataset, Y, random_state=2020, test_size=.32)



#Start of Classification Logistics Regression



log_clf = linear_model.LogisticRegression()



log_clf.fit(x_train, np.ravel(y_train))



# first column is the logistic regression value

# second column is the predicted probability of default == 1

pca_predict_log = pd.DataFrame(log_clf.predict_proba(x_test)[:,1], columns=['prob_default'])



pca_pred_df = pd.concat([y_test.reset_index(drop=True), predict_log],axis=1)

pca_pred_df



pca_accuracy = round(log_clf.score(x_test,  y_test),2)

pca_accuracy

round(default_hist_status_tab.iloc[:,1].sum() / pca_dataset.shape[0],2)
