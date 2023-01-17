import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
print(f'Features: {data.shape[1]}')
print(f'Records: {data.shape[0]}')
gender = data['Gender'].value_counts()
male_count = gender[0]
female_count = gender[1]
plt.bar(['Male', 'Female'], [male_count, female_count], color=['black', 'purple'])
plt.xlabel('Gender')
plt.show()
married = data['Married'].value_counts()
yes_count = married[0]
no_count = married[1]
plt.bar(['Yes', 'No'], [yes_count, no_count], color=['black', 'purple'])
plt.xlabel('Married')
plt.show()
dependents = data['Dependents'].value_counts()
zero_count = dependents[0]
one_count = dependents[1]
two_count = dependents[2]
three_bigger_count = dependents[3]
plt.bar(['0', '1', '2', '3+'], [zero_count, one_count, two_count, three_bigger_count], color=['black', 'purple', 'red', 'green'])
plt.xlabel('Dependents')
plt.show()
self_employed = data['Self_Employed'].value_counts()
yes_count = self_employed[1]
no_count = self_employed[0]
plt.bar(['Yes', 'No'], [yes_count, no_count], color=['black', 'purple'])
plt.xlabel('Self Employed')
plt.show()
education = data['Education'].value_counts()
graduate_count = education[0]
not_graduate_count = education[1]
plt.bar(['Graduate', 'Not Graduate'], [graduate_count, not_graduate_count], color=['black', 'purple'])
plt.xlabel('Education')
plt.show()
applicant_income = data['ApplicantIncome']
plt.hist(applicant_income, color='purple')
plt.xlabel('ApplicantIncome')
plt.show()
co_applicant_income = data['CoapplicantIncome']
plt.hist(co_applicant_income, color='purple')
plt.xlabel('CoapplicantIncome')
plt.show()
loan_amount = data['LoanAmount']
plt.hist(loan_amount, color='purple')
plt.xlabel('LoanAmount')
plt.show()
loan_amount_term = data['Loan_Amount_Term']
x_axis = ['360', '180', '480', '300', '84', '240', '120', '36', '60', '12']
for i in range(0, 10):
    plt.bar(x_axis[i], loan_amount_term.value_counts().tolist()[i], color='purple')
plt.xlabel('Loan Amount Term')
plt.show()
credit_history = data['Credit_History']
one_count = credit_history.value_counts()[1]
zero_count = credit_history.value_counts()[0]
plt.bar(['1', '0'], [one_count, zero_count], color=['purple', 'black'])
plt.xlabel('Credit History')
plt.show()
property_area = data['Property_Area']
semi_urban = property_area.value_counts()['Semiurban']
urban = property_area.value_counts()['Urban']
rural = property_area.value_counts()['Rural']
plt.bar(['Semi-Urban', 'Urban', 'Rural'], [semi_urban, urban, rural], color=['purple', 'black', 'green'])
plt.xlabel('Property Area')
plt.show()
loan_status = data['Loan_Status']
yes_count = loan_status.value_counts()['Y']
no_count = loan_status.value_counts()['N']
plt.bar(['Yes', 'No'], [yes_count, no_count], color=['purple', 'black'])
plt.xlabel('Loan Status')
plt.show()
data = data.drop(columns=['Loan_ID'])
numerical_data = data.replace(to_replace=['Male', 'Female', 'No', 'Yes', '3+', 'Not Graduate', 'Graduate', 'Urban', 'Rural', 'Semiurban', 'N', 'Y'], value=[0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0])
from sklearn.preprocessing import minmax_scale
numerical_data['CoapplicantIncome'] = minmax_scale(numerical_data['CoapplicantIncome'])
numerical_data['ApplicantIncome'] = minmax_scale(numerical_data['ApplicantIncome'])
numerical_data['LoanAmount'] = minmax_scale(numerical_data['LoanAmount'])
numerical_data.to_csv('Numerical_Data.csv')
def get_missing_values(numerical_data):
    count_nan = 0
    
    indexes_without_nan = list()
    indexes_one_nan = list()
    indexes_two_nan = list()
    indexes_three_nan = list()
    
    for iteration in range(len(numerical_data)):
        for column in numerical_data.columns.tolist():
            if str(numerical_data.loc[iteration, column]) == 'nan': count_nan += 1

        if count_nan == 0:
            indexes_without_nan.append(iteration)

        if count_nan == 1:
            indexes_one_nan.append(iteration)

        if count_nan == 2:
            indexes_two_nan.append(iteration)
            
        if count_nan == 3:
            indexes_three_nan.append(iteration)
            
        count_nan = 0
        
        
    yield np.array(indexes_without_nan)
    yield np.array(indexes_one_nan)
    yield np.array(indexes_two_nan)
    yield np.array(indexes_three_nan)
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
from sklearn.linear_model import LogisticRegression
data_without_miss_value = numerical_data.loc[without_miss]
gender_index_with_miss = list()

for iteration in one_miss:
    if str(numerical_data['Gender'][iteration]) == 'nan':
        gender_index_with_miss.append(iteration)
        
train_labels = data_without_miss_value.loc[:, 'Gender']
train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=['Gender'])
test_features = numerical_data.loc[gender_index_with_miss, 'Gender':'Loan_Status'].drop(columns=['Gender'])

model = LogisticRegression(max_iter=5000)
model.fit(train_features, train_labels)
pred = model.predict(test_features)

print(f'Gender with miss value: {gender_index_with_miss}')
print(f'Predicted: {pred}')

numerical_data.loc[gender_index_with_miss, 'Gender'] = pred
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
print(numerical_data.loc[one_miss, 'Married'].hasnans)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
data_without_miss_value = numerical_data.loc[without_miss]
dependents_index_with_miss = list()

for iteration in one_miss:
    if str(numerical_data['Dependents'][iteration]) == 'nan':
        dependents_index_with_miss.append(iteration)
                
train_labels = data_without_miss_value.loc[:, 'Dependents'].astype(float)
train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=['Dependents']).astype(float)
test_features = numerical_data.loc[dependents_index_with_miss, 'Gender':'Loan_Status'].drop(columns=['Dependents']).astype(float)

model = DecisionTreeClassifier()
model.fit(train_features, train_labels)
pred = model.predict(test_features)

print(f'Dependents with miss value: {dependents_index_with_miss}')
print(f'Predicted: {pred}')

numerical_data.loc[dependents_index_with_miss, 'Dependents'] = pred
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
data_without_miss_value = numerical_data.loc[without_miss]
self_employed_index_with_miss = list()

for iteration in one_miss:
    if str(numerical_data['Self_Employed'][iteration]) == 'nan':
        self_employed_index_with_miss.append(iteration)
                
train_labels = pd.to_numeric(data_without_miss_value.loc[:, 'Self_Employed'])
train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=['Self_Employed'])
test_features = numerical_data.loc[self_employed_index_with_miss, 'Gender':'Loan_Status'].drop(columns=['Self_Employed'])

model = LogisticRegression(max_iter=5000)
model.fit(train_features, train_labels)
pred = model.predict(test_features)

print(f'Self_Employed with miss value: {self_employed_index_with_miss}')
print(f'Predicted: {pred}')

numerical_data.loc[self_employed_index_with_miss, 'Self_Employed'] = pred
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
from sklearn.linear_model import LinearRegression
data_without_miss_value = numerical_data.loc[without_miss]
loan_amount_index_with_miss = list()

for iteration in one_miss:
    if str(numerical_data['LoanAmount'][iteration]) == 'nan':
        loan_amount_index_with_miss.append(iteration)
                
train_labels = pd.to_numeric(data_without_miss_value.loc[:, 'LoanAmount'])
train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=['LoanAmount'])
test_features = numerical_data.loc[loan_amount_index_with_miss, 'Gender':'Loan_Status'].drop(columns=['LoanAmount'])

model = LinearRegression()
model.fit(train_features, train_labels)
pred = model.predict(test_features)

print(f'LoanAmount with miss value: {loan_amount_index_with_miss}')
print(f'Predicted: {pred}')

numerical_data.loc[loan_amount_index_with_miss, 'LoanAmount'] = pred
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
data_without_miss_value = numerical_data.loc[without_miss]
loan_term_index_with_miss = list()

for iteration in one_miss:
    if str(numerical_data['Loan_Amount_Term'][iteration]) == 'nan':
        loan_term_index_with_miss.append(iteration)
                
train_labels = data_without_miss_value.loc[:, 'Loan_Amount_Term'].astype(float)
train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=['Loan_Amount_Term']).astype(float)
test_features = numerical_data.loc[loan_term_index_with_miss, 'Gender':'Loan_Status'].drop(columns=['Loan_Amount_Term']).astype(float)

model = KNeighborsClassifier(500)
model.fit(train_features, train_labels)
pred = model.predict(test_features)

print(f'Loan_Amount_Term with miss value: {loan_term_index_with_miss}')
print(f'Predicted: {pred}')

numerical_data.loc[loan_term_index_with_miss, 'Loan_Amount_Term'] = pred
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
data_without_miss_value = numerical_data.loc[without_miss]
credit_index_with_miss = list()

for iteration in one_miss:
    if str(numerical_data['Credit_History'][iteration]) == 'nan':
        credit_index_with_miss.append(iteration)
                
train_labels = data_without_miss_value.loc[:, 'Credit_History'].astype(float)
train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=['Credit_History']).astype(float)
test_features = numerical_data.loc[credit_index_with_miss, 'Gender':'Loan_Status'].drop(columns=['Credit_History']).astype(float)

model = LogisticRegression(max_iter=5000)
model.fit(train_features, train_labels)
pred = model.predict(test_features)

print(f'Credit_History with miss value: {credit_index_with_miss}')
print(f'Predicted: {pred}')

numerical_data.loc[credit_index_with_miss, 'Credit_History'] = pred
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
numerical_data.loc[two_miss].isna()
for iteration in two_miss:
    columns = numerical_data.loc[two_miss].isna().loc[iteration][numerical_data.loc[two_miss].isna().loc[iteration].values == True].keys()
    for column in columns:
        
        if column == 'Gender' or column == 'Married' or column == 'Education' or column == 'Self_Employed' or column == 'Credit_History':

            train_labels = data_without_miss_value.loc[:, column].astype(float)
            train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=[i for i in columns]).astype(float)
            test_features = np.array(numerical_data.loc[iteration, 'Gender':'Loan_Status'].drop(labels=[i for i in columns]).astype(float)).reshape(1, -1)
            
            model = LogisticRegression(max_iter=5000)
            model.fit(train_features, train_labels)
            pred = model.predict(test_features)
            
            numerical_data.loc[iteration, column] = pred

        elif column == 'Dependents' or column == 'Property_Area' or column == 'Loan_Amount_Term':
            
            train_labels = data_without_miss_value.loc[:, column].astype(float)
            train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=[i for i in columns]).astype(float)
            test_features = np.array(numerical_data.loc[iteration, 'Gender':'Loan_Status'].drop(labels=[i for i in columns]).astype(float)).reshape(1, -1)

            model = KNeighborsClassifier(550)
            model.fit(train_features, train_labels)
            pred = model.predict(test_features)
            
            numerical_data.loc[iteration, column] = pred
            
        elif column == 'LoanAmount' or column == 'CoapplicantIncome' or column == 'ApplicantIncome':
            
            train_labels = data_without_miss_value.loc[:, column].astype(float)
            train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=[i for i in columns]).astype(float)
            test_features = np.array(numerical_data.loc[iteration, 'Gender':'Loan_Status'].drop(labels=[i for i in columns]).astype(float)).reshape(1, -1)

            model = LinearRegression()
            model.fit(train_features, train_labels)
            pred = model.predict(test_features)
            
            numerical_data.loc[iteration, column] = pred
            
        else:
            pass
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
for iteration in three_miss:
    columns = numerical_data.loc[three_miss].isna().loc[iteration][numerical_data.loc[three_miss].isna().loc[iteration].values == True].keys()
    for column in columns:
        
        if column == 'Gender' or column == 'Married' or column == 'Education' or column == 'Self_Employed' or column == 'Credit_History':

            train_labels = data_without_miss_value.loc[:, column].astype(float)
            train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=[i for i in columns]).astype(float)
            test_features = np.array(numerical_data.loc[iteration, 'Gender':'Loan_Status'].drop(labels=[i for i in columns]).astype(float)).reshape(1, -1)
            
            model = LogisticRegression(max_iter=5000)
            model.fit(train_features, train_labels)
            pred = model.predict(test_features)
            
            numerical_data.loc[iteration, column] = pred

        elif column == 'Dependents' or column == 'Property_Area' or column == 'Loan_Amount_Term':
            
            train_labels = data_without_miss_value.loc[:, column].astype(float)
            train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=[i for i in columns]).astype(float)
            test_features = np.array(numerical_data.loc[iteration, 'Gender':'Loan_Status'].drop(labels=[i for i in columns]).astype(float)).reshape(1, -1)

            model = KNeighborsClassifier(550)
            model.fit(train_features, train_labels)
            pred = model.predict(test_features)
            
            numerical_data.loc[iteration, column] = pred
            
        elif column == 'LoanAmount' or column == 'CoapplicantIncome' or column == 'ApplicantIncome':
            
            train_labels = data_without_miss_value.loc[:, column].astype(float)
            train_features = data_without_miss_value.loc[:, 'Gender':'Loan_Status'].drop(columns=[i for i in columns]).astype(float)
            test_features = np.array(numerical_data.loc[iteration, 'Gender':'Loan_Status'].drop(labels=[i for i in columns]).astype(float)).reshape(1, -1)

            model = LinearRegression()
            model.fit(train_features, train_labels)
            pred = model.predict(test_features)
            
            numerical_data.loc[iteration, column] = pred
            
        else:
            pass
missing_values = list(get_missing_values(numerical_data))

without_miss = missing_values[0]
one_miss = missing_values[1]
two_miss = missing_values[2]
three_miss = missing_values[3]

print(f'Records without Missing-Value: {len(without_miss)}')
print(f'Records with one Missing-Value: {len(one_miss)} ')
print(f'Records with two Missing-Value: {len(two_miss)} ')
print(f'Records with three Missing-Value: {len(three_miss)} ')
from sklearn.model_selection import cross_val_score
y = numerical_data.loc[:, 'Loan_Status']
X = numerical_data.loc[:, 'Gender':'Loan_Status'].drop(columns=['Loan_Status'])


tree_score_mean = cross_val_score(DecisionTreeClassifier(), X, y, cv=10).mean()
knn_score_mean = cross_val_score(KNeighborsClassifier(300), X, y, cv=10).mean()
logistic_score_mean = cross_val_score(LogisticRegression(max_iter=5000), X, y, cv=10).mean()
svm_score_mean = cross_val_score(SVC(), X, y, cv=10).mean()
plt.bar(x=['DecisionTree', 'k-NN', 'LogisticRegression', 'SVM'], height=[tree_score_mean, knn_score_mean, logistic_score_mean, svm_score_mean], color=['purple', 'black', 'orange', 'red'])
plt.xlabel('Performances')

print(f'DecisionTree: {tree_score_mean}')
print(f'k-NN: {knn_score_mean}')
print(f'LogisticRegression: {logistic_score_mean}')
print(f'SVM: {svm_score_mean}')