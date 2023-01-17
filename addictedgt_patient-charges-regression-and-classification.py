# Hide Warnings.

import warnings

warnings.filterwarnings('ignore')



# Import pandas.

import pandas as pd

# Import numpy.

import numpy as np

# Import matplotlib.

import matplotlib.pyplot as plt

# Import Seaborn.

import seaborn as sns



# Read and display the data.

data = pd.read_csv('/kaggle/input/insurance/insurance.csv')



# Check for null values.

for i in data.columns:

    print('Null Values in {i} :'.format(i = i) , data[i].isnull().sum())



    

# # Change smoking to a categorical value.

# data['smoker'] = data['smoker'].astype('category')

data
plt.figure(figsize=(12,12))

sns.distplot(data['charges'])

plt.title('Distribution of Charges.')
plt.figure(figsize=(12,12))

sns.distplot(data['bmi'] , color='r')

plt.title('Distribution of BMI.')
# Create a categorical variable age_category based on different age ranges.

def age_category(age):

    if age <= 18:

        return 'Teenager(<19)'

    elif (age>18) & (age<=24):

        return 'Youth(<25)'

    elif (age>24) & (age<60):

        return 'Adult(<60)'

    else:

        return 'Senior Citizen(>60)'

data['age_category'] = data['age'].apply(age_category)    



# Converto age category to categorical variable.

data['age_category'] = data['age_category'].astype('category')



# Plot the data.

plt.figure(figsize=(12,12))

sns.countplot(y = 'smoker' ,data = data , hue = 'age_category' , palette = 'muted' , order=['no', 'yes'])

plt.title('Number of smokers as per Age Category.')
# Create a categorical variable age_category based on different bmi ranges.

def bmi_category(bmi):

    if (bmi>=18.5) & (bmi<=24.9):

        return 'Normal'

    elif (bmi>=25) & (bmi<=29.9):

        return 'Overweight'

    elif (bmi>=30) & (bmi<=34.9):

        return 'Class I Obestity'

    else:

        return 'Class II Obestity'

data['bmi_category'] = data['bmi'].apply(bmi_category)    



# Convert bmi category to categorical variable.

data['bmi_category'] = data['bmi_category'].astype('category')



# Plot a pie chart.

font = {'family' : 'monospace',

        'weight' : 'bold',

        'size'   : 22}

plt.rc('font', **font)

label = data.bmi_category.value_counts().sort_values().index.values

lst = data.bmi_category.value_counts().sort_values().values

plt.figure(figsize=(12,12))

plt.pie(x = lst , labels=label , explode = [0.1 , 0.1 , 0.1 , 0.1],  autopct='%.1f%%' , colors=['red' , 'green' , 'blue' , 'yellow'])

plt.title(' Percentage of people by BMI Categorisation.' , fontsize = 30)

plt.show()
plt.figure(figsize=(12,12))

sns.set_style('whitegrid')

sns.lineplot(x = 'age' , y = 'charges' , hue = 'sex' , data = data )

plt.title('Age vs Charges')
plt.figure(figsize=(12,12))

sns.boxplot(x = 'smoker' , y = 'bmi' , hue = 'sex' , data = data , palette='pastel')

plt.title('BMI vs Smoking')
sns.catplot(x = 'smoker' , y = 'charges' , data = data , kind = 'boxen' , palette = 'deep' , height = 12 )

plt.title('Correlation between smoking and medical charges.')
plt.figure(figsize=(12,12))

sns.violinplot(y = 'charges' , x = 'children' , data = data , palette='pastel' )

plt.title('BMI vs Smoking')
plt.figure(figsize=(12,12))

sns.lmplot(x = 'age' , y = 'charges' , fit_reg = True, data=data , hue='bmi_category' , height=12)

plt.title('Age vs Charges')
plt.figure(figsize=(12,12))

sns.lmplot(x = 'bmi' , y = 'charges' , fit_reg = True, data=data , hue='age_category' , height=12)

plt.title('BMI vs Charges')
plt.figure(figsize=(12,12))

sns.heatmap(data.corr() , annot=True)

plt.title('Correlation Matrix')
# Import libraries.

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score



# Code the categorical values.

from sklearn.preprocessing import LabelEncoder



x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]







# Get the categorical features and convert them to a list

categorical_features = x.dtypes==object

categorical_cols = list(x.columns[categorical_features])



# Label it using label encoder.

le = LabelEncoder()

x[categorical_cols] = x[categorical_cols].apply(lambda x:le.fit_transform(x))





categorical_features
# Encode it using one hot encoder.

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=categorical_features , sparse=False)

x_ohe = ohe.fit_transform(x)

x_ohe
# Apply linear regression

y = data['charges']

r2_scr = []

lr = LinearRegression()



# Apply kfold to train the data.

kf = KFold(n_splits = 10 , random_state=42)

for tr, ts in kf.split(x_ohe):

    x_tr = x.loc[tr]

    y_tr = y.loc[tr]

    x_ts = x.loc[ts]

    y_ts = y.loc[ts]

    lr.fit(x_tr , y_tr)

    y_pred = lr.predict(x_ts)

    r2_scr.append(r2_score(y_ts , y_pred))

np.mean(r2_scr)    
# Apply polynomial regression.

from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=2)

x_poly = pf.fit_transform(x)

poly_model = LinearRegression()

r2_scr_pf = []



# Apply kfold to train the data.

kf = KFold(n_splits = 10 , random_state=42)

for tr, ts in kf.split(x_poly):

    x_tr_pf = x_poly[tr]

    y_tr_pf = y.loc[tr]

    x_ts_pf = x_poly[ts]

    y_ts_pf = y.loc[ts]

    poly_model.fit(x_tr_pf , y_tr_pf)

    y_pred_poly = poly_model.predict(x_ts_pf)

    r2_scr_pf.append(r2_score(y_ts_pf, y_pred_poly))

np.mean(r2_scr_pf)    
# We will scale the data using standar scaler and find the feature importances.

dummy_data = pd.get_dummies(data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']] ,

                            columns=['sex' , 'smoker' , 'region' ])

scaler = StandardScaler()    

scaler.fit(dummy_data)

scaled_data = scaler.transform(dummy_data)

scaled_df = pd.DataFrame(scaled_data , columns= dummy_data.columns)



# Import the libraries.

from sklearn.ensemble import RandomForestRegressor



# We use Random Forest Regressor to find the top feature. 

rf = RandomForestRegressor( n_estimators=100 , oob_score=True , random_state=42)

rf.fit(scaled_df, y )

imps = rf.feature_importances_[rf.feature_importances_.argsort()[::-1]]

headings = dummy_data.columns[rf.feature_importances_.argsort()[::-1]]

feature_imp = pd.DataFrame(imps.reshape(1,11) , columns=headings.values)

print(feature_imp)



# Apply regression on top five values and see the increase in score.

x_5 = scaled_df[['smoker_no', 'smoker_yes', 'bmi', 'age' , 'children']]
# Applying Linear Regression.

r2_scr5 = []

lr_5 = LinearRegression()

for tr,ts in kf.split(x_5):

    x_tr5 = x_5.loc[tr]

    y_tr5 = y.loc[tr]

    x_ts5 = x_5.loc[ts]

    y_ts5 = y.loc[ts]

    lr_5.fit(x_tr5 , y_tr5)

    y_5 = lr_5.predict(x_ts5)

    r2_scr5.append(r2_score(y_ts5 , y_5))

np.mean(r2_scr5)    
# Applying polynomial regression.

r2_pf5 = []

pf_5 = PolynomialFeatures(degree=3)

x_pf5 = pf_5.fit_transform(x_5)

p5 = LinearRegression()

for tr,ts in kf.split(x_pf5):

    x_tr5 = x_pf5[tr]

    y_tr5 = y.loc[tr]

    x_ts5 = x_pf5[ts]

    y_ts5 = y.loc[ts]

    p5.fit(x_tr5 , y_tr5)

    y_5 = p5.predict(x_ts5)

    r2_pf5.append(r2_score(y_ts5 , y_5))

np.mean(r2_pf5)    
plt.figure(figsize=(12,12))

sns.boxplot(y = data['charges'])

plt.title('Plot of Charges representing the outliers')



# Find the outliers using IQR.

q1 = data['charges'].quantile(0.25)

q3 = data['charges'].quantile(0.75)

iqr = q3 - q1



# Create a new df removing the outliers.

data_iqr = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']][(data.charges>(q1-1.5*iqr))&(data.charges<(q3+1.5*iqr))]
# Encode the categorical values.

data_iqr = pd.get_dummies(data_iqr )

x_iqr = data_iqr[['smoker_no', 'smoker_yes', 'bmi', 'age' , 'children']].reset_index()

y_iqr = data_iqr['charges'].reset_index()



# Apply linear regression and observe change in accuracy.



r2_iqr = []

lr_iqr = LinearRegression()

for tr,ts in kf.split(x_iqr):

    x_tr_iqr = x_iqr.loc[tr]

    y_tr_iqr = y_iqr.loc[tr]

    x_ts_iqr = x_iqr.loc[ts]

    y_ts_iqr = y_iqr.loc[ts]

    lr_iqr.fit(x_tr_iqr , y_tr_iqr)

    y_pred_iqr = lr_iqr.predict(x_ts_iqr)

    r2_iqr.append(r2_score(y_ts_iqr , y_pred_iqr))

np.mean(r2_iqr)    
# Apply polynomial regression and observe the change in accuracy.



r2_pf_iqr = []

pf_iqr = PolynomialFeatures(degree=2)

x_ipf = pf_iqr.fit_transform(x_iqr)

p_pf = LinearRegression()

for tr,ts in kf.split(x_iqr):

    x_tr_ipf = x_ipf[tr]

    y_tr_ipf = y_iqr.loc[tr]

    x_ts_ipf = x_ipf[ts]

    y_ts_ipf = y_iqr.loc[ts]

    p_pf.fit(x_tr_ipf , y_tr_ipf)

    y_ipf = p_pf.predict(x_ts_ipf)

    r2_pf_iqr.append(r2_score(y_ts_ipf , y_ipf))

np.mean(r2_pf_iqr)    
# First encode the data.

data_encoded = data[['age', 'sex' , 'bmi', 'children', 'smoker', 'charges']]



# Encoding via labels and defining target variable.

data_encoded[pd.get_dummies(data['region'] , prefix = 'region_').columns] = pd.get_dummies(data['region'] , prefix = 'region_')

data_encoded['sex'] = data_encoded['sex'].apply(lambda x: 1 if x == 'male' else 0)

data_encoded['smoker'] = data_encoded['smoker'].apply(lambda x: 1 if x == 'yes' else 0)



# Define target variable.

X = data_encoded[['age', 'sex', 'bmi', 'children', 'charges',

       'region__northeast', 'region__northwest', 'region__southeast',

       'region__southwest']]

Y = data_encoded['smoker']
# Import the library.

from sklearn.linear_model import LogisticRegression



# Import the scores.

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score



# Define variables to store accuracy and precision.

acc = []

precision = []



# Define the kfold and the model.

kf = KFold(n_splits=10 , shuffle=True, random_state=42)

model = LogisticRegression()



# Find the mean scores.

for tr,ts in kf.split(x):

    X_tr = X.loc[tr]

    Y_tr = Y.loc[tr]

    X_ts = X.loc[ts]

    Y_ts = Y.loc[ts]

    model.fit(X_tr , Y_tr)

    y_pred = model.predict(X_ts)

    acc.append(accuracy_score(Y_ts , y_pred))

    precision.append(precision_score(Y_ts , y_pred))
print('The accuracy score is ' + str(np.mean(acc)))

print('The precision score is ' + str(np.mean(precision)))
# Define the train and test.

from sklearn.model_selection import train_test_split

tr_df_X, test_df_X , tr_df_Y , test_df_Y = train_test_split(X,Y,test_size = 0.33 , random_state = 42)
# Import the library

from sklearn.ensemble import GradientBoostingClassifier



# Hypertune the parameters.

n = [10,20,30,50,100,150,200,500,1000,1500,2000,2500,3000,5000]

n_e = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=i , random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' N-Estmators ' , i)

    print(' Accuracy ' , accuracy_score(test_df_Y , y_pred))

    print(' Precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        n_e = i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if  ((temp_pre)<precision_score(test_df_Y , y_pred)):

            n_e = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)

    

print(' The value n_estimators is ' , n_e , ' with Accuracy ', temp_acc , ' with Precision ' , temp_pre)    
n = [50,75,80,90,100,150,200,250,500]

mS = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=n_e , min_samples_split=i, random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' min-samples-split ' , i)

    print(' Accuracy Score ' , accuracy_score(test_df_Y , y_pred))

    print(' Precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        mS = i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if ((temp_pre)<precision_score(test_df_Y , y_pred)):

            mS = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)

    

print(' min-samples-split is ' , mS , ' with Accuracy ', temp_acc , ' with Precision ' , temp_pre)   
n = [10,20,25,30,35,38,32,40,50,75,80,90,100,150,200,250,500]

mL = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=n_e , min_samples_split=mS, min_samples_leaf=i, random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' min_samples_leaf ' , i)

    print(' Accuracy ' , accuracy_score(test_df_Y , y_pred))

    print(' Prescision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        mL = i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if  ((temp_pre)<precision_score(test_df_Y , y_pred)):

            mL = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)

    

print(' min_samples_leaf is ' , mL , ' with R2 ', temp_acc , ' with Prescision ' , temp_pre)      
n = [10,20,25,30,35,38,32,40,50,75,80,90,100,150,200,250,500,1000,2000]

mD = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=n_e , min_samples_split=mS, min_samples_leaf=mL,max_depth=i,  random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' max_depth ' , i)

    print(' Accuracy ' , accuracy_score(test_df_Y , y_pred))

    print(' Precision ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        mD = i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if  ((temp_pre)<precision_score(test_df_Y , y_pred)):

            mD = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)

    

print(' max_depth is ' , mD , ' with Accuracy ', temp_acc , ' with Precision ' , temp_pre)  
n = [2,3,4,5,6,8,9,10,20,25,30,35,38,32,40,50,75,80,90,100,150,200,250,400]

mxL = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=n_e , min_samples_split=mS, min_samples_leaf=mL,max_depth=i,max_leaf_nodes=i, random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' max_leaf_nodes ' , i)

    print(' Accuracy Score ' , accuracy_score(test_df_Y , y_pred))

    print(' Precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        mxL = i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if ((temp_pre)<precision_score(test_df_Y , y_pred)):

            mxL = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)

    

print(' max_leaf_nodes is ' , mD , ' with Accuracy ', temp_acc , ' with Precision ' , temp_pre)   
n = [0.1 * x for x in range(1,21)]

lR = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=n_e , min_samples_split=mS, min_samples_leaf=mL,max_depth=mD,max_leaf_nodes=mxL, learning_rate=i, random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' learning_rate ' , i)

    print(' accuracy Score ' , accuracy_score(test_df_Y , y_pred))

    print(' precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        lR = i

        temp_accuracy = accuracy_score(test_df_Y , y_pred)

        temp_precision = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if ((temp_precision)<precision_score(test_df_Y , y_pred)):

            lR = i

            temp_accuracy = accuracy_score(test_df_Y , y_pred)

            temp_precision = precision_score(test_df_Y , y_pred)

    

print(' learning_rates is ' , lR , ' with accuracy ', temp_accuracy , ' with precision ' , temp_precision)  
n = [2,3,4,5,6,7,8,9]

mF = 0

counter = 0

for i in n:

    gr = GradientBoostingClassifier(n_estimators=n_e , min_samples_split=mS, min_samples_leaf=mL,max_depth=mD,max_leaf_nodes=mxL, learning_rate=lR,max_features=i, random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' max_-features ' , i)

    print(' accuracy Score ' , accuracy_score(test_df_Y , y_pred))

    print(' precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        mF = i

        temp_accuracy = accuracy_score(test_df_Y , y_pred)

        temp_precision = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if ((temp_precision)<precision_score(test_df_Y , y_pred)):

            mF = i

            temp_accuracy = accuracy_score(test_df_Y , y_pred)

            temp_precision = precision_score(test_df_Y , y_pred)

    

print(' max-features ' , mF , ' with accuracy ', temp_accuracy , ' with precision ' , temp_precision)  
# Import the library

from sklearn.ensemble import AdaBoostClassifier



# Hypertune the parameters.

n = [10,20,30,50,100,150,200,500,1000]

n_e = 0

counter = 0

for i in n:

    gr = AdaBoostClassifier(n_estimators=i , random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' N-Estmators ' , i)

    print(' Accuracy Score ' , accuracy_score(test_df_Y , y_pred))

    print(' Precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        n_e = i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if  ((temp_pre)<precision_score(test_df_Y , y_pred)):

            n_e = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)

    

print(' The value n_estimators is ' , n_e , ' with Accuracy ', temp_acc , ' with Precision ' , temp_pre)  
n = [0.1 * x for x in range(1,51)]

lR = 0

counter = 0

for i in n:

    gr = AdaBoostClassifier(n_estimators=n_e , learning_rate=i , random_state=42)

    gr.fit(tr_df_X , tr_df_Y)

    y_pred = gr.predict(test_df_X)

    print(' learning-rate ' , i)

    print(' accuracy Score ' , accuracy_score(test_df_Y , y_pred))

    print(' precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        lR = i

        temp_accuracy = accuracy_score(test_df_Y , y_pred)

        temp_precision = precision_score(test_df_Y , y_pred)

        counter += 1

    else:

        if ((temp_precision)<precision_score(test_df_Y , y_pred)):

            lR = i

            temp_accuracy = accuracy_score(test_df_Y , y_pred)

            temp_precision = precision_score(test_df_Y , y_pred)

    

print(' learning-rate ' , lR , ' with accuracy ', temp_accuracy , ' with precision ' , temp_precision) 
from sklearn.svm import LinearSVC



# Hypertune parameters.

n = [2,8,10,12,15,20,25,28,30,35,50,55,60,70,75,80,100,150,125,175,200,250,300]

counter = 0

c = 0

for i in n:

    lSVR = LinearSVC(C=i )

    lSVR.fit(tr_df_X, tr_df_Y)

    y_pred = lSVR.predict(test_df_X)

    print(' C ' , i)

    print(' precision Score ' , accuracy_score(test_df_Y , y_pred))

    print(' Precision Score ' , precision_score(test_df_Y , y_pred))

    if counter == 0:

        counter+=1

        c=i

        temp_acc = accuracy_score(test_df_Y , y_pred)

        temp_pre = precision_score(test_df_Y , y_pred)

    else:

        if  ((temp_pre)<precision_score(test_df_Y , y_pred)):

            c = i

            temp_acc = accuracy_score(test_df_Y , y_pred)

            temp_pre = precision_score(test_df_Y , y_pred)   

 

    

print(' Penalty(C) ' , c , ' with accuracy ', temp_acc , ' with Precision ' , temp_pre)    