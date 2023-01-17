import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np
# Standard ML Models for comparison

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR



# Splitting data into training/testing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



# Metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error



# Distributions

import scipy
student = pd.read_csv('../input/student-mat.csv')

student.head()
print('Total students:',len(student))
student['G3'].describe()
student['G2'].describe()
student['G1'].describe()
plt.subplots(figsize=(10,10))

grade_counts = student['G3'].value_counts().sort_values().plot.barh(width=.9,color=sns.color_palette('husl',40))

grade_counts.axes.set_title('Number of students who scored a particular grade',fontsize=25)

grade_counts.set_xlabel('Number of students', fontsize=20)

grade_counts.set_ylabel('Final Grade', fontsize=20)

plt.show()
b = sns.countplot(student['G3'])

b.axes.set_title('Distribution of Final grade of students', fontsize = 25)

b.set_xlabel('Final Grade', fontsize = 20)

b.set_ylabel('Count', fontsize = 20)

plt.show()
student.isnull().any()
male_studs = len(student[student['sex'] == 'M'])

female_studs = len(student[student['sex'] == 'F'])

print('Number of male students:',male_studs)

print('Number of female students:',female_studs)
b = sns.kdeplot(student['age'], shade=True)

b.axes.set_title('Ages of students', fontsize = 30)

b.set_xlabel('Age', fontsize = 20)

b.set_ylabel('Count', fontsize = 20)

plt.show()
b = sns.countplot('age',hue='sex', data=student)

b.axes.set_title('Number of students in different age groups',fontsize=30)

b.set_xlabel("Age",fontsize=30)

b.set_ylabel("Count",fontsize=20)

plt.show()
b = sns.boxplot(x='age', y='G3', data=student)

b.axes.set_title('Age vs Final', fontsize = 30)

b.set_xlabel('Age', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.swarmplot(x='age', y='G3',hue='sex', data=student)

b.axes.set_title('Does age affect final grade?', fontsize = 30)

b.set_xlabel('Age', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.countplot(student['address'])

b.axes.set_title('Urban and rural students', fontsize = 30)

b.set_xlabel('Address', fontsize = 20)

b.set_ylabel('Count', fontsize = 20)

plt.show()
# Grade distribution by address

sns.kdeplot(student.loc[student['address'] == 'U', 'G3'], label='Urban', shade = True)

sns.kdeplot(student.loc[student['address'] == 'R', 'G3'], label='Rural', shade = True)

plt.title('Do urban students score higher than rural students?', fontsize = 20)

plt.xlabel('Grade', fontsize = 20);

plt.ylabel('Density', fontsize = 20)

plt.show()
b = sns.swarmplot(x='reason', y='G3', data=student)

b.axes.set_title('Reason vs Final grade', fontsize = 30)

b.set_xlabel('Reason', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
student.corr()['G3'].sort_values()
# Select only categorical variables

category_df = student.select_dtypes(include=['object'])



# One hot encode the variables

dummy_df = pd.get_dummies(category_df)



# Put the grade back in the dataframe

dummy_df['G3'] = student['G3']



# Find correlations with grade

dummy_df.corr()['G3'].sort_values()
# selecting the most correlated values and dropping the others

labels = student['G3']



# drop the school and grade columns

student = student.drop(['school', 'G1', 'G2'], axis='columns')

    

# One-Hot Encoding of Categorical Variables

student = pd.get_dummies(student)
# Find correlations with the Grade

most_correlated = student.corr().abs()['G3'].sort_values(ascending=False)



# Maintain the top 8 most correlation features with Grade

most_correlated = most_correlated[:9]

most_correlated
student = student.loc[:, most_correlated.index]

student.head()
b = sns.swarmplot(x=student['failures'],y=student['G3'])

b.axes.set_title('Students with less failures score higher', fontsize = 30)

b.set_xlabel('Number of failures', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
family_ed = student['Fedu'] + student['Medu'] 

b = sns.boxplot(x=family_ed,y=student['G3'])

b.axes.set_title('Educated families result in higher grades', fontsize = 30)

b.set_xlabel('Family education (Mother + Father)', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.swarmplot(x=family_ed,y=student['G3'])

b.axes.set_title('Educated families result in higher grades', fontsize = 30)

b.set_xlabel('Family education (Mother + Father)', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
student = student.drop('higher_no', axis='columns')

student.head()
b = sns.boxplot(x = student['higher_yes'], y=student['G3'])

b.axes.set_title('Students who wish to go for higher studies score more', fontsize = 30)

b.set_xlabel('Higher education (1 = Yes)', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.countplot(student['goout'])

b.axes.set_title('How often do students go out with friends', fontsize = 30)

b.set_xlabel('Go out', fontsize = 20)

b.set_ylabel('Count', fontsize = 20)

plt.show()
b = sns.swarmplot(x=student['goout'],y=student['G3'])

b.axes.set_title('Students who go out a lot score less', fontsize = 30)

b.set_xlabel('Going out', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
b = sns.swarmplot(x=student['romantic_no'],y=student['G3'])

b.axes.set_title('Students with no romantic relationship score higher', fontsize = 25)

b.set_xlabel('Romantic relationship (1 = None)', fontsize = 20)

b.set_ylabel('Final Grade', fontsize = 20)

plt.show()
# splitting the data into training and testing data (75% and 25%)

# we mention the random state to achieve the same split everytime we run the code

X_train, X_test, y_train, y_test = train_test_split(student, labels, test_size = 0.25, random_state=42)
X_train.head()
# Calculate mae and rmse

def evaluate_predictions(predictions, true):

    mae = np.mean(abs(predictions - true))

    rmse = np.sqrt(np.mean((predictions - true) ** 2))

    

    return mae, rmse
# find the median

median_pred = X_train['G3'].median()



# create a list with all values as median

median_preds = [median_pred for _ in range(len(X_test))]



# store the true G3 values for passing into the function

true = X_test['G3']
# Display the naive baseline metrics

mb_mae, mb_rmse = evaluate_predictions(median_preds, true)

print('Median Baseline  MAE: {:.4f}'.format(mb_mae))

print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))
# Evaluate several ml models by training on training set and testing on testing set

def evaluate(X_train, X_test, y_train, y_test):

    # Names of models

    model_name_list = ['Linear Regression', 'ElasticNet Regression',

                      'Random Forest', 'Extra Trees', 'SVM',

                       'Gradient Boosted', 'Baseline']

    X_train = X_train.drop('G3', axis='columns')

    X_test = X_test.drop('G3', axis='columns')

    

    # Instantiate the models

    model1 = LinearRegression()

    model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)

    model3 = RandomForestRegressor(n_estimators=100)

    model4 = ExtraTreesRegressor(n_estimators=100)

    model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')

    model6 = GradientBoostingRegressor(n_estimators=50)

    

    # Dataframe for results

    results = pd.DataFrame(columns=['mae', 'rmse'], index = model_name_list)

    

    # Train and predict with each model

    for i, model in enumerate([model1, model2, model3, model4, model5, model6]):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        

        # Metrics

        mae = np.mean(abs(predictions - y_test))

        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        

        # Insert results into the dataframe

        model_name = model_name_list[i]

        results.loc[model_name, :] = [mae, rmse]

    

    # Median Value Baseline Metrics

    baseline = np.median(y_train)

    baseline_mae = np.mean(abs(baseline - y_test))

    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))

    

    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]

    

    return results
results = evaluate(X_train, X_test, y_train, y_test)

results
plt.figure(figsize=(12, 8))



# Root mean squared error

ax =  plt.subplot(1, 2, 1)

results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'b', ax = ax, fontsize=20)

plt.title('Model Mean Absolute Error', fontsize=20) 

plt.ylabel('MAE', fontsize=20)



# Median absolute percentage error

ax = plt.subplot(1, 2, 2)

results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'r', ax = ax, fontsize=20)

plt.title('Model Root Mean Squared Error', fontsize=20) 

plt.ylabel('RMSE',fontsize=20)



plt.show()