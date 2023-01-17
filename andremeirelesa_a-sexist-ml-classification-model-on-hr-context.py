# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import Normalizer, scale, StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



hrdata = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")



#hrdata.head()
def generate_gender(x):

    if x.PerformanceRating == 4 and x.Gender == 'Female':

        if np.random.random_sample() >= 0.98:

            return 3

    if x.PerformanceRating == 3 and x.Gender == 'Male':

        if np.random.random_sample() >= 0.98:

            return 4

    return x.PerformanceRating



biased_y = hrdata.apply(generate_gender, axis=1)



hrdata['PerformanceRating'] = biased_y



hrdata.groupby(['PerformanceRating','Gender']).size()
y = hrdata['PerformanceRating']



y.head()



# hrdata.columns
X = hrdata[['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',\

            'Education', 'EducationField', 'Gender', 'HourlyRate', 'JobLevel', 'JobRole',\

            'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',\

            'OverTime', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',\

            'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']] 



X.head()


X['BusinessTravel'] = X['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 0.5, 'Travel_Frequently': 1})



X['DepSales'] = X['Department'].map({'Sales': 1, 'Research & Development': 0, 'Human Resources': 0})

X['DepResDev'] = X['Department'].map({'Sales': 0, 'Research & Development': 1, 'Human Resources': 0})

X['DepHR'] = X['Department'].map({'Sales': 0, 'Research & Development': 0, 'Human Resources': 1})



X['EducLifeScience'] = X['EducationField'].map({'Life Sciences': 1, 'Other':0, 'Medical':0, 'Marketing':0,

       'Technical Degree':0, 'Human Resources':0})

X['EducOther'] = X['EducationField'].map({'Life Sciences': 0, 'Other':1, 'Medical':0, 'Marketing':0,

       'Technical Degree':0, 'Human Resources':0})

X['EducMedical'] = X['EducationField'].map({'Life Sciences': 0, 'Other':0, 'Medical':1, 'Marketing':0,

       'Technical Degree':0, 'Human Resources':0})

X['EducMarketing'] = X['EducationField'].map({'Life Sciences': 0, 'Other':0, 'Medical':0, 'Marketing':1,

       'Technical Degree':0, 'Human Resources':0})

X['EducTechDegree'] = X['EducationField'].map({'Life Sciences': 0, 'Other':0, 'Medical':0, 'Marketing':0,

       'Technical Degree':1, 'Human Resources':0})

X['EducHR'] = X['EducationField'].map({'Life Sciences': 0, 'Other':0, 'Medical':0, 'Marketing':0,

       'Technical Degree':0, 'Human Resources':1})



X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})



X['RoleSalesExec'] = X['JobRole'].map({'Sales Executive': 1, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 0})

X['RoleResScientist'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 1, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 0})

X['RoleLabTech'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 1,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 0})

X['RoleManufactDir'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 1, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 0})

X['RoleHealthRep'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 1, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 0})

X['RoleManager'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 1,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 0})

X['RoleSalesRep'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 1, 'Research Director': 0, 'Human Resources': 0})

X['RoleResDir'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 1, 'Human Resources': 0})

X['RoleHR'] = X['JobRole'].map({'Sales Executive': 0, 'Research Scientist': 0, 'Laboratory Technician': 0,

       'Manufacturing Director': 0, 'Healthcare Representative': 0, 'Manager': 0,

       'Sales Representative': 0, 'Research Director': 0, 'Human Resources': 1})



X['Single'] = X['MaritalStatus'].map({'Single': 1, 'Married':0, 'Divorced':0})

X['Married'] = X['MaritalStatus'].map({'Single': 0, 'Married':1, 'Divorced':0})

X['Divorced'] = X['MaritalStatus'].map({'Single': 0, 'Married':0, 'Divorced':1})



X['OverTime'] = X['OverTime'].map({'Yes': 1, 'No':0})



X['BelowCollege'] = X['Education'].map({1: 1, 2: 0, 3: 0, 4: 0, 5: 0})

X['College'] = X['Education'].map({1: 0, 2: 1, 3: 0, 4: 0, 5: 0})

X['Bachelor'] = X['Education'].map({1: 0, 2: 0, 3: 1, 4: 0, 5: 0})

X['Master'] = X['Education'].map({1: 0, 2: 0, 3: 0, 4: 1, 5: 0})

X['Doctor'] = X['Education'].map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1})



X.loc[:50, ['Gender','EducationField','EducLifeScience', 'EducOther', 'EducMedical', 'EducMarketing','EducTechDegree', 'EducHR', \

'JobRole','RoleSalesExec', 'RoleResScientist', 'RoleLabTech', 'RoleManufactDir', 'RoleHealthRep', 'RoleManager', \

'RoleSalesRep', 'RoleResDir', 'RoleHR', 'MaritalStatus','Single', 'Married', 'Divorced', 'OverTime', 'Education', 'BelowCollege','College',\

             'Bachelor', 'Master', 'Doctor']]
X = X.drop(['Department', 'Education','EducationField', 'JobRole', 'MaritalStatus'], axis=1)



X.head()
scaler = StandardScaler()



features_to_scale = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',\

                    'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',\

                    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

scaled_features = pd.DataFrame(scaler.fit_transform(X[features_to_scale]), columns=features_to_scale)



X[features_to_scale] = scaled_features;



X.head()

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



from sklearn.svm import SVC 

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 

svm_predictions = svm_model_linear.predict(X_test) 

  

# model accuracy for X_test   

accuracy = svm_model_linear.score(X_test, y_test)



print(accuracy)
X_male = X.copy()

X_female = X.copy()



X_male.Gender = 0

X_female.Gender = 1



male_prediction = svm_model_linear.predict(X_male)

female_prediction = svm_model_linear.predict(X_female)



diff_predictions = pd.Series(male_prediction == female_prediction)



print('There are', diff_predictions[diff_predictions == False].size, 'cases where the model classified with different PerformanceRating for samples having only the gender as different attribute from each other')