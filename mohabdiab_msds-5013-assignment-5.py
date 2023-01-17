import pandas as pd
import numpy as np
employee_df  = pd.read_csv('../input/employeechinook/Employee.csv')
employee_df.head(1)
employee_df.full_name = employee_df.FirstName + str(' ') + employee_df.LastName
employee_df.full_name.head(7)
def address():
    pd.options.display.max_colwidth = 100
    employee_df['part1'] =  employee_df[['Address', 'City']].apply(lambda x: ' '.join(x), axis = 1)
    employee_df['part2'] = employee_df[['State','Country']].apply(lambda x: ', '.join(x), axis = 1)
    employee_df['part3'] = '     Postal Code: ' + employee_df[['PostalCode']].astype(str)
    emp_address = employee_df['address'] = employee_df.part1 + str(', ') + employee_df.part2 + employee_df.part3
    return emp_address
address()
tit_train  = pd.read_csv('../input/titanic/train_data.csv')
tit_train.head(3)
#using for loop
total_fair = 0
for i in tit_train.Fare:
    total_fair += i
total_fair
#using sum function
sum(tit_train.Fare)
def total_fare_time_comparison():
    import time
    start_for_loop = time.time()
    total_fare_for_loop = 0
    for i in tit_train.Fare:
        total_fare_for_loop += i
    total_fare_for_loop
    elapsed_time_for_loop = (time.time() - start_for_loop) 
    start_sum = time.time()
    total_fare_sum = sum(tit_train.Fare)
    elapsed_time_sum = (time.time() - start_sum) 
    print('Time Delta for for loop execution is {}, and time Delta for the Sum function execution is {}'
          .format(elapsed_time_for_loop, elapsed_time_sum))
total_fare_time_comparison()
class student:
    def __init__(my_student, LastName, FirstName, Class, Grade):
        my_student.LastName = LastName
        my_student.FirstName = FirstName
        my_student.Class = Class
        my_student.Grade = Grade
    def student_entry(my_student):
        Student_record = str('Student Name is ') + my_student.FirstName + str(' ') + my_student.LastName + str(' in class ') + my_student.Class + str(' and his Grade is ') + my_student.Grade 
        print(Student_record)
Albert = student('Einstein', 'Albert', 'MSDS5013', 'F')
Nikola = student('Tesla', "Nikola", 'MSDS5014', 'F')
my_list = list()
my_list.append(student('Einstein', 'Albert', 'MSDS5013', 'F'))
my_list.append(student('Tesla', "Nikola", 'MSDS5014', 'F'))
Albert.student_entry()
Nikola.student_entry()
heart_disease  = pd.read_csv('../input/heartdisease/HeartDisease.csv')
heart_disease.head()
heart_disease.shape
heart_disease.describe()
heart_disease.isna().any()
heart_disease.isna()
def is_there_null(dataset):
    null_val = dict(dataset.isna().sum())
    missing_counter = int(0)

    for key, value in null_val.items():
        if value == 0:
            #print('There is no Missing Value Here')
            missing_counter = missing_counter + 1
    #print(missing_counter)
        else:
            print('we have at least one missing value here')
    if missing_counter == len(dataset.columns):
        print('There is no Missing Values in any Column in The Heart Disease Data')
    return 
is_there_null(heart_disease)
Age_range = heart_disease.Age.describe()[7] - heart_disease.Age.describe()[3]
print(str('The Age range is ') + str(Age_range) + str(' years'))
heart_disease['Sex'] = heart_disease['Sex'].replace(0, 'Female')
heart_disease['Sex'] = heart_disease['Sex'].replace(1, 'Male')
print(str('The number of Males on the ship was ') + str(heart_disease.groupby('Sex').size()[1]) + str(', Meanwhile the number of females was ') + str(heart_disease.groupby('Sex').size()[0]))
heart_disease['restecg'] = heart_disease['restecg'].replace(0, 'Normal')
heart_disease['restecg'] = heart_disease['restecg'].replace(1, ' having ST-T')
heart_disease['restecg'] = heart_disease['restecg'].replace(2, 'hypertrophy')
heart_disease['restecg'].describe()
heart_disease['restecg'].value_counts()
import seaborn as sns
sns.catplot(x="restecg", y="Age", kind="box", data=heart_disease)
if any(heart_disease['trestbps'] > 140) or any(heart_disease['trestbps'] < 60):
    print('There is at least on abnormal Peak PB that is above 140 mmHg')
else:
    print('No Peak Recorded')
above_140 = heart_disease['trestbps'][heart_disease['trestbps'] > 140].count()
below_60 = heart_disease['trestbps'][heart_disease['trestbps'] < 60].count()
print('There are {} Records with peak above 140 mmHg, and {} Records which are below 60 mmHg'.format(above_140, below_60))
heart_disease_sample = heart_disease.sample(n = 200, replace = True)
Age_range_sample = heart_disease_sample.Age.describe()[7] - heart_disease_sample.Age.describe()[3]
print(str('The Age range is ') + str(Age_range_sample) + str(' years'))
#the function takes around 30-50 seconds to run becaue If you give it too big number of samples
def age_range_func(no_of_samples, size_of_sample): 
    age_range = np.array([])
    for i in range(no_of_samples):
        heart_disease_sample = heart_disease.sample(n = no_of_samples, replace = True)
        Age_range_sample = heart_disease_sample.Age.describe()[7] - heart_disease_sample.Age.describe()[3]
        age_range = np.append(age_range, Age_range_sample)
    return age_range.mean()
age_range_func(1000, 300)