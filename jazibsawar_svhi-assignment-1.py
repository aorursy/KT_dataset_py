import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
studentInfo = pd.read_csv('/kaggle/input/student-demographics-online-education-dataoulad/studentInfo.csv')

# students with previous attempts

studentsWithPreviousAttempts = studentInfo[studentInfo.num_of_prev_attempts > 0]

studentsWithPreviousAttempts = studentsWithPreviousAttempts['final_result'].value_counts()

studentsWithPreviousAttempts.plot(kind='bar')
# students without previous attempts

studentsWithOutPreviousAttempts = studentInfo[studentInfo.num_of_prev_attempts == 0]

studentsWithOutPreviousAttempts = studentsWithOutPreviousAttempts['final_result'].value_counts()

studentsWithOutPreviousAttempts.plot(kind='bar')
studentsWithMoreCredits = studentInfo[studentInfo.studied_credits > 150]

studentsWithMoreCredits = studentsWithMoreCredits['final_result'].value_counts()

studentsWithMoreCredits.plot(kind='bar')
studentsWithLessCredits = studentInfo[studentInfo.studied_credits <= 150]

studentsWithLessCredits = studentsWithLessCredits['final_result'].value_counts()

studentsWithLessCredits.plot(kind='bar')
studentRegistration = pd.read_csv('/kaggle/input/student-demographics-online-education-dataoulad/studentRegistration.csv')

studentRegistrationWithInfo = studentRegistration.join(studentInfo.set_index('id_student'), on='id_student', how='left', lsuffix='_left', rsuffix='_right')



studentsWithLateRegistration = studentRegistrationWithInfo[studentRegistrationWithInfo.date_registration > 30]

studentsWithLateRegistration['final_result'].value_counts().plot(kind='bar')
studentsWithEarlyRegistration = studentRegistrationWithInfo[studentRegistrationWithInfo.date_registration <= 30]

studentsWithEarlyRegistration['final_result'].value_counts().plot(kind='bar')