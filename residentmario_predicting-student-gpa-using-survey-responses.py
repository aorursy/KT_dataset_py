import pandas as pd

students = pd.read_csv('../input/STUDENT-SURVEY.csv', encoding='latin-1')

students.head(3)
import seaborn as sns

sns.kdeplot(students['S.S.C (GPA)'])

sns.kdeplot(students['H.S.C (GPA)'])
students.loc[:, ['S.S.C (GPA)', 'H.S.C (GPA)']].corr()
target_var = 'H.S.C (GPA)'
base = (pd.get_dummies(students.Faculty)

     .rename(columns={'Arts': 'English Degree',

                      'Law': 'Law Degree'})

     .drop('Business', axis='columns')).join(

 pd.get_dummies(students['Business Program']).add_suffix(' Business Degree')

)

base.head(3)
students_under_consideration = students.loc[students['Masters Academic Year in EU'].isnull()]

base = base.iloc[students_under_consideration.index.values]
base = base.assign(

    Year=students_under_consideration.iloc[:, 8].map(lambda v: v.split(" ")[0][:1] if pd.notnull(v) else v).astype(float)

)
students['Classes are mostly'].value_counts()
students.groupby('Regular/Irregular')['H.S.C (GPA)'].mean(), students.groupby('Regular/Irregular')['H.S.C (GPA)'].std()
base = base.assign(

    Coaching=students['Did you ever attend a Coaching center?'].map(lambda v: v == "Yes"),

    Regularity=students['Regular/Irregular'].astype(bool),

    Quality_Has_Improved=students['Do you feel that the quality of education improved at EU over the last year?'].map(lambda v: v == "Yes"),

    Image_Has_Improved=students['Do you feel that the image of the University improved over the last year?'].map(lambda v: v == "Yes")

)
survey_results = base.join(students_under_consideration.iloc[:, 30:80])
survey_results = survey_results.dropna()
survey_results.shape
from sklearn.linear_model import Ridge

import numpy as np



clf = Ridge(alpha=1.0)

clf.fit(survey_results, students_under_consideration.loc[survey_results.index.values][target_var])
Y = clf.predict(survey_results)
# sns.kdeplot(students['S.S.C (GPA)'])

sns.kdeplot(students['H.S.C (GPA)'].rename('GPA'))

sns.kdeplot(pd.Series(Y).rename('GPA (Predicted)'))
sns.jointplot(x=students['H.S.C (GPA)'].rename('GPA'), 

              y=pd.Series(Y).rename('GPA (Predicted)'))