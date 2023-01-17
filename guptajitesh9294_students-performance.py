import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data=pd.read_csv("../input/studentsperformance/StudentsPerformance.csv")
data.head()
# Rename Columns
data.columns=['gender','race','parentsdegree','lunch','course','mathscore','readingscore','writingscore']
data.head()
# Now check the missing values
miss=data.isnull().any()
miss  # Hence no missing values 
# Lets check the percentage 
data['Percentage']=(data['mathscore']+data['readingscore']+data['writingscore'])/3
data.head()
data.groupby(['race','parentsdegree']).mean()
# Lets check the score according to gender
data.groupby('gender').mean()
# Here we can say that female percentage is greater than male percenage by some exent 
# while talking about marks female maths score is low while in other subject it is greater than other two.
# Lets check the relationship between gender , cource and percentage
course_gender = data.groupby(['gender','course']).mean().reset_index()
sns.factorplot(x='gender', y='Percentage', hue='course', data=course_gender, kind='bar')
# Now we can say that Parents Degree is also crucial in students score
course_gender = data.groupby(['gender','parentsdegree']).mean().reset_index()
sns.factorplot(x='gender', y='Percentage', hue='parentsdegree', data=course_gender, kind='bar')
data.parentsdegree.unique()
for i in range(len(data)):
    if data.iloc[i,2] in ['high school', 'some high school']:
        data.iloc[i,2] = 'No_Degree'
    else:
        data.iloc[i,2] = 'has_Degree'
        
data.head()
Lunch_course = data.groupby(['lunch','course']).mean().reset_index()
sns.factorplot(x='lunch', y='Percentage', hue='course', data=Lunch_course, kind='bar')
data.parentsdegree.value_counts()
final_data = data.groupby(['gender','parentsdegree','course','lunch','race']).mean().reset_index()
after_sort = final_data.sort_values(by= ['Percentage'],ascending = False)
after_sort.drop(columns=['mathscore','readingscore','writingscore'],inplace = True)
after_sort
# Top Performer
print("Top 10 Performer \n",after_sort[:10])
# #Simply, if you complete course, Have standard lunch, you   can score good grades.
# see bottom performers
print("Bottom Performer \n",after_sort[-10:])
#Simply, if you complete course, Have standard lunch, you   can score good grades.
base = pd.get_dummies(final_data,columns=['gender','race','parentsdegree','course','lunch'],dtype = int)
base.sample()
base.info() 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
train_x,test_x,train_y,test_y = train_test_split(base.iloc[:,4:],base.iloc[:,3],test_size = 0.05)
model = XGBRegressor(max_depth = 6)
model.fit(train_x,train_y)
target = model.predict(test_x)
mean_squared_error(target,test_y)


