import pandas as pd
data = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')
# lookup columns
data.head()
# Summarize only the data needed for analysis
data[['course_title','course_rating','course_difficulty','course_students_enrolled']]
# create DataFrame
df = pd.DataFrame(data[['course_title','course_rating','course_difficulty','course_students_enrolled']])
df
df.columns=['title', 'rating', 'level', 'enrolled']
df
# lookup level column
df['level'].unique()
# Transforms 'level column' into numeric form
def levels(x):
    if x == 'Beginner':
        return 1
    elif x == 'Mixed':
        return 2
    elif x == 'Intermediate':
        return 3
    elif x == 'Advanced':
        return 4
df['level'] = df['level'].apply(levels)
df['level'].unique()
# lookup enrolled column
df['enrolled']
# Transforms 'K' into numeric form
def counts(x):
    rx = x.replace('k','000')
    if '.' in rx:
        rx = rx.replace('.','')
        rx = rx[:-1]
        return int(rx)
    return int(rx)
# Test function 'counts()'
df['enrolled'].apply(counts)
# Apply the function to the dataframe
df['enrolled'] = df['enrolled'].apply(counts)
df
# Extract/check the lecture containing the word 'Data' in the Frame
df[df['title'].str.contains('Data')]
# Extract/check the lecture containing the word 'data' in the Frame
df[df['title'].str.contains('data')]
# All start with the form 'D'(Uppercase). Sort courses containing 'Data' by 'enrolled'
df[df['title'].str.contains('Data')].sort_values(by=['enrolled'], ascending=False)
# Exclude low rated courses
df[df['title'].str.contains('Data')].sort_values(by=['enrolled'], ascending=False)['rating'].min()
# All the ratings are good. Save it as a new variable 'Courses_data'.
Courses_data = df[df['title'].str.contains('Data')].sort_values(by=['enrolled'], ascending=False)
# Make sure it is well saved
Courses_data.head()
# Extract Courses that 100,000 or more students are enrolled
Courses_data[Courses_data['enrolled']>100000]
# Look up the number of the courses
len(Courses_data[Courses_data['enrolled']>100000])
# Save courses with 100000 or more students enrolled and Extract Top 10 Courses seperately.
AboutData_BestSellers = Courses_data[Courses_data['enrolled']>100000]
AboutData_Top_Ten = BestSellers[:10]
AboutData_Top_Ten
# look up all 'python' courses
df[df['title'].str.contains('Python')].sort_values(by=['enrolled'], ascending=False)
# 'python' courses that 100,000 or more students enrolled
Python_Courses = df[df['title'].str.contains('Python')].sort_values(by=['enrolled'], ascending=False)
Python_Courses[Python_Courses['enrolled']>100000]
# save data
Python_BestSellers = Python_Courses[Python_Courses['enrolled']>100000]
# look up R courses(1)
df[df['title'].str.contains('R ')].sort_values(by=['enrolled'], ascending=False)
# look up R courses(2)
# Extract 691 row
df[df['title'].str.contains('R ')].sort_values(by=['enrolled'], ascending=False).iloc[0,:]
# save the extracted row
R_temp_1 = df[df['title'].str.contains('R ')].sort_values(by=['enrolled'], ascending=False).iloc[0,:]
R_temp_1=pd.DataFrame(R_temp_1).T
# look up R courses(2)
df[df['title'].str[-1] =='R'].sort_values(by=['enrolled'], ascending=False)
# look up R courses(2)
# Extract 199, 753 rows
df[df['title'].str[-1] =='R'].sort_values(by=['enrolled'], ascending=False).iloc[:2,:]
# save rows that extracted
R_temp_2 = df[df['title'].str[-1] =='R'].sort_values(by=['enrolled'], ascending=False).iloc[:2,:]
pd.concat([R_temp_1, R_temp_2])
# save to R_BestSellers
R_BestSellers = pd.concat([R_temp_1, R_temp_2])
# look up Java courses
df[df['title'].str.contains('Java')].sort_values(by=['enrolled'], ascending=False)
Java_Courses = df[df['title'].str.contains('Java')].sort_values(by=['enrolled'], ascending=False)
Java_BestSellers = Java_Courses[Java_Courses['enrolled']>100000]
Java_BestSellers
# JavaScript is also included. Exclude it
Java_BestSellers = Java_BestSellers[Java_BestSellers['title'].str.contains('JavaScript')==False]
Java_BestSellers = Java_BestSellers[Java_BestSellers['title'].str.contains('Javascript')==False]
Java_BestSellers
# look up JavaSc
df[df['title'].str.contains('JavaScript')].sort_values(by=['enrolled'], ascending=False)
JavaScript_temp_1 = df[df['title'].str.contains('JavaScript')].sort_values(by=['enrolled'], ascending=False).iloc[0,:]
JavaScript_temp_1 = pd.DataFrame(JavaScript_temp_1).T
# Temporary allocation before merging
JavaScript_temp_1
df[df['title'].str.contains('Javascript')].sort_values(by=['enrolled'], ascending=False)
# Temporary allocation before merging
JavaScript_temp_2 = df[df['title'].str.contains('Javascript')].sort_values(by=['enrolled'], ascending=False)
pd.concat([JavaScript_temp_1, JavaScript_temp_2])
# save to JavaScript_BestSellers
JavaScript_BestSellers = pd.concat([JavaScript_temp_1, JavaScript_temp_2])
# look up courses about Django
df[df['title'].str.contains('Django')].sort_values(by=['enrolled'], ascending=False)
# look up courses about Flask
df[df['title'].str.contains('Flask')].sort_values(by=['enrolled'], ascending=False)
# look up courses about React
df[df['title'].str.contains('React')].sort_values(by=['enrolled'], ascending=False)
React_Courses = df[df['title'].str.contains('React')].sort_values(by=['enrolled'], ascending=False)
# save to React_BestSellers
React_BestSellers = React_Courses[React_Courses['enrolled']>100000]
React_BestSellers
# look up the courses about Vue
df[df['title'].str.contains('Vue')].sort_values(by=['enrolled'], ascending=False)

df[df['title'].str.contains('vue')].sort_values(by=['enrolled'], ascending=False)
# look up the courses about Tensorflow
df[df['title'].str.contains('TensorFlow')].sort_values(by=['enrolled'], ascending=False)
Tensorflow_Courses = df[df['title'].str.contains('TensorFlow')].sort_values(by=['enrolled'], ascending=False)
# Save to Tensorflow_BestSellers
Tensorflow_BestSellers = Tensorflow_Courses[Tensorflow_Courses['enrolled']>100000]
Tensorflow_BestSellers
# look up the courses about PyTorch
df[df['title'].str.contains('PyTorch')].sort_values(by=['enrolled'], ascending=False)
# look up the courses about 'Web'
df[df['title'].str.contains('Web')].sort_values(by=['enrolled'], ascending=False)
Web_Courses = df[df['title'].str.contains('Web')].sort_values(by=['enrolled'], ascending=False)
# save to Web_BestSellers
Web_BestSellers = Web_Courses[Web_Courses['enrolled']>100000]
Web_BestSellers
# look up the courses about 'App'
df[df['title'].str.contains('App')].sort_values(by=['enrolled'], ascending=False)
App_Courses = df[df['title'].str.contains('App')].sort_values(by=['enrolled'], ascending=False)
# save to App_BestSellers
App_BestSellers = App_Courses[App_Courses['enrolled']>100000]
App_BestSellers
# look up the courses about 'Algorithm'
df[df['title'].str.contains('Algorithm')].sort_values(by=['enrolled'], ascending=False)
Algorithm_Courses = df[df['title'].str.contains('Algorithm')].sort_values(by=['enrolled'], ascending=False)
# save to Algorigthm_BestSellers
Algorithm_BestSellers = Algorithm_Courses[Algorithm_Courses['enrolled']>100000]
Algorithm_BestSellers
# look up the courses about 'Computer'
df[df['title'].str.contains('Computer')].sort_values(by=['enrolled'], ascending=False)
Computer_Courses = df[df['title'].str.contains('Computer')].sort_values(by=['enrolled'], ascending=False)
# save to Computer_BestSellers
Computer_BestSellers = Computer_Courses[Computer_Courses['enrolled']>100000]
Computer_BestSellers
# look up the courses about 'Hacking'
df[df['title'].str.contains('Hacking')].sort_values(by=['enrolled'], ascending=False)
# Check related data
df[df['title'].str.contains('Security')].sort_values(by=['enrolled'], ascending=False)
Security_Courses = df[df['title'].str.contains('Security')].sort_values(by=['enrolled'], ascending=False)
# save to Security_BestSellers
Security_BestSellers = Security_Courses[Security_Courses['enrolled']>100000]
Security_BestSellers
# look up the courses about 'Finance'
df[df['title'].str.contains('Finance')].sort_values(by=['enrolled'], ascending=False)
Finance_Courses = df[df['title'].str.contains('Finance')].sort_values(by=['enrolled'], ascending=False)
Finance_BestSellers = Finance_Courses[Finance_Courses['enrolled']>100000]
Finance_BestSellers
# look up the courses about 'Commerce'
df[df['title'].str.contains('Commerce')].sort_values(by=['enrolled'], ascending=False)
# Check related data
df[df['title'].str.contains('Shopping')].sort_values(by=['enrolled'], ascending=False)
# Check related data
df[df['title'].str.contains('Business')].sort_values(by=['enrolled'], ascending=False)
Business_Courses = df[df['title'].str.contains('Business')].sort_values(by=['enrolled'], ascending=False)
# save to Business_BestSellers
Business_BestSellers = Business_Courses[Business_Courses['enrolled']>100000]
Business_BestSellers
# look up the courses about 'Neural Network'
df[df['title'].str.contains('Neural')].sort_values(by=['enrolled'], ascending=False)
# Check related data
df[df['title'].str.contains('CNN')].sort_values(by=['enrolled'], ascending=False)
# Check related data
df[df['title'].str.contains('RNN')].sort_values(by=['enrolled'], ascending=False)
# Check related data
df[df['title'].str.contains('Reinforcement')].sort_values(by=['enrolled'], ascending=False)
# look up the courses about 'Neural Network'
Neural_Courses = df[df['title'].str.contains('Neural')].sort_values(by=['enrolled'], ascending=False)
# save to 'Neural_BestSellers'
Neural_BestSellers = Neural_Courses[Neural_Courses['enrolled']>100000]
Neural_BestSellers
# look up the courses about 'Machine Learning'
df[df['title'].str.contains('Machine')].sort_values(by=['enrolled'], ascending=False)
Machine_Courses = df[df['title'].str.contains('Machine')].sort_values(by=['enrolled'], ascending=False)
# save to 'Machine_BestSellers'
Machine_BestSellers = Machine_Courses[Machine_Courses['enrolled']>100000]
Machine_BestSellers
# look up the courses about 'Deep Learning'
df[df['title'].str.contains('Deep')].sort_values(by=['enrolled'], ascending=False)
# All over 100,000 students, so save to 'BestSellers'
Deep_BestSellers = df[df['title'].str.contains('Deep')].sort_values(by=['enrolled'], ascending=False)