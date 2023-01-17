import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sklearn as sk
print(sk.__version__)
exam_data = pd.read_csv('../data/exams.csv', quotechar='"')
exam_data.head()
math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = average = exam_data['writing score'].mean()

print('Math Avg: ', math_average)
print('Reading Avg: ', reading_average)
print('Writing Avg: ', writing_average)
# preprocessing.scale(my_df[['column1']]) will standardize column1 by using calculation (x - avg(x))/stdev(x)
from sklearn import preprocessing

exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])
exam_data.head()
math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = average = exam_data['writing score'].mean()

print('Math Avg: ', math_average)
print('Reading Avg: ', reading_average)
print('Writing Avg: ', writing_average)
preprocessing.LabelEncoder() # is used when there are comparisons between categorical variables e.g. low,medium,high
le = preprocessing.LabelEncoder() 
exam_data['gender'] = le.fit_transform(exam_data['gender'].astype(str))
exam_data.head() # gender column values will be converted to numbers , binary in this case
le.classes_ # classes_ variable will give the unique values of the column
# pandas.get_dummies allows us to convert each category into a column with the One-Hot encoding values
pd.get_dummies(exam_data['race/ethnicity']).head()
exam_data = pd.get_dummies(exam_data, columns=['race/ethnicity'])
exam_data.head()
# get_dummies function can work on multiple columns at the same time
exam_data = pd.get_dummies(exam_data, columns=['parental level of education','lunch','test preparation course'])
exam_data.head()