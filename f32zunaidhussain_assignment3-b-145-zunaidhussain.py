import pandas as pd
#Write a Pandas program to get the powers of an array values element-wise.

#Write a Pandas program to create and display a DataFrame from a specified dictionary data which has the index labels
df=pd.DataFrame({'Students':['A','B','C','D'],'Subjects':['DSD','AI/ML','COA','ALGORITHMS'],'Marks':[50,90,80,80]})
df
#Write a Pandas program to get the first 3 rows of a given DataFrame.
data=pd.read_csv('../input/iris/Iris.csv')
data.iloc[:3]
#Write a Pandas program to select the specified columns and rows from a given data frame.
data.loc[:3,'SepalLengthCm']
#Write a Pandas program to select the rows where the score is missing, i.e. is NaN.
data2=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data2.head()
data2.loc[data2.serum_creatinine.isnull()]
#Since no NAN is found