import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
plt.figure(figsize=(12,8))
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
passmark = 40 #minimum mark to PASS
# Math Score Distribution
p = sns.countplot("math score", data = data)
_ = plt.setp(p.get_xticklabels(), rotation=90)
# Maths Pass
data['Math_PassStatus'] = np.where(data['math score']<passmark, 'F', 'P')
data.Math_PassStatus.value_counts()
# Parents Education influence on MathsPass
p = sns.countplot(x='parental level of education', data = data, hue='Math_PassStatus')
_ = plt.setp(p.get_xticklabels(), rotation=90) 
# Reading Score Distribution
plt.figure(figsize=(12,8))
p = sns.countplot("reading score", data = data)
_ = plt.setp(p.get_xticklabels(), rotation=90)
# Reading Pass
data['Reading_PassStatus'] = np.where(data['reading score']<passmark, 'F', 'P')
data.Reading_PassStatus.value_counts()
 # Parents Education influence on ReadingPass
p = sns.countplot(x='parental level of education', data = data, hue='Reading_PassStatus')
_ = plt.setp(p.get_xticklabels(), rotation=90) 
# Writing Score Distribution
plt.figure(figsize=(12,8))
p = sns.countplot("writing score", data = data)
_ = plt.setp(p.get_xticklabels(), rotation=90)
# Reading Pass
data['Writing_PassStatus'] = np.where(data['writing score']<passmark, 'F', 'P')
data.Writing_PassStatus.value_counts()
# Parents Education influence on WritingPass
p = sns.countplot(x='parental level of education', data = data, hue='Writing_PassStatus')
_ = plt.setp(p.get_xticklabels(), rotation=90)
# Calculate Overall Pass
data['OverAll_PassStatus'] = data.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 
                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)

data.OverAll_PassStatus.value_counts()
# Parents Education influence on ReadingPass
p = sns.countplot(x='parental level of education', data = data, hue='OverAll_PassStatus')
_ = plt.setp(p.get_xticklabels(), rotation=90)
# Calculating the Percentages
data['Total_Marks'] = data['math score']+data['reading score']+data['writing score']
data['Percentage'] = data['Total_Marks']/3
# Calculate Grade
def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

data['Grade'] = data.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)

data.Grade.value_counts()
# Parents Education influence on Percentages
p = sns.countplot(x='parental level of education', data = data, hue='Grade')
_ = plt.setp(p.get_xticklabels(), rotation=90)