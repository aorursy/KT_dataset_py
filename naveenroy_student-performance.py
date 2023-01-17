import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("../input/studentperformance/StudentsPerformance.csv")
data
data.info()
plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                      wspace=0.5, hspace=0.1)
plt.subplot(131)
plt.title('Math Scores')
sns.barplot(x="gender", y="math score", data=data)
plt.subplot(132)
plt.title('Reading Scores')
sns.barplot(x="gender", y="reading score", data=data)
plt.subplot(133)
plt.title('Writing Scores')
sns.barplot(x="gender", y="writing score", data=data)
plt.show()
plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,wspace=0.5, hspace=0.1)
plt.subplot(131)
plt.title('Math Scores')
sns.barplot(x="race/ethnicity", y="math score", data=data)
plt.subplot(132)
plt.title('Reading Scores')
sns.barplot(x="race/ethnicity", y="reading score", data=data)
plt.subplot(133)
plt.title('Writing Scores')
sns.barplot(x="race/ethnicity", y="writing score", data=data)
plt.show()
plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,wspace=0.5, hspace=0.1)
plt.subplot(131)
plt.title('Math Scores')
sns.barplot(x="lunch", y="math score", data=data)
plt.subplot(132)
plt.title('Reading Scores')
sns.barplot(x="lunch", y="reading score", data=data)
plt.subplot(133)
plt.title('Writing Scores')
sns.barplot(x="lunch", y="writing score", data=data)
plt.show()

#As you can see there is absolutely no correlation with marks with score and lunch 
plt.figure(figsize=(15,5))
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,wspace=0.5, hspace=0.1)
plt.subplot(131)
plt.title('Gender vs Reading')
sns.barplot(x="gender", y="reading score", data=data)
plt.subplot(132)
plt.title('Race vs Math Score')
sns.barplot(x="race/ethnicity", y="math score", data=data)
plt.subplot(133)
plt.title('Lunch vs Writing Score')
sns.barplot(x="lunch", y="writing score", data=data)
plt.show()
df3=pd.DataFrame(data)
df3['total'] = df3['reading score'] + df3['writing score']+df3['math score']
df3['percentage']=df3['total']/300*100
df3
def determine_grade(scores):
    if scores >= 85:
        return 'Grade A'
    elif scores >= 70:
        return 'Grade B'
    elif scores >= 55:
        return 'Grade C'
    elif scores >= 35:
        return 'Grade D'
    elif scores >= 0 :
        return 'Grade E'
    
df3['grades']=df3['percentage'].apply(determine_grade)
    
df3
df3['grades'].value_counts().plot.pie()
plt.show()