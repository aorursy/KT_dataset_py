#Import libraries and open the Multiple Choice dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

multipleChoice_df = pd.read_csv('../input/multipleChoiceResponses.csv')
#Question 5 gives us the undergraduate majors of the respondents.
q5 = multipleChoice_df['Q5'][1:].value_counts()
plt.figure(figsize=(10,5))
plt.title('Which best describes your undergraduate major?')
sns.barplot(q5.values,q5.index)
plt.xlabel('Respondents')
plt.show()
q5_remove = q5.drop('Environmental science or geology')
plt.figure(figsize=(10,5))
plt.title('Which best describes your undergraduate major? Non-STEM disciplines')
sns.barplot(q5_remove[7:].values,q5_remove[7:].index)
plt.xlabel('Respondents')
plt.show()
print('NON-STEM Majors had',sum(q5_remove[7:].values),'respondents over',sum(q5.values),
      'total respondents.')
print("That's",round(sum(q5_remove[7:].values)/sum(q5.values),4)*100,'%.')
