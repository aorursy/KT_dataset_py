import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from IPython.display import HTML
sns.set(rc = {'figure.figsize':(15,8)})
warnings.filterwarnings('ignore')
pd.options.display.max_rows = 50000
pd.options.display.max_columns = 1000

sns.set(rc = {'figure.figsize':(15,8)})

def printmd(string):
    display(Markdown(string))
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Toggle Hide/Show code"></form>''')
data = pd.read_csv('../input/StudentsPerformance.csv')
data.loc[:,'id'] = range(len(data))
data.columns = ['gender', 'ethnicity', 'parent_lvl_of_edu', 'lunch',
       'test_prep_course', 'math_score', 'reading_score',
       'writing_score','id']
data = data[['id','gender', 'ethnicity', 'parent_lvl_of_edu', 'lunch',
       'test_prep_course', 'math_score', 'reading_score',
       'writing_score']]
print('Rows : ',data.shape[0],'\nColumns : ',data.shape[1],'\n')
print('Data Types of Each Column\n',data.dtypes)
data.head()
data.isnull().sum()
sns.set(rc = {'figure.figsize':(15,10)})
plt.subplots_adjust(hspace = 0.4)
plt.subplot(221)
sns.countplot(data.gender)
plt.title('Gender Distribution')
plt.subplot(222)
sns.countplot(data.ethnicity)
plt.title('Race/Ethnicity Distribution')
plt.subplot(223)
sns.countplot(data.parent_lvl_of_edu)
plt.title('Education Level of Parent Distribution')
plt.xticks(rotation = 45)
plt.subplot(224)
sns.countplot(data.lunch)
plt.title('Lunch Distribution')
plt.show()
fig,ax = plt.subplots(figsize = [6,6])
sns.countplot(data.test_prep_course,axes = ax)
plt.title('Test preperation course Distribution')
plt.show()
sns.set(rc = {'figure.figsize':(15,9)})
grid = plt.GridSpec(1, 5, wspace=0.2, hspace=1)
plt.subplot(grid[0, :2])
ax = sns.violinplot(data.math_score, orient = 'v', color='red')
ax = sns.violinplot(data.reading_score,orient = 'v',color = 'blue')
ax = sns.violinplot(data.writing_score, orient = 'v', color="orange")
plt.setp(ax.collections, alpha=.5)
plt.ylabel('Score')
plt.title('Overlapped distritbution of different Scores')
plt.subplot(grid[0, 2])
ax = sns.violinplot(data.math_score, orient = 'v', color='red')
plt.setp(ax.collections, alpha=.5)
plt.ylabel('')
plt.yticks([])
plt.title('Math Score')
plt.subplot(grid[0, 3])
ax = sns.violinplot(data.reading_score, orient = 'v', color='blue')
plt.setp(ax.collections, alpha=.5)
plt.ylabel('')
plt.yticks([])
plt.title('Reading Score')
plt.subplot(grid[0, 4])
ax = sns.violinplot(data.writing_score,orient = 'v',color = 'orange')
plt.setp(ax.collections, alpha=.5)
plt.ylabel('')
plt.yticks([])
plt.title('Writing Score')
plt.show()

sns.violinplot(x = 'variable', y = 'value', hue = 'lunch',
              data = pd.melt(data, 
               id_vars = ['id_var','lunch'], 
               value_vars=['reading_score','math_score','writing_score']),
              split = True,
              palette = 'Set1')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Types of Lunch')
plt.show()
sns.violinplot(x = 'variable', y = 'value',hue='gender',
               data = pd.melt(data, 
               id_vars = ['id_var','gender'], 
               value_vars=['reading_score','math_score','writing_score'])
               ,split = True,
              palette = 'Set2')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Types of Gender')
plt.show()
sns.boxplot(x = 'variable', y = 'value',hue='test_prep_course',
               data = pd.melt(data, 
               id_vars = ['id_var','test_prep_course'], 
               value_vars=['reading_score','math_score','writing_score'])
               ,
              palette = 'Set3')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Test preparation course flag')
plt.show()
grid = plt.GridSpec(1, 7, wspace=0.6, hspace=1)
plt.subplot(grid[0, :4])
sns.boxplot(x = 'variable', y = 'value',hue='ethnicity',
               data = pd.melt(data, 
               id_vars = ['id_var','ethnicity'], 
               value_vars=['reading_score','math_score','writing_score']),
              palette = 'Set1')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Ethnicities')

plt.subplot(grid[0,4:])
sns.countplot(data.ethnicity,palette = 'Set1')
plt.title('Race/Ethnicity Distribution')
plt.xlabel('')

plt.show()
grid = plt.GridSpec(1, 7, wspace=0.6, hspace=1)
plt.subplot(grid[0, :4])
sns.boxplot(x = 'variable', y = 'value',hue='parent_lvl_of_edu',
               data = pd.melt(data, 
               id_vars = ['id_var','parent_lvl_of_edu'], 
               value_vars=['reading_score','math_score','writing_score']),
              palette = 'Set3')
plt.xticks(range(3),['Reading Score','Math Score', 'Writing Score'])
plt.xlabel('Type of Score')
plt.title('Distribution of Scores across Different Levels of Parent Education')

plt.subplot(grid[0,4:])
sns.countplot(data.parent_lvl_of_edu,palette = 'Set3')
plt.title('Parent Education Level Distribution')
plt.xticks(rotation = 45)
plt.xlabel('')
plt.show()
fig,ax = plt.subplots(figsize = [9,8])
sns.heatmap(data[['reading_score','math_score','writing_score']].corr(),
           cmap="YlGnBu",annot= True,axes = ax)
plt.show()

