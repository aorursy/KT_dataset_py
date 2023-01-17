## Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
## function to add data to plot
def annot_plot(ax,w,h):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
survey_data=pd.read_csv('../input/survey_results_public.csv')
survey_data.head()
survey_data.shape
survey=survey_data.OpenSource.value_counts()
labels=np.array(survey.index)
sizes = survey.values
colors = ['#ad0000', '#228B22']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Do you contribute to open source projects?')
plt.axis('equal')
plt.show()

ax=sns.countplot('OpenSource',data=survey_data,palette="Set1")
annot_plot(ax,0.3,100)
survey=survey_data.Hobby.value_counts()
labels=np.array(survey.index)
sizes = survey.values
colors = ['gold', 'lightskyblue']
 
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)

plt.title('Do you code as a hobby?')
plt.axis('equal')
plt.show()

ax=sns.countplot('Hobby',data=survey_data,palette="Set1")
annot_plot(ax,0.3,1)
plt.show()
ax=sns.countplot('YearsCoding',data=survey_data,palette="Set1",order=survey_data['YearsCoding'].value_counts().index)
annot_plot(ax,0.06,1)
plt.xticks(rotation=80)
plt.show()
ax=sns.countplot('JobSatisfaction',data=survey_data,palette="Set2",order=survey_data['JobSatisfaction'].value_counts().index)
annot_plot(ax,0.06,1)
plt.xticks(rotation=80)
plt.show()

ax=sns.countplot('OperatingSystem',data=survey_data,palette="Set3",order=survey_data['OperatingSystem'].value_counts().index)
annot_plot(ax,0.2,1)
plt.xticks(rotation=80)
plt.show()

survey_data.groupby('AIFuture')['AIFuture'].agg(['count']).sort_values('count').reset_index().plot(x='AIFuture',y='count',kind='barh');
survey_data['AIFuture'].value_counts()


survey_data.groupby('AIDangerous')['AIDangerous'].agg(['count']).sort_values('count').reset_index().plot(x='AIDangerous',y='count',kind='barh');
survey_data['AIDangerous'].value_counts()
survey_data.groupby('AIInteresting')['AIInteresting'].agg(['count']).sort_values('count').reset_index().plot(x='AIInteresting',y='count',kind='barh');
survey_data['AIInteresting'].value_counts()
survey_data.groupby('AIResponsible')['AIResponsible'].agg(['count']).sort_values('count').reset_index().plot(x='AIResponsible',y='count',kind='barh');
survey_data['AIResponsible'].value_counts()
temp1 = pd.DataFrame(survey_data['LanguageWorkedWith'].dropna().str.split(';').tolist()).stack().reset_index().rename(columns={0:'Language'})
temp1=temp1[temp1['Language']!=' ']
temp2=temp1.groupby('Language').size().reset_index().sort_values(0,ascending=False)[:10]
ax=temp2.plot(x='Language',y=0,kind='bar')# ax=sns.countplot('Language',data=temp1,palette="Set3",order=temp1['Language'].value_counts().index)
annot_plot(ax,0.2,1)
plt.xticks(rotation=80)
plt.show()
temp1 = pd.DataFrame(survey_data['LanguageDesireNextYear'].dropna().str.split(';').tolist()).stack().reset_index().rename(columns={0:'Language'})
temp1=temp1[temp1['Language']!=' ']
temp2=temp1.groupby('Language').size().reset_index().sort_values(0,ascending=False)[:10]
ax=temp2.plot(x='Language',y=0,kind='bar')# ax=sns.countplot('Language',data=temp1,palette="Set3",order=temp1['Language'].value_counts().index)
annot_plot(ax,0.2,1)
plt.xticks(rotation=80)
plt.show()
