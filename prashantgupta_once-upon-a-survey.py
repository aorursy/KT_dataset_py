# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
surveySchema = pd.read_csv("../input/SurveySchema.csv")
freeFormResponses = pd.read_csv("../input/freeFormResponses.csv", low_memory=False)
multipleChoiceResponses = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False)
toolDf = multipleChoiceResponses[[x for x in multipleChoiceResponses.columns if 'Q13' in x][:-1]]
cols = {}
for column in toolDf.columns:
    cols[column] = multipleChoiceResponses.at[0, column].split('-')[2]
toolDf = toolDf.rename(columns=cols).drop([0])
tools = toolDf.count().sort_values(ascending=False)
plt.figure(figsize=(18, 10))
plt.title('Top 10 Frameworks used by Respondents', fontsize=17)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.xlabel('', fontsize=15)
plt.ylabel('', fontsize=15)
sns.barplot(y=tools.index, x=tools.values, palette='GnBu_d', orient='h')
toolDf['Technologies'] = multipleChoiceResponses.drop([0])['Q12_MULTIPLE_CHOICE']
tool = []
tech = []
for index, row in toolDf.iterrows():
    for col in toolDf.columns[:-1]:
        if str(row[col]) != 'nan':
            tool.append(col)
            tech.append(row[toolDf.columns[-1]])
techDf = pd.DataFrame(np.column_stack([tool, tech]), columns=['Tool', 'Tech'])
techDf = techDf[techDf['Tool'] != ' None']
def getPerc(tech, tool):
    return round(len(techDf[(techDf['Tool'] == tool) & (techDf['Tech'] == tech)])/len(multipleChoiceResponses) - 1, 6)
techByTool = pd.DataFrame(columns=list(techDf['Tool'].unique()), index=list(techDf['Tech'].unique()))
for tech in techDf['Tech'].unique():
    for tool in techDf['Tool'].unique():
        techByTool.loc[tech, tool] = getPerc(tech, tool)
plt.figure(figsize=(12, 10))
sns.heatmap(techByTool.astype(float), cmap='Blues')
title = plt.title('Famous IDE in different environments they are used', fontsize=15)
toolDf['Experience'] = multipleChoiceResponses.drop([0])['Q8']
tool = []
exp = []
for index, row in toolDf.iterrows():
    for col in toolDf.columns[:-1]:
        if str(row[col]) != 'nan':
            tool.append(col)
            exp.append(row[toolDf.columns[-1]])
expDf = pd.DataFrame(np.column_stack([tool, exp]), columns=['Tool', 'Experience'])
expDf = expDf[expDf['Tool'] != ' None']
def getPerc(exp, tool):
    return round(len(expDf[(expDf['Tool'] == tool) & (expDf['Experience'] == exp)])/len(multipleChoiceResponses) - 1, 6)
expByTool = pd.DataFrame(columns=list(expDf['Tool'].unique()), index=list(expDf['Experience'].unique()))
for exp in expDf['Experience'].unique():
    for tool in expDf['Tool'].unique():
        expByTool.loc[exp, tool] = getPerc(exp, tool)
plt.figure(figsize=(12, 10))
sns.heatmap(expByTool.astype(float), cmap='Blues')
title = plt.title('Famous IDE based on Experience', fontsize=15)
from matplotlib.colors import ListedColormap
pie=multipleChoiceResponses['Q10'].drop([0]).value_counts().plot.pie(figsize=(20, 20), 
                                                                 colormap=ListedColormap(sns.color_palette("GnBu", 100)), 
                                                                 fontsize=35, autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()
p=plt.title('Exploring ML', fontsize=50)
stackDf=multipleChoiceResponses.drop([0]).groupby(['Q12_MULTIPLE_CHOICE', 'Q10']).size().unstack().fillna(0)
stack=stackDf.plot.bar(stacked=True, figsize=(15, 12), colormap=ListedColormap(sns.color_palette("GnBu", 100)))
plt.xlabel('Environment')
stack.get_legend().set_title('ML Adoption')
import warnings  
warnings.filterwarnings('ignore')
plt.figure(figsize=(20, 10))
mlFrame = multipleChoiceResponses.drop([0])[multipleChoiceResponses['Q10'].isin(['No (we do not use ML methods)', 
                                                                        'We have well established ML methods' \
                                                                        ' (i.e., models in production for more than 2 years)'])][['Q10', 'Q6']]
print(mlFrame['Q10'].value_counts())
ax = sns.countplot(x='Q6', hue='Q10', data=mlFrame, palette='GnBu_d', orient='h')
ax.set_title('People adopting ML by Occupation', fontsize=17)
ax.get_legend().set_title('ML Adoption')
plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('Occupation', fontsize=15)
p=plt.ylabel('Count', fontsize=15)
plt.figure(figsize=(20, 10))
mlFrame = multipleChoiceResponses.drop([0])[multipleChoiceResponses['Q10'].isin(['No (we do not use ML methods)', 
                                                                        'We have well established ML methods' \
                                                                        ' (i.e., models in production for more than 2 years)'])][['Q10', 'Q7']]
ax = sns.countplot(x='Q7', hue='Q10', data=mlFrame, palette='GnBu_d', orient='h')
ax.set_title('People adopting ML by Industry', fontsize=17)
ax.get_legend().set_title('ML Adoption')
plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.xticks(fontsize=12, rotation=90)
plt.yticks(fontsize=12)
plt.xlabel('Industry', fontsize=15)
p=plt.ylabel('Count', fontsize=15)
ocI = multipleChoiceResponses.drop([0])[(multipleChoiceResponses['Q10'] == 'No (we do not use ML methods)') & 
                                        ~(multipleChoiceResponses['Q10'] == 'I do not know') & 
                                        ~(multipleChoiceResponses['Q7'].isin(['Computers/Technology', 'I am a student', 'Academics/Education'])) & 
                                        ~(multipleChoiceResponses['Q6'].isin(['Data Scientist', 'Student', 'Software Engineer']))]
def getPerc(occ, ind):
    return round(len(ocI[(ocI['Q6'] == occ) & (ocI['Q7'] == ind)])/len(ocI), 10)
occByInd = pd.DataFrame(columns=list(ocI['Q6'].unique()), index=list(ocI['Q7'].unique()))
for occ in ocI['Q6'].unique():
    for ind in ocI['Q7'].unique():
        occByInd.loc[ind, occ] = getPerc(occ, ind)
occByInd.drop(occByInd.columns[0], axis=1, inplace=True)
occByInd.drop(occByInd.index[-1], axis=0, inplace=True)
plt.figure(figsize=(12, 10))
sns.heatmap(occByInd.astype(float), cmap='GnBu_r')
title = plt.title('Occupation of non ML Adopters by Industry they work in', fontsize=15)
plt.figure(figsize=(15, 10))
genders = multipleChoiceResponses[(multipleChoiceResponses['Q1'].isin(['Male', 'Female'])) 
                                  & (~multipleChoiceResponses['Q2'].str.contains('What'))][['Q1', 'Q2']]
print(genders['Q1'].value_counts())
ax = sns.countplot(x='Q2', hue='Q1', data=genders, palette='BuGn_d', orient='h')
ax.set_title('Distribution of Respondents by Age Group and Gender', fontsize=17)
ax.get_legend().set_title('Gender')
plt.setp(ax.get_legend().get_texts(), fontsize='12') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Age Group', fontsize=15)
plt.ylabel('Count', fontsize=15)
df = multipleChoiceResponses.drop(multipleChoiceResponses.index[0])
df['Time from Start to Finish (seconds)'] = df['Time from Start to Finish (seconds)'].apply(lambda x : int(x))
plt.figure(figsize=(12, 10))
plt.title('Time taken for survey, by different age groups')
sns.boxplot(x="Time from Start to Finish (seconds)", y="Q2", data=df, palette='rainbow')
plt.ylabel('Age Group', fontsize=12)
x= plt.xlabel('Time taken', fontsize=12)
deC = multipleChoiceResponses.drop(multipleChoiceResponses.index[0])
# nDeg = multipleChoiceResponses['Q2'].nunique()
# nComp = multipleChoiceResponses['Q9'].nunique()
def getPerc(deg, comp):
    return round(len(deC[(deC['Q4'] == deg) & (deC['Q9'] == comp)])/len(deC), 6)
degByComp = pd.DataFrame(columns=list(deC['Q9'].unique()), index=list(deC['Q4'].unique()))
for deg in deC['Q4'].unique():
    for comp in deC['Q9'].unique():
        degByComp.loc[deg, comp] = getPerc(deg, comp)
degByComp.drop(degByComp.columns[0], axis=1, inplace=True)
degByComp.drop(degByComp.index[-1], axis=0, inplace=True)
plt.figure(figsize=(12, 10))
sns.heatmap(degByComp.astype(float), cmap='Reds')
title = plt.title('Yearly compensation with highest level of education', fontsize=15)
msProf = multipleChoiceResponses[multipleChoiceResponses['Q4'] == 'Master’s degree']
# msProf = msProf.value_counts()
plt.figure(figsize=(12, 10))
sns.countplot(y='Q6', data=msProf, palette=sns.dark_palette('red'))
plt.xticks(rotation=90)
plt.title('Profession of kagglers with Masters degree', fontsize=15)
yL=plt.ylabel('Profession')
proCo = multipleChoiceResponses.drop(multipleChoiceResponses.index[0])
def getPerc(prof, comp):
    return round(len(proCo[(proCo['Q6'] == prof) & (proCo['Q9'] == comp)])/len(proCo), 6)
profByComp = pd.DataFrame(columns=list(proCo['Q9'].unique()), index=list(proCo['Q6'].unique()))
for prof in proCo['Q6'].unique():
    for comp in proCo['Q9'].unique():
        profByComp.loc[prof, comp] = getPerc(prof, comp)
profByComp.drop(profByComp.columns[0], axis=1, inplace=True)
profByComp.drop(profByComp.index[-1], axis=0, inplace=True)
plt.figure(figsize=(14, 12))
sns.heatmap(profByComp.astype(float), cmap='Reds')
title = plt.title('Yearly compensation with profession', fontsize=15)
import warnings  
warnings.filterwarnings('ignore')
def getDataScientistPerc(comp):
    return round(len(deC[(deC['Q6'] == 'Data Scientist') & (deC['Q9'] == comp)])/len(deC), 6)
ms = pd.DataFrame(index=list(degByComp.columns), columns=['% with MS'])
ms['% with MS'] = degByComp.loc['Master’s degree', :]
ms['% who are Data Scientist'] = [getDataScientistPerc(x) for x in ms.index]
p=sns.jointplot(x='% with MS', y='% who are Data Scientist', data=ms, kind='kde', color='red')



