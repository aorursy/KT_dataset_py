import os
import gc
import warnings
import re
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
#matplotlib.rc['font.size'] = 9.0
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/survey_2014.csv')
data.shape
data.sample(10)
data.nunique()
gender_clean = {
    "female":"Female",
    "male":"Male",
    "Male":"Male",
    "male-ish":"Male",
    "maile":"Male",
    "trans-female":"Female",
    "cis female":"Female",
    "f":"Female",
    "m":"Male",
    "M":"Male",
    "something kinda male?":"Male",
    "cis male":"Male",
    "woman":"Female",
    "mal":"Male",
    "male (cis)":"Male",
    "queer/she/they":"Female",
    "non-binary":"Unspecified",
    "femake":"Female",
    "make":"Male",
    "nah":"Unspecified",
    "all":"Unspecified",
    "enby":"Unspecified",
    "fluid":"Unspecified",
    "genderqueer":"Unspecified",
    "androgyne":"Unspecified",
    "agender":"Unspecified",
    "cis-female/femme":"Female",
    "guy (-ish) ^_^":"Male",
    "male leaning androgynous":"Male",
    "man":"Male",
    "male ":"Male",
    "trans woman":"Female",
    "msle":"Male",
    "neuter":"Unspecified",
    "female (trans)":"Female",
    "queer":"Unspecified",
    "female (cis)":"Female",
    "mail":"Male",
    "a little about you":"Unspecified",
    "malr":"Male",
    "p":"Unspecified",
    "femail":"Female",
    "cis man":"Male",
    "ostensibly male, unsure what that really means":"Male",
    "female ":"Female",
    "Female":"Female",
    "Male-ish":"Male"
}

data.Gender = data.Gender.str.lower()
data.Gender = data.Gender.apply(lambda x: gender_clean[x])
f, ax = plt.subplots(1,2, figsize=(15,7))
ax1 = ax[0].pie(list(data['Gender'].value_counts()), 
                   labels=['Male','Female','Unspecified'],
                  autopct='%1.1f%%', shadow=True, startangle=90,
             colors=['#66b3ff','#ff9999','#99ff99'])
ax[0].set_title("Gender Distribution")
ax[1].set_title("Distribution of Ages")
ax2 = sns.distplot(data.Age.clip(15,70), ax=ax[1])
#Extraction of basic stats from all numeric columns
pd.DataFrame(data.Age.clip(15,60).describe())
sns.set_style("darkgrid")
plt.figure(figsize=(15,20))
sns.countplot(y='Country', data=data, 
              orient='h', order=data.Country.value_counts().index)
plt.show()
f, ax = plt.subplots(1,2, figsize=(15,10))
patches, texts, autotexts = ax[0].pie(list(data['no_employees'].value_counts()), 
                   labels=['6-25', '26-100', '>1000', '100-500', '1-5', '500-1000'],
                  autopct='%1.1f%%', shadow=True, startangle=90)
new = ax[1].pie(list(data['remote_work'].value_counts()),
                                     labels=['Non-Remote', 'Remote Work'],
                                     autopct='%1.1f%%', shadow=True, startangle=0,
                                        colors=['#66b3ff','#ff9999'])
sns.set_style("whitegrid")
plt.figure(figsize=(15,7))
sns.countplot(x='leave', data=data, order=data.leave.value_counts().index, 
              hue='tech_company', color='r')
plt.xlabel("How East/Difficult is it to take a leave?")
plt.ylabel("# Reponses")
company_characs = [
    "treatment",
    "benefits",
    "care_options",
    "wellness_program",
]
sns.set_style("darkgrid")
company_chars_corr = data[company_characs].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                                                                            min_periods=1)
plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(company_chars_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title("Correlation Heatmap of Company Policies \ntowards employee's wellness");
wellbeing_indicators = [
    'seek_help',
    'mental_health_consequence',
    'obs_consequence',
    'mental_health_interview',
    'phys_health_consequence'
]
wellbeing_indicators_corr = data[wellbeing_indicators].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                                                                            min_periods=1)
plt.figure(figsize = (12, 9))

# Heatmap of correlations
sns.heatmap(wellbeing_indicators_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title("Correlation Heatmap of Well Being\n indicators of Employees");
wellbeing_policy_corr = data[wellbeing_indicators + company_characs].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                                                                            min_periods=1)
plt.figure(figsize = (12, 9))

# Heatmap of correlations
sns.heatmap(wellbeing_policy_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title("Correlation Heatmap between Employee Well Being \nand Company Policies");
plt.figure(figsize=(20,20))
wordcloud = WordCloud(
                          background_color='white',
                          width=1024,
                          height=1024,
                         ).generate(re.sub(r'[^\w\s]',''," ".join(list(data.comments.unique()[1:]))))
plt.imshow(wordcloud)
