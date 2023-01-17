

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

# =pd.read_csv('multipleChoiceResponses.csv')

multipleChoiceResponses = pd.read_csv("../input/multipleChoiceResponses.csv" , encoding='ISO-8859-1')
multipleChoiceResponses.shape
multipleChoiceResponses['GenderSelect']=multipleChoiceResponses['GenderSelect'].apply(lambda x: 'No Answer' if x=='Non-binary, genderqueer, or gender non-conforming' else x)
a4_dims = (8,4)

fig, ax = plt.subplots(figsize=a4_dims)

# plt.xticks(rotation='vertical')

sns.countplot(multipleChoiceResponses['GenderSelect'],ax=ax,orient = "v")
multipleChoiceResponses['Country']=multipleChoiceResponses['Country'].fillna('Not_Answered')

a4_dims = (16,4)

fig, ax = plt.subplots(figsize=a4_dims)

plt.xticks(rotation='vertical')

multipleChoiceResponses.groupby('Country')['Country'].count().sort_values( ascending=False).plot(kind='BAR')
countr=multipleChoiceResponses.groupby('Country')['Country'].count().sort_values( ascending=False)

countr=countr/sum(countr)

a4_dims = (16,4)

fig, ax = plt.subplots(figsize=a4_dims)

plt.xticks(rotation='vertical')

countr.plot(kind='BAR',ax=ax)
multipleChoiceResponses['EmploymentStatus']=multipleChoiceResponses['EmploymentStatus'].fillna('I prefer not to say')

a4_dims = (8,4)

fig, ax = plt.subplots(figsize=a4_dims)

plt.xticks(rotation='vertical')

sns.countplot(multipleChoiceResponses['EmploymentStatus'],ax=ax)
sns.countplot(multipleChoiceResponses['JobSkillImportancePython'])
multipleChoiceResponses['EmploymentStatus']=multipleChoiceResponses['EmploymentStatus'].fillna('I prefer not to say')

a4_dims = (8,5)

fig, ax = plt.subplots(figsize=a4_dims)

plt.xticks(rotation='vertical')

sns.countplot(multipleChoiceResponses['LanguageRecommendationSelect'])