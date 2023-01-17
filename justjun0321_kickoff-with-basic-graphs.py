# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import dataset
free_from = pd.read_csv('../input/freeFormResponses.csv')
free_from.columns = free_from.iloc[0]
free_from = free_from.iloc[1:]
free_from.head()
multiple = pd.read_csv('../input/multipleChoiceResponses.csv')
multiple.columns = multiple.iloc[0]
multiple = multiple.iloc[1:]
multiple.head()
sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='What is your gender? - Selected Choice',data=multiple,
              order = multiple['What is your gender? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('What is your gender?')
sns.countplot(y='What is your age (# years)?',data=multiple,
              order = multiple['What is your age (# years)?'].value_counts().index)
plt.ylabel('')
plt.title('What is your age (# years)?')
sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',data=multiple,
             order = multiple['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().index)
plt.ylabel('')
plt.title('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?')
sns.set(rc={'figure.figsize':(12,12)})
sns.countplot(y='In which country do you currently reside?',data=multiple,
             order = multiple['In which country do you currently reside?'].value_counts().index)
plt.ylabel('')
plt.title('In which country do you currently reside?')
sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='Which best describes your undergraduate major? - Selected Choice',data=multiple,
             order = multiple['Which best describes your undergraduate major? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('Which best describes your undergraduate major?')
sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(y='Select the title most similar to your current role (or most recent title if retired): - Selected Choice',data=multiple,
             order = multiple['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('Select the title most similar to your current role (or most recent title if retired)')
sns.set(rc={'figure.figsize':(7,4)})
sns.countplot(y='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',data=multiple,
             order = multiple['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().index)
plt.ylabel('')
plt.title('What is the highest level of formal education that you have attained or plan to attain within the next 2 years?')
sns.countplot(y='In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice',data=multiple,
             order = multiple['In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('In what industry is your current employer/contract (or your most recent employer if retired)?')
sns.set(rc={'figure.figsize':(7,3)})
sns.countplot(y='How many years of experience do you have in your current role?',data=multiple,
             order = multiple['How many years of experience do you have in your current role?'].value_counts().index)
plt.ylabel('')
plt.title('How many years of experience do you have in your current role?')
sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(y='What is your current yearly compensation (approximate $USD)?',data=multiple,
             order = multiple['What is your current yearly compensation (approximate $USD)?'].value_counts().index)
plt.ylabel('')
plt.title('What is your current yearly compensation (approximate $USD)?')
# unstack Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years?
df = multiple.loc[:,"Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Selected Choice - Jupyter/IPython":"Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - Other - Text"]
df.fillna(0,inplace=True)
s = df.astype(bool).sum(axis=0)
s.index
s = s.rename(lambda x: x.replace("Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? (Select all that apply) - ", ''))
s = s.rename(lambda x: x.replace("Selected Choice - ", ''))
s = s.rename(lambda x: x.replace("Other - ", ''))
sns.barplot(s.index, s.values, alpha=0.8)
plt.xticks(rotation=90)
plt.title('IDE distribution')
sns.set(rc={'figure.figsize':(7,5)})
sns.countplot(y='What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice',data=multiple,
             order = multiple['What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('What programming language would you recommend an aspiring data scientist to learn first?')
sns.countplot(y='What specific programming language do you use most often? - Selected Choice',data=multiple,
             order = multiple['What specific programming language do you use most often? - Selected Choice'].value_counts().index)
plt.ylabel('')
plt.title('What specific programming language do you use most often?')
df = multiple.loc[:,"What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Python":"What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice - Other"]
df.fillna(0,inplace=True)
s = df.astype(bool).sum(axis=0)
s = s.rename(lambda x: x.replace("What programming languages do you use on a regular basis?", ''))
s = s.rename(lambda x: x.replace("(Select all that apply)", ''))
s = s.rename(lambda x: x.replace(" - Selected Choice - ", ''))
s = s.rename(lambda x: x.replace("Other - ", ''))
sns.barplot(s.index, s.values, alpha=0.8)
plt.xticks(rotation=90)
plt.title('IDE distribution')
