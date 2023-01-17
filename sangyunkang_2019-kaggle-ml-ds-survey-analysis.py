# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
multi_response_path = '/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv'

text_response_path = '/kaggle/input/kaggle-survey-2019/other_text_responses.csv'

question_path = '/kaggle/input/kaggle-survey-2019/questions_only.csv'

survey_schema_path = '/kaggle/input/kaggle-survey-2019/survey_schema.csv'
# reading csv file  

multi_response = pd.read_csv(multi_response_path)

text_response = pd.read_csv(text_response_path)

question = pd.read_csv(question_path)

survey = pd.read_csv(survey_schema_path)
# basic function of python

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sb

%matplotlib inline

import numpy as np

from pandas import ExcelWriter

from pandas import ExcelFile

import xlrd

from scipy import stats

from datetime import datetime



# feature selection

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import feature_selection



# oversampling

from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



# building the models

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



# import tensorflow

# from tensorflow.contrib.keras import models, layers

# from tensorflow.contrib.keras import activations, optimizers, losses



# standardize the vaiable

from sklearn.preprocessing import StandardScaler



# from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split



# validation

from sklearn.metrics import confusion_matrix,classification_report
multi_response.dtypes
print (f'Shape of multiple choice responses: {multi_response.shape}')

print (f'Shape of questions only: {question.shape}')

print (f'Shape of survey schema: {survey.shape}')

print (f'Shape of text responses: {text_response.shape}')
multi_response.head()
text_response.head()
question.head()
survey['2019 Kaggle Machine Learning and Data Science Survey']
multi_response = multi_response.drop([0])

multi_response = multi_response.reset_index(drop=True)

multi_response.head()
X_enc = multi_response.copy()
X_enc = pd.get_dummies(X_enc, columns=['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8',\

                                       'Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8',\

                                       'Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12',\

                                       'Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8',\

                                       'Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12',\

                                       'Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8',\

                                       'Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12',\

                                       'Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8',\

                                       'Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12',\

                                       'Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8',\

                                       'Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12',\

                                       'Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8',\

                                       'Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12',\

                                       'Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5',\

                                       'Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8',\

                                       'Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12',\

                                       'Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8',\

                                       'Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7',\

                                       'Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6',\

                                       'Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8',\

                                       'Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12',\

                                       'Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8',\

                                       'Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12',\

                                       'Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8',\

                                       'Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12',\

                                       'Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8',\

                                       'Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12',\

                                       'Q32_Part_1','Q32_Part_2','Q32_Part_3','Q32_Part_4','Q32_Part_5','Q32_Part_6','Q32_Part_7','Q32_Part_8',\

                                       'Q32_Part_9','Q32_Part_10','Q32_Part_11','Q32_Part_12',\

                                       'Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8',\

                                       'Q33_Part_9','Q33_Part_10','Q33_Part_11','Q33_Part_12',\

                                       'Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8',\

                                       'Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12'])
X_enc.head()
# X_enc = X_enc.drop(['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8',\

#                                        'Q12_Part_1','Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8',\

#                                        'Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12',\

#                                        'Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8',\

#                                        'Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12',\

#                                        'Q16_Part_1','Q16_Part_2','Q16_Part_3','Q16_Part_4','Q16_Part_5','Q16_Part_6','Q16_Part_7','Q16_Part_8',\

#                                        'Q16_Part_9','Q16_Part_10','Q16_Part_11','Q16_Part_12',\

#                                        'Q17_Part_1','Q17_Part_2','Q17_Part_3','Q17_Part_4','Q17_Part_5','Q17_Part_6','Q17_Part_7','Q17_Part_8',\

#                                        'Q17_Part_9','Q17_Part_10','Q17_Part_11','Q17_Part_12',\

#                                        'Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8',\

#                                        'Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12',\

#                                        'Q20_Part_1','Q20_Part_2','Q20_Part_3','Q20_Part_4','Q20_Part_5','Q20_Part_6','Q20_Part_7','Q20_Part_8',\

#                                        'Q20_Part_9','Q20_Part_10','Q20_Part_11','Q20_Part_12',\

#                                        'Q21_Part_1','Q21_Part_2','Q21_Part_3','Q21_Part_4','Q21_Part_5',\

#                                        'Q24_Part_1','Q24_Part_2','Q24_Part_3','Q24_Part_4','Q24_Part_5','Q24_Part_6','Q24_Part_7','Q24_Part_8',\

#                                        'Q24_Part_9','Q24_Part_10','Q24_Part_11','Q24_Part_12',\

#                                        'Q25_Part_1','Q25_Part_2','Q25_Part_3','Q25_Part_4','Q25_Part_5','Q25_Part_6','Q25_Part_7','Q25_Part_8',\

#                                        'Q26_Part_1','Q26_Part_2','Q26_Part_3','Q26_Part_4','Q26_Part_5','Q26_Part_6','Q26_Part_7',\

#                                        'Q27_Part_1','Q27_Part_2','Q27_Part_3','Q27_Part_4','Q27_Part_5','Q27_Part_6',\

#                                        'Q28_Part_1','Q28_Part_2','Q28_Part_3','Q28_Part_4','Q28_Part_5','Q28_Part_6','Q28_Part_7','Q28_Part_8',\

#                                        'Q28_Part_9','Q28_Part_10','Q28_Part_11','Q28_Part_12',\

#                                        'Q29_Part_1','Q29_Part_2','Q29_Part_3','Q29_Part_4','Q29_Part_5','Q29_Part_6','Q29_Part_7','Q29_Part_8',\

#                                        'Q29_Part_9','Q29_Part_10','Q29_Part_11','Q29_Part_12',\

#                                        'Q30_Part_1','Q30_Part_2','Q30_Part_3','Q30_Part_4','Q30_Part_5','Q30_Part_6','Q30_Part_7','Q30_Part_8',\

#                                        'Q30_Part_9','Q30_Part_10','Q30_Part_11','Q30_Part_12',\

#                                        'Q31_Part_1','Q31_Part_2','Q31_Part_3','Q31_Part_4','Q31_Part_5','Q31_Part_6','Q31_Part_7','Q31_Part_8',\

#                                        'Q31_Part_9','Q31_Part_10','Q31_Part_11','Q31_Part_12',\

#                                        'Q32_Part_1','Q32_Part_2','Q32_Part_3','Q32_Part_4','Q32_Part_5','Q32_Part_6','Q32_Part_7','Q32_Part_8',\

#                                        'Q32_Part_9','Q32_Part_10','Q32_Part_11','Q32_Part_12',\

#                                        'Q33_Part_1','Q33_Part_2','Q33_Part_3','Q33_Part_4','Q33_Part_5','Q33_Part_6','Q33_Part_7','Q33_Part_8',\

#                                        'Q33_Part_9','Q33_Part_10','Q33_Part_11','Q33_Part_12',\

#                                        'Q34_Part_1','Q34_Part_2','Q34_Part_3','Q34_Part_4','Q34_Part_5','Q34_Part_6','Q34_Part_7','Q34_Part_8',\

#                                        'Q34_Part_9','Q34_Part_10','Q34_Part_11','Q34_Part_12'],axis=1)
#FinalData = pd.concat([multi_response,X_enc], axis=1)
X_enc = X_enc.drop(['Q2_OTHER_TEXT','Q5_OTHER_TEXT','Q9_OTHER_TEXT','Q12_OTHER_TEXT','Q13_OTHER_TEXT','Q14_Part_1_TEXT','Q14_Part_2_TEXT','Q14_Part_3_TEXT',\

                            'Q14_Part_4_TEXT','Q14_Part_5_TEXT','Q14_OTHER_TEXT','Q16_OTHER_TEXT','Q17_OTHER_TEXT','Q18_OTHER_TEXT','Q19_OTHER_TEXT',\

                            'Q20_OTHER_TEXT','Q21_OTHER_TEXT','Q24_OTHER_TEXT','Q25_OTHER_TEXT','Q26_OTHER_TEXT','Q27_OTHER_TEXT','Q28_OTHER_TEXT',\

                            'Q29_OTHER_TEXT','Q30_OTHER_TEXT','Q31_OTHER_TEXT','Q32_OTHER_TEXT','Q33_OTHER_TEXT','Q34_OTHER_TEXT'],axis=1)
X_enc.dtypes
X_enc.head()
plt.figure(figsize=(10,5))

sb.countplot(x = 'Q2',data = X_enc)

plt.show()
plt.figure(figsize=(10,5))

sb.countplot(x = 'Q3',data = X_enc)

plt.xticks(rotation=90)

plt.show()
# Let's see top10 countries more detailed.

X_enc['Q3'].value_counts()[:10].reset_index()
top_10_country = X_enc['Q3'].value_counts()[:10].reset_index()
#Pie Chart

top_10_country['Q3'] = top_10_country['Q3']*100/top_10_country['Q3'].sum()

#explode=(0.1,0,0,0,0,0,0,0,0,0)

colors=['g','b','r','c','m','y','r','k','m','b']

fig, ax=plt.subplots(figsize=(10,5))

ax.pie(top_10_country['Q3'],labels = top_10_country['index'],colors=colors)

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,5))

sb.countplot(x = 'Q1',hue = 'Q2',data = X_enc)

plt.show()
job = pd.DataFrame(X_enc.iloc[1:]['Q5'].value_counts().sort_values(ascending=False)).reset_index().head(25)

job
plt.figure(figsize=(12,5))

sb.barplot(x=job['Q5'],y=job['index'], palette='viridis')

plt.xlabel('Count')

plt.ylabel('', fontsize=10)

plt.xticks(rotation=0)

plt.yticks(rotation=0)

plt.title('Job title')
job = X_enc['Q5'].value_counts()

job
#total = int(X_enc.iloc[1]['Q19'])

plt.figure(figsize=(12,5))

ax = sb.barplot(X_enc.groupby(['Q19']).size().reset_index(name='counts')['Q19'][:-1], X_enc.groupby(['Q19']).size().reset_index(name='counts')['counts'][:-1])
# checking the imbalance

sb.countplot(x='Q10',data=X_enc,palette='RdBu_r') # Barplot for the dependent variable
X_enc['Q10'].value_counts()
# # Category variables -> Numerical variables

list_feat=['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q10','Q11','Q14','Q15','Q19','Q22','Q23']
for feature in list_feat:

    labels = X_enc[feature].astype('category').cat.categories.tolist()

    replace_map_comp = {feature : {k: v for k,v in zip(labels,list(range(0,len(labels)+1)))}}



    X_enc.replace(replace_map_comp, inplace=True)
X_enc.head()
#Rename the columns for better understanding

data = X_enc.rename(columns={

             'Time from Start to Finish (seconds)':'Time duration',

             'Q1':'Age',

             'Q2':'Gender',

             'Q3':'Country',

             'Q4':'Education level',

             'Q5':'Job title',

             'Q6':'Company size',

             'Q7':'The number of data scientist in company',

             'Q8':'ML implementaion in company',

             #Q9: Activities that make up an important part of your role at work(Multiple choices)

            'Q10':'Yearly compensation',

            'Q11':'Investment on ML',

            #Q12: Favorite media sources that report on data science topics(Multiple choices)

            #Q13: Which platforms for studying data science(Multiple choices)

            'Q14':'Primary tool used for ML',

            'Q15':'Coding experience',

            #Q16: Integrated development environments (IDE's) you use on a regular basis(Multiple choices)

            #Q17: Hosted notebook products you use on a regular basis(Multiple choices)

            #Q18: Programming languages you use on a regular basis(Multiple choices)

            'Q19': 'Programming language to learn first',

            #Q20: Data visualization libraries or tools you use on a regular basis(Multiple choices)

            #Q21: Specialized hardware you use on a regular basis(Multiple choices)

            'Q22':'Experience in TPU',

            'Q23': 'Years in using machine learning methods'

            #Q24: ML algorithms you use on a regular basis(Multiple choices)

            #Q25: Categories of ML tools you use on a regular basis(Multiple choices)

            #Q26: Categories of computer vision methods you use on a regular basis(Multiple choices)

            #Q27: Natural language processing (NLP) methods you use on a regular basis(Multiple choices)

            #Q28: Machine learning frameworks you use on a regular basis(Multiple choices)

            #Q29: Cloud computing platforms you use on a regular basis(Multiple choices)

            #Q30: Specific cloud computing products you use on a regular basis(Multiple choices)

            #Q31: Specific big data / analytics products you use on a regular basis(Multiple choices)

            #Q32: Machine learning products you use on a regular basis(Multiple choices)

            #Q33: Automated machine learning tools (or partial AutoML tools) you use on a regular basis(Multiple choices)

            #Q34: Relational database products do you use on a regular basis(Multiple choices)



})
data.head()
data.shape
data['Time duration'] = data['Time duration'].apply(pd.to_numeric)

#data['Time duration'] = data['Time duration'].astype('int')
data.isnull() # Checking missing values
# There are too many variables so heatmap is not useful

sb.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
missing_data = data.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")
data.shape
final_data = data.dropna()
final_data.shape
final_data.head()
# Import the random forest model.

from sklearn.ensemble import RandomForestClassifier



X = final_data.iloc[:, np.r_[:,0:9,10:218]]  #independent columns

y = final_data.iloc[:, np.r_[:,9]]   #target column: salary



# This line instantiates the model. 

model3 = RandomForestClassifier() 

# Fit the model on your training data.

model3.fit(X, y)

feature_importances = pd.DataFrame(model3.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances
(pd.Series(model3.feature_importances_, index=X.columns).nlargest(20).plot(kind='barh'))
from xgboost import XGBClassifier

from xgboost import plot_importance



X = final_data.iloc[:, np.r_[:,0:9,10:218]]  #independent columns

y = final_data.iloc[:, np.r_[:,9]]   #target column: salary

# fit model no training data

model2 = XGBClassifier()

model2.fit(X,y)

# feature importance

print(model2.feature_importances_)

# plot feature importance



plt.figure(figsize=(3,6))

plot_importance(model2,max_num_features=20)

plt.show()
X = final_data.iloc[:, np.r_[:,0:9,10:218]]  #independent columns

y = final_data.iloc[:, np.r_[:,9]]   #target column: salary

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model1 = ExtraTreesClassifier()

model1.fit(X,y)

print(model1.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model1.feature_importances_, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.show()
# X = final_data.iloc[:, np.r_[:,0:9,10:218]]  #independent columns

# y = final_data.iloc[:, np.r_[:,9]]   #target column: Salary

# #get correlations of each features in dataset

# corrmat = final_data.corr()

# top_corr_features = corrmat.index

# plt.figure(figsize=(20,20))

# #plot heat map

# g=sb.heatmap(final_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
multi_response.head()
req = multi_response[1:]
req['Q7']
####Checking the proportion of survey takers ----> large interest shown by 25-29 year old age group ----> find the distribtuion ohis population by country

age_group = req.groupby(['Q1'],as_index=False).count().reset_index().loc[:,'Q1':'Time from Start to Finish (seconds)']
age_group.sort_values('Time from Start to Finish (seconds)', inplace=True)
# import seaborn as sns

# sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

# ax = sns.barplot(x="Q1", y="Time from Start to Finish (seconds)", data=age_group)

# ax.set(xlabel='Age Group Participating In The Survey', ylabel='Number of people participating in the survey')
#Age group 25-29

age_25 = req[req.Q1 == "25-29"]
age_25_educ = age_25.groupby(['Q4'],as_index=False)[['Q1']].count()
age_25_educ
age_25_educ.sort_values('Q1',inplace=True)
#Preliminary visualization ---> most people in this age group are pursuing a Master's degree,followed by Bachelor's, Master's and phD

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(35,15))

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q4", y="Q1", data=age_25_educ)

ax.set_title("Proportion of participants between 25-29 years pursuing different professions",fontsize=20)

ax.set(xlabel='Professional Degree being pursued by age_group 25-29', ylabel='Number of people participating in the survey')
#AGE GROUP DISRTRIBUTION BY GENDER --->

gender_age_25 = age_25.groupby(['Q2'],as_index=False)[['Q1']].count()
gender_age_25.sort_values('Q1',inplace=True)
gender_age_25.reset_index(drop=True)
#Preliminary visualization ---> most people in this age group are pursuing a Master's degree,followed by Bachelor's, Master's and phD

import seaborn as sns

sns.set(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(20,10))

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q2", y="Q1", data=gender_age_25)

ax.set_title("Gender Proportion of participants between 25-29 years ",fontsize=20)

ax.set(xlabel='Gender of participant in the age_group 25-29', ylabel='Number of people participating in the survey')
#between ages 25-29 by country

country_age_25 = age_25.groupby(['Q3'],as_index=False)[['Q1']].count()
country_age_25.reset_index(drop=True)
# Prepare Data

country_age_25.sort_values('Q1',inplace=True)

#country.reset_index(inplace=True)



# Draw plot

import matplotlib.patches as patches



fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)

ax.vlines(x=country_age_25.index, ymin=0, ymax=country_age_25.Q1, color='firebrick', alpha=0.7, linewidth=20)



# Annotate Text

for i, q1 in enumerate(country_age_25.Q1):

    ax.text(i, q1+0.5, round(q1, 1), horizontalalignment='center')





# Title, Label, Ticks and Ylim

ax.set_title('Bar Chart for Proportion of Survey Attempts by participants between 25-29 by Country distribution', fontdict={'size':22})

ax.set(ylabel='Number of 25-29 year old people participated in survey', ylim=(0, 1500))

ax.set(xlabel='Country participating in Survey')

plt.xticks(country_age_25.index, country_age_25.Q3.str.upper(), rotation=60, horizontalalignment='right', fontsize=12)



# Add patches to color the X axis labels

p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)

p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)

fig.add_artist(p1)

fig.add_artist(p2)

plt.show()
########25-29 year olds different professions

profession_age_25 = age_25.groupby(['Q5'],as_index=False)[['Q1']].count()
profession_age_25.sort_values('Q1',inplace=True)
profession_age_25.reset_index(drop=True)
profession_age_25['Q5'] = profession_age_25[profession_age_25['Q5']!=0]
#Preliminary visualization ---> most people in this age group are pursuing a Master's degree,followed by Bachelor's, Master's and phD

import seaborn as sns

sns.set(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(25,10))

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q5", y="Q1", data=profession_age_25)

ax.set_title("Work Profession of participants between 25-29 years ",fontsize=20)

ax.set(xlabel='Profession of participant in the age_group 25-29', ylabel='Number of people participating in the survey')
#LARGE NOT EMPLOYED CROWD INTERESTED....OTHER <----> proportion of unemployed people interested by country

#increase number of unemployed people in specific country <----> look at growth of machine learning in these countries

unemployed = req[req['Q5']=='Not employed']
unemployed.reset_index(drop=True)
#CHECK PROPORTION OF UNEMPLOYED PEOPLE BY EDUCATION...AND GENDER...AND COUNTRY...AND AGE GROUPS

unemployed_by_country = unemployed.groupby(['Q3','Q1']).count().reset_index().loc[:,'Q3':'Time from Start to Finish (seconds)']
unemployed_by_country.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
unemployed_by_country
import pandas as pd



pivot = pd.pivot_table(unemployed_by_country, index='Q3', columns='Q1', values='Frequency')
ax = pivot.plot(kind='bar', figsize=(40,20),fontsize=20)

ylab = ax.set_ylabel('Number of paticipants Unemployed',fontsize=20)

xlab = ax.set_xlabel('Country',fontsize=20)

#stacked=True,
###AREAS OF INTERESTED UNEMPLOYED NUMBER IN THE AGE-GROUPS 25-29,30-34,35-39,40-44,45-49,

unemployed.groupby(['Q3', 'Q1']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))
####Check unemployed rates by highest level of education attained of participants

ax = unemployed.groupby(['Q4', 'Q1']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

ax.set_xlabel("Professional degree attained by participant",fontsize=15)

ax.set_ylabel("Number of Unemployed Participants by Age",fontsize=15)

ax.legend(title="Age-Groups of participants")
#Number of unemployed people giving the survey by country

ax = unemployed.groupby(['Q3', 'Q4']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

ax.set_xlabel("Country of participant",fontsize=15)

ax.set_ylabel("Number of unemployed participants by acquired level of education",fontsize=15)

ax.set_title("Proportion of acquired levels of education by unemployed participants across different countries")

ax.legend(title = "List of levels of education of participants")
#Distribution of various levels of education across different countries

ax = req.groupby(['Q3', 'Q4']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

ax.set_xlabel("Countries",fontsize=15)

ax.set_ylabel("Number of People posessing a certain level of education",fontsize=15)

ax.set_title("Proportion of level of education of different participants across the countries",fontsize=20)

ax.legend(title="Professional degree attained by participant")
#Gender<--->Education Level proportions

r = req[(req['Q2']=="Male")|(req['Q2']=="Female")]
re = r[r['Q4'] != 0]
re.groupby(['Q4', 'Q2']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

ax.set_xlabel("Level of education attained",fontsize=15)

ax.set_ylabel("Number of People posessing a certain level of education distributed by gender",fontsize=15)

ax.set_title("Gender distribution of Highest level of education attained",fontsize=20)

ax.legend(title="Gender of participant")
#Distribution of gender and profession

ax = req.groupby(['Q5', 'Q2']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

ax.set_xlabel("Current Role of Participant",fontsize=15)

ax.set_ylabel("Number of People distributed by Gender",fontsize=15)

ax.set_title("Gender distribution of Work Roles of Participants",fontsize=20)

ax.legend(title="Gender of participant")
##############ABOVE PLOTS CAN BE GENERATED FOR EACH COUNTRY ???

l = req.Q3.unique()

for num in l:

    if num in ['India','']:

        m = req[req['Q3']==num]

        ax = m.groupby(['Q5', 'Q2']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

        ax.set_xlabel("Current Role of Participant",fontsize=15)

        ax.set_ylabel("Number of People distributed by Gender",fontsize=15)

        ax.set_title("Gender distribution of Work Roles of Participants in {}".format(num),fontsize=20)

        ax.legend(title="Gender of participant")
l = req.Q3.unique()

for num in l:

    m = req[req['Q3']==num]

    ax = m.groupby(['Q4', 'Q2']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

    ax.set_xlabel("Current Role of Participant",fontsize=15)

    ax.set_ylabel("Number of People distributed by Gender",fontsize=15)

    ax.set_title("Gender distribution of Work Roles of Participants in {}".format(num),fontsize=20)

    ax.legend(title="Gender of participant")
################GENDER IMBALANCED CATEGORY <----> DEVELOP INSIGHTS BASED ON BALANCED FEATURE <---> DISTRIBUTION OF WORK PROFESSIONS AND SIZE OF COMPANY

l = req.Q3.unique()

for num in l:

    m = req[(req['Q3']==num)&(req['Q6']!=0)&(req['Q5']!=0)]

    ax = m.groupby(['Q6', 'Q5']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

    ax.set_xlabel("Current Size of Company Employed In",fontsize=15)

    ax.set_ylabel("Number of People distributed by work position",fontsize=15)

    ax.set_title("Work Roles of Participants in {} distributed by Size Company".format(num),fontsize=20)

    ax.legend(title="Work Role of participant")
###########HAVE TO ANALYZE OTHER RESPONSES FOR ROLES <---> what profession exists more with a particular company size<---> focus on what softwares/ technologies are used

#******compare the level of work experience and the advancement in technologies and softwares used across different countries*********#
#####Compare the tools and softwares used by data scientists, 

##############ABOVE PLOTS CAN BE GENERATED FOR EACH COUNTRY ???

l = req.Q3.unique()

for num in l:

    m = req[req['Q3']==num]

    ax = m.groupby(['Q4', 'Q2']).size().unstack().plot(kind='bar', stacked=True,figsize=(20,10))

    ax.set_xlabel("Highest Education Attained By Participant",fontsize=15)

    ax.set_ylabel("Number of People distributed by Gender",fontsize=15)

    ax.set_title("Gender distribution of Education of Participants in {}".format(num),fontsize=20)

    ax.legend(title="Gender of participant")
s = ['Q12_Part_1','Q12_Part_2','Q12_Part_3']

sources = ['Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

l = list()

for i in sources:

    df = req.groupby([i]).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Source_for_learning'}, inplace = True)

    l.append(df)
fav_sources_to_learn_from = pd.concat(l)
fav_sources_to_learn_from = fav_sources_to_learn_from[fav_sources_to_learn_from['Source_for_learning']!=0]
fstlf = fav_sources_to_learn_from.sort_values('Frequency',ascending=False)
f = fstlf.reset_index(drop=True)

f
import seaborn as sns

sns.set(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(90,20))

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Source_for_learning", y="Frequency", data=f)

ax.set_title("Distribution of different sources of learning used by our participants ".format(i))

ax.set(xlabel='Source for learning used', ylabel='Frequency')

ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
l = req.Q3.unique()

sources = ['Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

l = list()

for i in sources:

    df = req.groupby([i,'Q3']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Source_for_learning'}, inplace = True)

    l.append(df)
sources_to_learn_by_country = pd.concat(l)
sources_to_learn_by_country = sources_to_learn_by_country[sources_to_learn_by_country.Source_for_learning!=0]
stl = sources_to_learn_by_country.reset_index(drop=True)

stl
#####REQUIRED COUNTRY LIST TO ANALYZE COMPARISON OF PREFERENCE OF DIFFERENT SOURCES OF LEARNING

countries = req.Q3.unique()

for i in countries:

    d = stl[stl['Q3']==i]

    dr = d.reset_index(drop=True)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(90,20))

#     tips = sns.load_dataset("tips")

    ax = sns.barplot(x="Source_for_learning", y="Frequency", data=dr)

    ax.set_title("Distribution of different sources of learning preferred by participants in {}".format(i),fontsize=30)

    ax.set(xlabel='Source for learning used', ylabel='Frequency')

    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

    ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)



#IRELAND --> Increased Youtube

#SLACK Communities ---> Russia

#SAUDI, ALGERIA--> Youtube over kaggle
########REQUIRED SOURCE FOR LEARNING LIST TO ANALYZE THE USAGE OF EACH SOURCE ACROSS DIFFERENT COUNTRIES

sources = stl.Source_for_learning.unique()

for i in sources:

    d = stl[stl['Source_for_learning']==i]

    dr = d.reset_index(drop=True)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(110,30))

#     tips = sns.load_dataset("tips")

    ax = sns.barplot(x="Q3", y="Frequency", data=dr)

    ax.set_title("Distribution of different participants from different countries using {}".format(i),fontsize=30)

    ax.set(xlabel='Source for learning used', ylabel='Frequency')

    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)

    ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
#Find distribution of main source of learning by age group --> see which audience is better target audience

l = req.Q3.unique()

sources = ['Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

l = list()

for i in sources:

    df = req.groupby([i,'Q1']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Source_for_learning'}, inplace = True)

    l.append(df)
sources_to_learn_by_age = pd.concat(l)
sources_to_learn_by_age = sources_to_learn_by_age[sources_to_learn_by_age.Source_for_learning!=0]
stla = sources_to_learn_by_age.reset_index(drop=True)

stla
####

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(stla.index, stla.Q1.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q1", y="Frequency", hue="Source_for_learning", data=stla)

ax.legend(fontsize=20)

ax.set_title('Distribution of preferred source reporting data science by age group',fontsize=20)

ax.set(xlabel='Age Group', ylabel='Distribution of age groups of people participating in the survey')





##############Older people tend to stick to paper version of information... might explain why there is a higher proportion of 70plus people dependng on journal publications

############18-29 reddit increases and then steady decline; same for kaggle

############
###age distribution by country ...

##########more visual analysis required

l = req.Q3.unique()

sources = ['Q12_Part_2','Q12_Part_3','Q12_Part_4','Q12_Part_5','Q12_Part_6','Q12_Part_7','Q12_Part_8','Q12_Part_9','Q12_Part_10','Q12_Part_11','Q12_Part_12']

l = list()

for i in sources:

    df = req.groupby([i,'Q1','Q3']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Source_for_learning'}, inplace = True)

    l.append(df)

countries = req.Q3.unique()

sources_to_learn_by_age = pd.concat(l)

sources_to_learn_by_age = sources_to_learn_by_age[sources_to_learn_by_age.Source_for_learning!=0]

stla = sources_to_learn_by_age.reset_index(drop=True)

for i in countries:

    d = stla[stla['Q3']==i]

    dr = d.reset_index(drop=True)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(90,20))

#     tips = sns.load_dataset("tips")

    ax = sns.barplot(x="Q1", y="Frequency", hue="Source_for_learning", data=dr)

    ax.legend(fontsize=20)

    ax.set_title('Distribution of preferred source reporting data science by age group in country {}'.format(i),fontsize=20)

    ax.set(xlabel='Age Group', ylabel='Distribution of age groups of people participating in the survey')

    ax.legend(loc='best')
########Just a fun dive at q9 <----> category -> "None of these activites are an important part of my role at work"

###########See the popuplations where the curiosity to learn the language exists <----> age group grouping, country grouping

#Q9 ---> Has 8 parts

l = req.Q3.unique()

role_at_work = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']

l = list()

for i in role_at_work:

    df = req.groupby([i,'Q1']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Role_at_Work'}, inplace = True)

    l.append(df)
role_at_work = pd.concat(l)
role_at_work
role_at_work = role_at_work[role_at_work.Role_at_Work != 0]
####

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(role_at_work.index, role_at_work.Q1.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q1", y="Frequency", hue="Role_at_Work", data=role_at_work)

ax.legend(fontsize=20)

ax.set_title('Distribution of Role at work by age group',fontsize=20)

ax.set(xlabel='Age Group', ylabel='Distribution of role at work of people participating in the survey')





##############Older people tend to stick to paper version of information... might explain why there is a higher proportion of 70plus people dependng on journal publications

############18-29 reddit increases and then steady decline; same for kaggle

############
###NOT MUCH INSIGHT BY GENDER DISTRIBUTION



role_at_work = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']

l = list()

for i in role_at_work:

    df = req.groupby([i,'Q2']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Role_at_Work'}, inplace = True)

    l.append(df)
role_at_work_gender = pd.concat(l)
role_at_work_gender = role_at_work_gender[role_at_work_gender.Role_at_Work != 0]

role_at_work_gender
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(role_at_work_gender.index, role_at_work_gender.Q2.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q2", y="Frequency", hue="Role_at_Work", data=role_at_work_gender)

ax.legend(fontsize=20)

ax.set_title('Distribution of Role at work by gender',fontsize=20)

ax.set(xlabel='Gender', ylabel='Distribution of gender of people participating in the survey')
role_at_work = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']

l = list()

for i in role_at_work:

    df = req.groupby([i,'Q6']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Role_at_Work'}, inplace = True)

    l.append(df)
role_at_work_size = pd.concat(l)
role_at_work_size = role_at_work_size[role_at_work_size.Role_at_Work != 0]
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(role_at_work_size.index, role_at_work_size.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", hue="Role_at_Work", data=role_at_work_size)

ax.legend(fontsize=20)

ax.set_title('Distribution of Role at work by size of company',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Distribution of role at work of participants participating in the survey')









#is ML finding more tendency for application in smaller scaled companies ?????
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(role_at_work_size.index, role_at_work_size.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", hue="Role_at_Work", data=role_at_work_size)

ax.legend(fontsize=20)

ax.set_title('Distribution of Role at work by size of company',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Distribution of role at work of participants participating in the survey')









#is ML finding more tendency for application in smaller scaled companies ?????
##########Distribution of people involved in 

role_at_work = ['Q9_Part_1','Q9_Part_2','Q9_Part_3','Q9_Part_4','Q9_Part_5','Q9_Part_6','Q9_Part_7','Q9_Part_8']

l = list()

for i in role_at_work:

    df = req.groupby([i,'Q6','Q3']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

    df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

    df.rename(columns = {i:'Role_at_Work'}, inplace = True)

    l.append(df)

countries = req.Q3.unique()

role_at_work_size = pd.concat(l)

role_at_work_size = role_at_work_size[role_at_work_size.Role_at_Work != 0]

stla = role_at_work_size.reset_index(drop=True)

countries = req.Q3.unique()

for i in countries:

    d = stla[stla['Q3']==i]

    dr = d.reset_index(drop=True)

    #plt.xticks(role_at_work_size.index, role_at_work_size.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(90,20))

#     tips = sns.load_dataset("tips")

    ax = sns.barplot(x="Q6", y="Frequency", hue="Role_at_Work", data=dr)

    ax.legend(fontsize=20)

    ax.set_title('Distribution of Role at work by size of company in {}'.format(i),fontsize=20)

    ax.set(xlabel='Size of company', ylabel='Distribution of role at work of people participating in the survey')
########Amount spent by company v/s size of company employed 

df = req.groupby(['Q6','Q11']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)
d = df[df.Q6 != 0]
dr = d[d.Q11 != 0]
y = dr
y
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", hue="Q11", data=dr)

ax.legend(fontsize=20)

ax.set_title('Distribution of Money invested in ML and Cloud Products by size of company',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Distribution of amount invested by companies of participants participating in the survey')









#is ML finding more tendency for application in smaller scaled companies ?????
###########Amount invested in companies in the field of machine learning and cloud computing products by country

df = req.groupby(['Q6','Q11','Q3']).count().loc[:,:'Time from Start to Finish (seconds)'].reset_index()

df.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True)

d = df[df.Q6 != 0]

dr = d[d.Q11 != 0]

countries = req.Q3.unique()

for i in countries:

    d = dr[dr['Q3']==i]

    drr = d.reset_index(drop=True)

    #plt.xticks(role_at_work_size.index, role_at_work_size.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=(20,20))



    sns.set(style="whitegrid")

#     tips = sns.load_dataset("tips")

    ax = sns.barplot(x="Q6", y="Frequency", hue="Q11", data=drr)

    ax.legend(fontsize=20)

    ax.set_title('Distribution of Money invested in ML and Cloud Products by size of company in {}'.format(i),fontsize=20)

    ax.set(xlabel='Size of company', ylabel='Distribution of amount invested by companies of participants participating in the survey')
#Analysis within each gender

##########GENDER DISTRIBUTION ATTEMPTING SURVEY

gender = req.groupby(['Q2'],as_index=False).count().reset_index().loc[:,'Q2':'Time from Start to Finish (seconds)']
gender.sort_values('Time from Start to Finish (seconds)', inplace=True)
import seaborn as sns

sns.set(style="whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(15,15))

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q2", y="Time from Start to Finish (seconds)", data=gender)

ax.set(xlabel='Gender Participating In The Survey', ylabel='Number of people participating in the survey')
#####DISTRIBUTION BY COUNTRY 

country = req.groupby(['Q3'],as_index=False).count().reset_index().loc[:,'Q3':'Time from Start to Finish (seconds)']
country.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
# Prepare Data

country.sort_values('Frequency', inplace=True)

#country.reset_index(inplace=True)



# Draw plot

import matplotlib.patches as patches



fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)

ax.vlines(x=country.index, ymin=0, ymax=country.Frequency, color='firebrick', alpha=0.7, linewidth=20)



# Annotate Text

for i, Frequency in enumerate(country.Frequency):

    ax.text(i, Frequency+0.5, round(Frequency, 1), horizontalalignment='center')





# Title, Label, Ticks and Ylim

ax.set_title('Bar Chart for Proportion of Survey Attempts by Country distribution', fontdict={'size':22})

ax.set(ylabel='Number of people participated in survey', ylim=(0, 5000))

ax.set(xlabel='Country participating in Survey')

plt.xticks(country.index, country.Q3.str.upper(), rotation=60, horizontalalignment='right', fontsize=12)



# Add patches to color the X axis labels

p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)

p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)

fig.add_artist(p1)

fig.add_artist(p2)

plt.show()
#####DISTRIBUTION BY EDUCATION -----> missing values present

education = req.groupby(['Q4'],as_index=False).count().reset_index().loc[:,'Q4':'Time from Start to Finish (seconds)']
education.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
education.sort_values('Frequency', inplace=True)
education = education[education['Q4'] != 0]
education = education[education['Q4'] != "I prefer not to answer"]
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(education.index, education.Q4.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q4", y="Frequency", data=education)

ax.set_title('Proportion of different levels of education attained by the participants',fontsize=20)

ax.set(xlabel='Highest Level of Education Attained By Person Participating In The Survey', ylabel='Number of people participating in the survey')
#####DISTRIBUTION BY profession of participant

profession = req.groupby(['Q5'],as_index=False).count().reset_index().loc[:,'Q5':'Time from Start to Finish (seconds)']
profession.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
profession.sort_values('Frequency', inplace=True)
profession.reset_index(drop=True)
profession = profession[profession['Q5'] != 0]
####SIGNIFICANT NUMBER OF OTHER RESPONSE ---> NEED MORE ANALYSIS INTO OTHER RESPONSE OF Q5

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(profession.index, profession.Q5.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q5", y="Frequency", data=profession)

ax.set_title('Proportion of different levels of education attained by the participants',fontsize=20)

ax.set(xlabel='Highest Level of Education Attained By Person Participating In The Survey', ylabel='Number of people participating in the survey')
#####DISTRIBUTION BY size of establishment

size = req.groupby(['Q6'],as_index=False).count().reset_index().loc[:,'Q6':'Time from Start to Finish (seconds)']
size.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
size.sort_values('Frequency', inplace=True)
size.reset_index(drop=True)
size = size[size['Q6'] != 0]
####INTERESTING OBSERVATION: survey undertaken by a large population working in a smaller scale industry

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(size.index, size.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", data=size)

ax.set_title('Proportion of sizes of companies the participants are employed in',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Number of people participating in the survey')
####taking a deeper dive into the small scale establishment employees

#size_by_profession = 

size_by_profession = req.groupby(['Q6','Q5'],as_index=False).count().reset_index().loc[:,'Q6':'Time from Start to Finish (seconds)']
size_by_profession = size_by_profession[size_by_profession['Q6'] != 0]
size_by_profession = size_by_profession[size_by_profession['Q5'] != 0]
####

s1 = size_by_profession[size_by_profession['Q6'] == '0-49 employees']

s2 = size_by_profession[size_by_profession['Q6'] == '1000-9,999 employees']

s3 = size_by_profession[size_by_profession['Q6'] == '250-999 employees']

s4 = size_by_profession[size_by_profession['Q6'] == '50-249 employees']

s5 = size_by_profession[size_by_profession['Q6'] == '> 10,000 employees']

# s1.sort_values('Frequency', inplace=True)

# s2.sort_values('Frequency', inplace=True)

# s3.sort_values('Frequency', inplace=True)

# s4.sort_values('Frequency', inplace=True)

# s5.sort_values('Frequency', inplace=True)
size_by_profession = pd.concat([s1,s2,s3,s4,s5])
size_by_profession.reset_index(drop=True)
size_by_profession.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
####

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(size_by_profession.index, size_by_profession.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", hue="Q5", data=size_by_profession)

ax.legend(fontsize=20)

ax.set_title('Distribution of Profession of the participant taking the survey by company size',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Number of people participating in the survey')
####

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(size_by_profession.index, size_by_profession.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", hue="Q5", data=size_by_profession)

ax.legend(fontsize=20)

ax.set_title('Distribution of Profession of the participant taking the survey by company size',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Number of people participating in the survey')
req['Q10']
num_inv_in_datascience = req.groupby(['Q7'],as_index=False).count().reset_index().loc[:,'Q7':'Time from Start to Finish (seconds)']
def func(line):

    l = line

    if line == '14-Oct':

        l = '10-14'

    elif line == '2-Jan':

        l = '1-2'

    elif line == '4-Mar':

        l = '3-4'

    elif line == '9-May':

        l = '5-9'

    return l

num_inv_in_datascience['Q7']=num_inv_in_datascience['Q7'].apply(func)
num_inv_in_datascience.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
num_inv_in_datascience.sort_values('Frequency',inplace = True)
#############LARGE NUMBER OF MISSING VALUES...

num_inv_in_datascience = num_inv_in_datascience[num_inv_in_datascience['Q7']!=0]
num_inv_in_datascience
####INTERESTING OBSERVATION: survey undertaken by a large population working in a smaller scale industry

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(num_inv_in_datascience.index, num_inv_in_datascience.Q7.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q7", y="Frequency", data=num_inv_in_datascience)

ax.set_title('Proportion of number of individuals involved in Datascience in the company the participant is employed in',fontsize=20)

ax.set(xlabel='Number of people involved in data-science ', ylabel='Number of people participating in the survey')
#GROUP BY SIZE OF COMPANY AND NUMBER OF DATA SCIENCE PEOPLE INVOLVED ...ADD A COUNTRY COMPARISON TOO

req['Q7']
#size_by_profession 

size_by_ds = req.groupby(['Q6','Q7'],as_index=False).count().reset_index().loc[:,'Q6':'Time from Start to Finish (seconds)']
size_by_ds.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
size_by_ds = size_by_ds[size_by_ds.Q6 != 0]

size_by_ds = size_by_ds[size_by_ds.Q7 != 0]

size_by_ds['Q7'] = size_by_ds['Q7'].apply(func)
size_by_ds['Q7'].value_counts()

####

s1 = size_by_ds[size_by_ds['Q6'] == '0-49 employees']

s2 = size_by_ds[size_by_ds['Q6'] == '50-249 employees']

s3 = size_by_ds[size_by_ds['Q6'] == '250-999 employees']

s4 = size_by_ds[size_by_ds['Q6'] == '1000-9,999 employees']

s5 = size_by_ds[size_by_ds['Q6'] == '> 10,000 employees']

s1
# s1.sort_values('Frequency', inplace=True)

# s2.sort_values('Frequency', inplace=True)

# s3.sort_values('Frequency', inplace=True)

# s4.sort_values('Frequency', inplace=True)

# s5.sort_values('Frequency', inplace=True)

# s6.sort_values('Frequency', inplace=True)

# s7.sort_values('Frequency', inplace=True)
size_by_ds = pd.concat([s1,s2,s3,s4,s5])
size_by_ds
####

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(30,30))



plt.xticks(size_by_profession.index, size_by_ds.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q6", y="Frequency", hue="Q7", data=size_by_ds)

ax.legend(fontsize=20)

ax.set_title('Distribution of number of data science individuals by company size',fontsize=20)

ax.set(xlabel='Size of company', ylabel='Number of data science individuals in the company')

chartBox = ax.get_position()

ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])

ax.legend(loc='upper center', bbox_to_anchor=(0.5,1), shadow=True, ncol=1)

plt.show()
#country,company size and distribution of individual involved

#Let's consider the top 5 countries from where maximum survey participants



size_by_ds_country = req.groupby(['Q3','Q6','Q7'],as_index=False).count().reset_index().loc[:,'Q3':'Time from Start to Finish (seconds)']
size_by_ds_country.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 

size_by_ds_country
size_by_ds_country = size_by_ds_country[size_by_ds_country.Q6 != 0]

size_by_ds_country = size_by_ds_country[size_by_ds_country.Q7 != 0]

size_by_ds_country['Q7'] = size_by_ds_country['Q7'].apply(func)
size_by_ds_country.reset_index(drop=True)
#Taking mean of each category to get effective number of data science individuals

def func2(line):

    if line == '0':

        line = float(0)

    elif line == '1-2':

        line = float(1.5)

    elif line == '3-4':

        line = float(3.5)

    else:

        line = 12

    return line



size_by_ds_country['Q7'] = size_by_ds_country['Q7'].apply(func2)
size_by_ds_country['Q7']
size_by_ds_country['Effective_Frequency'] = size_by_ds_country.Q7*size_by_ds_country.Frequency
sdc = size_by_ds_country.groupby(['Q3','Q6'],as_index=False)[['Effective_Frequency']].sum()
sdc['Q3'].value_counts()
##TOP 5 COUNTRIES ----> INDIA, UNITED STATES OF AMERICA, OTHER, BRAZIL, JAPAN, RUSSIA, CHINA, GERMANY

##India 

india = sdc[(sdc.Q3 == 'India') | (sdc.Q3 == 'United States of America')| (sdc.Q3 == 'China')| (sdc.Q3 == 'Brazil')| (sdc.Q3 == 'Japan')| (sdc.Q3 == 'Russia')|(sdc.Q3 == 'Germany')]
india
####

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(20,10))



plt.xticks(india.index, india.Q6.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

#tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q3", y="Effective_Frequency", hue="Q6", data=india)

ax.legend(fontsize=20)

ax.set_title('Distribution of number of data science individuals by company size in India',fontsize=20)

ax.set(xlabel='COUNTRY', ylabel='Number of DATA SCIENCE individuals in the company')

chartBox = ax.get_position()

ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])

ax.legend(loc='upper center', bbox_to_anchor=(0.5,1), shadow=True, ncol=1)

plt.show()
####DISTRIBUTION OF SURVEY TAKEN BY PROFESSION OF THE INDIVIDUALS AND COUNTRY

#########DISTRIBUTION OF SURVEY TAKEN BY AGE GROUP, GENDER, COUNTRY, PROFESSION(INTERMIX THESE 4 AND TRY TO FIND TRENDS)

gender_by_country = req.groupby(['Q2','Q3'],as_index=False).count().reset_index().loc[:,'Q2':'Time from Start to Finish (seconds)']
gender_by_country.rename(columns = {'Time from Start to Finish (seconds)':'Frequency'}, inplace = True) 
gender_by_country = gender_by_country[(gender_by_country.Q2 == 'Female') | (gender_by_country.Q2 == 'Male')]
####from pre-liminary visual analysis of the plot, higher proportion of female participants in Brazil, Canada, China, Germany

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(40,60))



plt.xticks(gender_by_country.index, gender_by_country.Q3.str.upper(), rotation=60, horizontalalignment='right', fontsize=20)

sns.set(style="whitegrid")

# tips = sns.load_dataset("tips")

ax = sns.barplot(x="Q3", y="Frequency", hue="Q2", data=gender_by_country)

ax.legend(fontsize=20)

ax.set_title('Distribution of gender of participant by country',fontsize=20)

ax.set(xlabel='COUNTRY', ylabel='Number of DATA SCIENCE individuals in the company')

chartBox = ax.get_position()

ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])

ax.legend(loc='upper center', bbox_to_anchor=(0.5,1), shadow=True, ncol=1)

plt.show()
#Calculate percentage of male and female participants in survey from each country...

gender_by_country
##groupby gender and profession....profession and country.... gender, profession and country

gender_by_profession = req.groupby(['Q2','Q3'],as_index=False).count().reset_index().loc[:,'Q2':'Time from Start to Finish (seconds)']
gender_by_profession
question.head()
question.transpose().head()
q = question.transpose()

q.columns =['Content'] 

q.head()
import nltk

stopwords = nltk.corpus.stopwords.words('english')

new_words=('selected','select','basic','follow','current','highest','formal','choice','follow', 'use', 'regular')

for i in new_words:

    stopwords.append(i)

print(stopwords)

a = list(stopwords)
def create_Word_Corpus(df):

    words_corpus = ''

    for val in df["Content"]:

        text = val.lower()

        tokens = nltk.word_tokenize(text)

        tokens = [word for word in tokens if word not in string.punctuation]

        tokens = [word for word in tokens if word not in a]

        tokens = stemming(tokens)

        for words in tokens:

            words_corpus = words_corpus + words + ' '

    return words_corpus
def plot_Cloud(wordCloud):

    plt.figure( figsize=(20,10), facecolor='k')

    plt.imshow(wordCloud)

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.show()

#     plt.savefig('wordclouds.png', facecolor='k', bbox_inches='tight')
def stemming(tokens):

    ps=PorterStemmer()

    stem_words=[]

    for x in tokens:

        stem_words.append(ps.stem(x))

    return stem_words
# importing all the required Libraries

import glob

import json

import csv

import pandas as pd

import numpy as np

from wordcloud import WordCloud

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer

import string

import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer

import warnings

warnings.filterwarnings("ignore")

question_wordcloud = WordCloud(width=900, height=500).generate(create_Word_Corpus(q))
plot_Cloud(question_wordcloud)