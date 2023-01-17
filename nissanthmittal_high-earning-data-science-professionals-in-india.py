#importing the required packages

import pandas as pd

import matplotlib.pyplot as plt

from pandas.api.types import CategoricalDtype

import seaborn as sns

import numpy as np

from scipy import stats

#import researchpy as rp
#reading the raw data

input_data = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")



#data processing

#removing 1st row and assigning the right column headers

input_data.columns = input_data.iloc[0]

input_data = input_data.drop([0])



#check the shape 

print(input_data.shape)

#input_data.head()
#Renaming columns

processed_data = input_data



processed_data = processed_data.rename(columns={

    'In which country do you currently reside?': 'Country', 

    'What is your gender? - Selected Choice': 'Gender',

    'What is your current yearly compensation (approximate $USD)?': 'Salary',

    'What is your age (# years)?': 'Age',

    'Select the title most similar to your current role (or most recent title if retired): - Selected Choice': 'Job Title',

    'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Education',

    'What is the size of the company where you are employed?': 'Organization Size',

    'What is the size of the company where you are employed?': 'Company Size',

    'Approximately how many individuals are responsible for data science workloads at your place of business?':'Team Size',

    'Does your current employer incorporate machine learning methods into their business?':'Employer incorporate ML',

    'Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?':'Spend on ML/Cloud products',

    'How long have you been writing code to analyze data (at work or at school)?':'Experience in writing codes',

    'What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice':'Recommended first programming lang',

    'For how many years have you used machine learning methods?':'Years of using ML',

    'Have you ever used a TPU (tensor processing unit)?':'Used TPU',

    'What is the primary tool that you use at work or school to analyze data? (Include text response) - Selected Choice':'Primary analysis tool'

}

                                    )
#Filtering out the data for India

country_data = processed_data[processed_data['Country'] == 'India']

country_data.shape
#Filtering out the data for Data Scientist related job titles in India

country_data = country_data[(country_data['Job Title'] == 'Data Analyst') |

                            (country_data['Job Title'] == 'Business Analyst') |

                            (country_data['Job Title'] == 'Data Scientist')]

country_data.shape
#Grouping the salaries

country_data.loc[country_data['Salary'].isin(['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499'

]), 'Salary'] = '0-7,499'



country_data.loc[country_data['Salary'].isin(['7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999'

]), 'Salary'] = '7,500-24,999'



country_data.loc[country_data['Salary'].isin(['25,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999'

]), 'Salary'] = '25,000-69,999'



country_data.loc[country_data['Salary'].isin(['70,000-79,999','80,000-89,999','90,000-99,999','100,000-124,999','125,000-149,999'

]), 'Salary'] = '70,000-149,999'



country_data.loc[country_data['Salary'].isin(['150,000-199,999','200,000-249,999','250,000-299,999','300,000-500,000','> $500,000'

]), 'Salary'] = '>150,000'



#country_data['Salary'].value_counts()



cat_type_1 = CategoricalDtype(categories=['0-7,499', '7,500-24,999', '25,000-69,999','70,000-149,999','>150,000'],

                             ordered=True)

country_data['Salary'] = country_data['Salary'].astype(cat_type_1)
#Grouping the age

country_data.loc[country_data['Age'].isin(['22-24', '25-29']), 'Age'] = '22-29'

country_data.loc[country_data['Age'].isin(['30-34', '35-39']), 'Age'] = '30-39'

country_data.loc[country_data['Age'].isin(['40-44', '45-49']), 'Age'] = '40-49'

country_data.loc[country_data['Age'].isin(['50-54', '55-59', '60-69','70+']), 'Age'] = '>=50'



#country_data['Age'].value_counts()



cat_type_2 = CategoricalDtype(categories=['18-21','22-29', '30-39', '40-49','>=50'],

                             ordered=True)



country_data['Age'] = country_data['Age'].astype(cat_type_2)
#grouping team sizes

country_data.loc[country_data['Team Size'].isin(['3-4', '5-9']), 'Team Size'] = '3-9'

country_data.loc[country_data['Team Size'].isin(['10-14', '15-19']), 'Team Size'] = '10-19'



#country_data['Team Size'].value_counts()



cat_type_3 = CategoricalDtype(categories=['0','1-2','3-9','10-19','20+'],

                             ordered=True)



country_data['Team Size'] = country_data['Team Size'].astype(cat_type_3)
#grouping Experience in writing codes

country_data.loc[country_data['Experience in writing codes'].isin(['I have never written code', '< 1 years']), 

                 'Experience in writing codes']= '<1 years'



cat_type_4 = CategoricalDtype(categories=['< 1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years'],

                             ordered=True)



country_data['Experience in writing codes'] = country_data['Experience in writing codes'].astype(cat_type_4)



#country_data['Experience in writing codes'].value_counts()
#grouping Years of using ML

country_data.loc[country_data['Years of using ML'].isin(['1-2 years', '2-3 years']), 

                 'Years of using ML']= '1-3 years'

country_data.loc[country_data['Years of using ML'].isin(['3-4 years', '4-5 years']), 

                 'Years of using ML']= '3-5 years'



cat_type_5 = CategoricalDtype(categories=['< 1 years','1-3 years','3-5 years','5-10 years','10-15 years','20+ years'],

                             ordered=True)



country_data['Years of using ML'] = country_data['Years of using ML'].astype(cat_type_5)



#country_data['Years of using ML'].value_counts()
#ordering Company Size

cat_type_6 = CategoricalDtype(categories=['0-49 employees','50-249 employees','250-999 employees',

                                          '1000-9,999 employees','> 10,000 employees'],

                             ordered=True)

country_data['Company Size'] = country_data['Company Size'].astype(cat_type_6)



#ordering Team Size

cat_type_7 = CategoricalDtype(categories=['0','1-2','3-9','10-19','20+'],

                             ordered=True)

country_data['Team Size'] = country_data['Team Size'].astype(cat_type_7)



#ordering Spend on ML/Cloud products

cat_type_8 = CategoricalDtype(categories=['$0 (USD)','$1-$99','$100-$999','$1000-$9,999','$10,000-$99,999','> $100,000 ($USD)'],

                             ordered=True)

country_data['Spend on ML/Cloud products'] = country_data['Spend on ML/Cloud products'].astype(cat_type_8)

ct_age = pd.crosstab(country_data['Salary'],country_data['Age']).apply(lambda r: round((r/r.sum())*100), axis=0)

print(ct_age)



ct_age = pd.crosstab(country_data['Salary'],country_data['Age']).apply(lambda r: round((r/r.sum())*100), axis=1)

ct_age.plot.bar(stacked=True)

plt.legend(title='Age',loc='upper center', bbox_to_anchor=(1.1, 1), shadow=True, ncol=1)

plt.show()
pd.crosstab(country_data['Salary'],country_data['Experience in writing codes']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct_exp_code = pd.crosstab(country_data['Salary'],country_data['Experience in writing codes']).apply(lambda r: round((r/r.sum())*100), axis=1)

ct_exp_code.plot.bar(stacked=True)

plt.legend(title='Experience in writing codes',loc='upper center', bbox_to_anchor=(1.22, 1), shadow=True, ncol=1)

plt.show()
pd.crosstab(country_data['Salary'],country_data['Years of using ML']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct_exp_usingML = pd.crosstab(country_data['Salary'],country_data['Years of using ML']).apply(lambda r: round((r/r.sum())*100), axis=1)

ct_exp_usingML.plot.bar(stacked=True)

plt.legend(title='Years of using ML',loc='upper center', bbox_to_anchor=(1.2, 1), shadow=True, ncol=1)

plt.show()
(pd.crosstab(country_data['Age'],country_data['Experience in writing codes']).apply(lambda r: round((r/r.sum())*100), axis=1))
(pd.crosstab(country_data['Age'],country_data['Years of using ML']).apply(lambda r: round((r/r.sum())*100), axis=1))
pd.crosstab(country_data['Salary'],country_data['Education']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct_Education = pd.crosstab(country_data['Salary'],country_data['Education']).apply(lambda r: round((r/r.sum())*100), axis=1)

ct_Education.plot.bar(stacked=True)

plt.legend(title='Education',loc='upper center', bbox_to_anchor=(1.6, 1), shadow=True, ncol=1)

plt.show()



print("Chi-Square P-value",stats.chi2_contingency(pd.crosstab(country_data['Salary'],country_data['Education']))[1])



#table, results = rp.crosstab(country_data['Salary'],country_data['Education'], prop= 'col', test= 'chi-square')

#print(results)
pd.crosstab(country_data['Salary'],country_data['Company Size']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct_CompanySize = pd.crosstab(country_data['Salary'],country_data['Company Size']).apply(lambda r: round((r/r.sum())*100), axis=1)

ct_CompanySize.plot.bar(stacked=True)

plt.legend(title='Company Size',loc='upper center', bbox_to_anchor=(1.25, 1), shadow=True, ncol=1)

plt.show()
pd.crosstab(country_data['Salary'],country_data['Team Size']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct1 = pd.crosstab(country_data['Company Size'],country_data['Team Size']).apply(lambda r: round((r/r.sum())*100), axis=1)



ct1.plot.bar(stacked=True)

plt.legend(title='Team Size',loc='upper center', bbox_to_anchor=(1.1, 1), shadow=True, ncol=1)

plt.show()
ct_EmpML = pd.crosstab(country_data['Salary'],country_data['Employer incorporate ML']).apply(lambda r: round((r/r.sum())*100), axis=1)



ct_EmpML.plot.bar(stacked=True)

plt.legend(title='Employer incorporate ML',loc='upper center', bbox_to_anchor=(1.8, 1), shadow=True, ncol=1)

plt.show()
pd.crosstab(country_data['Company Size'],country_data['Employer incorporate ML']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct_spend = pd.crosstab(country_data['Salary'],country_data['Spend on ML/Cloud products']).apply(lambda r: round((r/r.sum())*100), axis=1)



ct_spend.plot.bar(stacked=True)

plt.legend(title='Spend on ML/Cloud products',loc='upper center', bbox_to_anchor=(1.25, 1), shadow=True, ncol=1)

plt.show()
pd.crosstab(country_data['Company Size'],country_data['Spend on ML/Cloud products']).apply(lambda r: round((r/r.sum())*100), axis=0)
ct_UsedTPU = pd.crosstab(country_data['Salary'],country_data['Used TPU']).apply(lambda r: round((r/r.sum())*100), axis=1)



ct_UsedTPU.plot.bar(stacked=True)

plt.legend(title='Used TPU',loc='upper center', bbox_to_anchor=(1.15, 1), shadow=True, ncol=1)

plt.show()
ct_tool = pd.crosstab(country_data['Salary'],country_data['Primary analysis tool']).apply(lambda r: round((r/r.sum())*100), axis=1)



ct_tool.plot.bar(stacked=True)

plt.legend(title='Primary analysis tool',loc='upper center', bbox_to_anchor=(1.55, 1), shadow=True, ncol=1)

plt.show()
pd.crosstab(country_data['Salary'],country_data['Gender']).apply(lambda r: round((r/r.sum())*100), axis=0)
#table, results = rp.crosstab(country_data['Salary'],country_data['Gender'], prop= 'col', test= 'chi-square')

#print(results)

print("Chi-Square P-value",stats.chi2_contingency(pd.crosstab(country_data['Salary'],

                                                              country_data['Gender']))[1])
#table, results = rp.crosstab(country_data['Experience in writing codes'],country_data['Gender'], prop= 'col', test= 'chi-square')

#print(results)

print("Chi-Square P-value",stats.chi2_contingency(pd.crosstab(country_data['Experience in writing codes'],

                                                              country_data['Gender']))[1])
#table, results = rp.crosstab(country_data['Years of using ML'],country_data['Gender'], prop= 'col', test= 'chi-square')

#print(results)

print("Chi-Square P-value",stats.chi2_contingency(pd.crosstab(country_data['Years of using ML'],

                                                              country_data['Gender']))[1])
#table, results = rp.crosstab(country_data['Age'],country_data['Gender'], prop= 'col', test= 'chi-square')

#print(results)

print("Chi-Square P-value",stats.chi2_contingency(pd.crosstab(country_data['Age'],

                                                              country_data['Gender']))[1])
#table, results = rp.crosstab(country_data['Company Size'],country_data['Gender'], prop= 'col', test= 'chi-square')

#print(results)

print("Chi-Square P-value",stats.chi2_contingency(pd.crosstab(country_data['Company Size'],

                                                              country_data['Gender']))[1])
ct_reclang = pd.crosstab(country_data['Salary'],country_data['Recommended first programming lang']).apply(lambda r: round((r/r.sum())*100), axis=1)

ct_reclang.plot.bar(stacked=True)

plt.legend(title='Recommended first programming lang',loc='upper center', bbox_to_anchor=(1.35, 1), shadow=True, ncol=1)

plt.show()
#Filtering out the data for Data Scientist related job titles in India

country_data = country_data[(country_data['Salary'] == '70,000-149,999') |

                            (country_data['Salary'] == '>150,000')]

country_data.shape
Q11=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions'].value_counts().to_frame().reset_index()

Q12=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data'].value_counts().to_frame().reset_index()

Q13=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build prototypes to explore applying machine learning to new areas'].value_counts().to_frame().reset_index()

Q14=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Build and/or run a machine learning service that operationally improves my product or workflows'].value_counts().to_frame().reset_index()

Q15=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Experimentation and iteration to improve existing ML models'].value_counts().to_frame().reset_index()

Q16=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Do research that advances the state of the art of machine learning'].value_counts().to_frame().reset_index()

Q17=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - None of these activities are an important part of my role at work'].value_counts().to_frame().reset_index()

Q18=country_data['Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Other'].value_counts().to_frame().reset_index()



important_activities_concat = [Q11,Q12,Q13,Q14,Q15,Q16,Q17,Q18]



important_activities = pd.concat(important_activities_concat, sort=False)



#Rename index

important_activities = important_activities.rename({'index':'labels'}, axis='columns')



#Plot the graph

important_activities.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(important_activities.labels,loc='center left', bbox_to_anchor=(1, 0.6), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q21=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Twitter (data science influencers)'].value_counts().to_frame().reset_index()

Q22=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Hacker News (https://news.ycombinator.com/)'].value_counts().to_frame().reset_index()

Q23=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)'].value_counts().to_frame().reset_index()

Q24=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)'].value_counts().to_frame().reset_index()

Q25=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)'].value_counts().to_frame().reset_index()

Q26=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)'].value_counts().to_frame().reset_index()

Q27=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)'].value_counts().to_frame().reset_index()

Q28=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)'].value_counts().to_frame().reset_index()

Q29=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)'].value_counts().to_frame().reset_index()

Q30=country_data['Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Slack Communities (ods.ai, kagglenoobs, etc)'].value_counts().to_frame().reset_index()



fav_media_concat = [Q21,Q22,Q23,Q24,Q25,Q26,Q27,Q28,Q29,Q30]



fav_media = pd.concat(fav_media_concat, sort=False)



#Rename index

fav_media = fav_media.rename({'index':'labels'}, axis='columns')



#Plot the graph

fav_media.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(fav_media.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q31=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udacity'].value_counts().to_frame().reset_index()

Q32=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Coursera'].value_counts().to_frame().reset_index()

Q33=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - edX'].value_counts().to_frame().reset_index()

Q34=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataCamp'].value_counts().to_frame().reset_index()

Q35=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - DataQuest'].value_counts().to_frame().reset_index()

Q36=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Kaggle Courses (i.e. Kaggle Learn)'].value_counts().to_frame().reset_index()

Q37=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Fast.ai'].value_counts().to_frame().reset_index()

Q38=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - Udemy'].value_counts().to_frame().reset_index()

Q39=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - LinkedIn Learning'].value_counts().to_frame().reset_index()

Q40=country_data['On which platforms have you begun or completed data science courses? (Select all that apply) - Selected Choice - University Courses (resulting in a university degree)'].value_counts().to_frame().reset_index()



courses_concat = [Q31,Q32,Q33,Q34,Q35,Q36,Q37,Q38,Q39,Q40]



courses = pd.concat(courses_concat, sort=False)



#Rename index

courses = courses.rename({'index':'labels'}, axis='columns')



#Plot the graph

courses.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(courses.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q71=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Ggplot / ggplot2 '].value_counts().to_frame().reset_index()

Q72=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Matplotlib '].value_counts().to_frame().reset_index()

Q73=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Altair '].value_counts().to_frame().reset_index()

Q74=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Shiny '].value_counts().to_frame().reset_index()

Q75=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3.js '].value_counts().to_frame().reset_index()

Q76=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Plotly / Plotly Express '].value_counts().to_frame().reset_index()

Q77=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Bokeh '].value_counts().to_frame().reset_index()

Q78=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Seaborn '].value_counts().to_frame().reset_index()

Q79=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Geoplotlib '].value_counts().to_frame().reset_index()

Q80=country_data['What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  Leaflet / Folium '].value_counts().to_frame().reset_index()



visual_concat = [Q71,Q72,Q73,Q74,Q75,Q76,Q77,Q78,Q79,Q80]



visual = pd.concat(visual_concat, sort=False)



#Rename index

visual = visual.rename({'index':'labels'}, axis='columns')



#Plot the graph

visual.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(visual.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q61=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Linear or Logistic Regression'].value_counts().to_frame().reset_index()

Q62=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Decision Trees or Random Forests'].value_counts().to_frame().reset_index()

Q63=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Gradient Boosting Machines (xgboost, lightgbm, etc)'].value_counts().to_frame().reset_index()

Q64=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Bayesian Approaches'].value_counts().to_frame().reset_index()

Q65=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Evolutionary Approaches'].value_counts().to_frame().reset_index()

Q66=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Dense Neural Networks (MLPs, etc)'].value_counts().to_frame().reset_index()

Q67=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Convolutional Neural Networks'].value_counts().to_frame().reset_index()

Q68=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Generative Adversarial Networks'].value_counts().to_frame().reset_index()

Q69=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Recurrent Neural Networks'].value_counts().to_frame().reset_index()

Q70=country_data['Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-2, etc)'].value_counts().to_frame().reset_index()



MLalgo_concat = [Q61,Q62,Q63,Q64,Q65,Q66,Q67,Q68,Q69,Q70]



MLalgo = pd.concat(MLalgo_concat, sort=False)



#Rename index

MLalgo = MLalgo.rename({'index':'labels'}, axis='columns')



#Plot the graph

MLalgo.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(MLalgo.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q41=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   Scikit-learn '].value_counts().to_frame().reset_index()

Q42=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -   TensorFlow '].value_counts().to_frame().reset_index()

Q43=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Keras '].value_counts().to_frame().reset_index()

Q44=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  RandomForest'].value_counts().to_frame().reset_index()

Q45=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Xgboost '].value_counts().to_frame().reset_index()

Q46=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  PyTorch '].value_counts().to_frame().reset_index()

Q47=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Caret '].value_counts().to_frame().reset_index()

Q48=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  LightGBM '].value_counts().to_frame().reset_index()

Q49=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Spark MLib '].value_counts().to_frame().reset_index()

Q50=country_data['Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply) - Selected Choice -  Fast.ai '].value_counts().to_frame().reset_index()



MLframework_concat = [Q41,Q42,Q43,Q44,Q45,Q46,Q47,Q48,Q49,Q50]



MLframework = pd.concat(MLframework_concat, sort=False)



#Rename index

MLframework = MLframework.rename({'index':'labels'}, axis='columns')



#Plot the graph

MLframework.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(MLframework.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q81=country_data['Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - CPUs'].value_counts().to_frame().reset_index()

Q82=country_data['Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - GPUs'].value_counts().to_frame().reset_index()

Q83=country_data['Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - TPUs'].value_counts().to_frame().reset_index()



hardware_concat = [Q81,Q82,Q83]



hardware = pd.concat(hardware_concat, sort=False)



#Rename index

hardware = hardware.rename({'index':'labels'}, axis='columns')



#Plot the graph

hardware.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(hardware.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')

Q51=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) '].value_counts().to_frame().reset_index()

Q52=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) '].value_counts().to_frame().reset_index()

Q53=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure '].value_counts().to_frame().reset_index()

Q54=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud '].value_counts().to_frame().reset_index()

Q55=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Alibaba Cloud '].value_counts().to_frame().reset_index()

Q56=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Salesforce Cloud '].value_counts().to_frame().reset_index()

Q57=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Oracle Cloud '].value_counts().to_frame().reset_index()

Q58=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  SAP Cloud '].value_counts().to_frame().reset_index()

Q59=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  VMware Cloud '].value_counts().to_frame().reset_index()

Q60=country_data['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Red Hat Cloud '].value_counts().to_frame().reset_index()



cloud_concat = [Q51,Q52,Q53,Q54,Q55,Q56,Q57,Q58,Q59,Q60]



cloud = pd.concat(cloud_concat, sort=False)



#Rename index

cloud = cloud.rename({'index':'labels'}, axis='columns')



#Plot the graph

cloud.plot.bar(width=10, cmap='Paired')



legend_properties = {'weight':'bold','size': 12}

plt.legend(cloud.labels,loc='center left', bbox_to_anchor=(1, 0.5), prop=legend_properties)

plt.yticks(fontsize=12, weight='bold')

plt.ylabel('Number of respondents',fontsize=13, weight='bold')
