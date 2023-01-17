# import the necessary libraries

import numpy as np 

import pandas as pd



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 16, 10

#plt.rcParams['image.cmap'] = 'viridis'





import os



# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
os.listdir('../input/kaggle-survey-2019/')
# Importing the 2019 Dataset

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df_2019.columns = df_2019.iloc[0]

df_2019=df_2019.drop([0]) # The first row just contains the column names, so we can drop it.
# Create a boolean column if they use Vim/Emacs.

# This is mainly so we don't have to refer to such a long column name every time.

df_2019['vim/emacs_user'] = '  Vim / Emacs  ' == df_2019["Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -   Vim / Emacs  "] 
df_2019['numerical_age'] = 0 # Initialize the numerical_age column

np.random.seed(2019) # Set random seed for reproducability

df_2019.loc[df_2019['What is your age (# years)?']=='18-21', 'numerical_age'] = np.random.uniform(low=18,high=22,size=df_2019.loc[df_2019['What is your age (# years)?']=='18-21'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='22-24', 'numerical_age'] = np.random.uniform(low=22,high=25,size=df_2019.loc[df_2019['What is your age (# years)?']=='22-24'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='25-29', 'numerical_age'] = np.random.uniform(low=25,high=30,size=df_2019.loc[df_2019['What is your age (# years)?']=='25-29'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='30-34', 'numerical_age'] = np.random.uniform(low=30,high=35,size=df_2019.loc[df_2019['What is your age (# years)?']=='30-34'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='35-39', 'numerical_age'] = np.random.uniform(low=35,high=40,size=df_2019.loc[df_2019['What is your age (# years)?']=='35-39'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='40-44', 'numerical_age'] = np.random.uniform(low=40,high=45,size=df_2019.loc[df_2019['What is your age (# years)?']=='40-44'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='45-49', 'numerical_age'] = np.random.uniform(low=45,high=50,size=df_2019.loc[df_2019['What is your age (# years)?']=='45-49'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='50-54', 'numerical_age'] = np.random.uniform(low=50,high=55,size=df_2019.loc[df_2019['What is your age (# years)?']=='50-54'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='55-59', 'numerical_age'] = np.random.uniform(low=55,high=60,size=df_2019.loc[df_2019['What is your age (# years)?']=='55-59'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='60-69', 'numerical_age'] = np.random.uniform(low=60,high=70,size=df_2019.loc[df_2019['What is your age (# years)?']=='60-69'].shape[0])

df_2019.loc[df_2019['What is your age (# years)?']=='70+', 'numerical_age']   = np.random.uniform(low=70,high=90,size=df_2019.loc[df_2019['What is your age (# years)?']=='70+'].shape[0])
plt.rcParams['figure.figsize'] = 16, 10

ax = sns.kdeplot(df_2019.loc[~df_2019['vim/emacs_user'],'numerical_age'], shade=True, color='blue', label='Non-Vim/Emacs')

sns.kdeplot(df_2019.loc[df_2019['vim/emacs_user'], 'numerical_age'], shade=True, color='orange', label='Vim/Emacs', ax=ax)
ax = sns.kdeplot(df_2019.loc[~df_2019['vim/emacs_user'],'numerical_age'], shade=True, color='blue', label='Non-Vim/Emacs')

sns.kdeplot(df_2019.loc[df_2019['vim/emacs_user'], 'numerical_age'], shade=True, color='orange', label='Vim/Emacs', ax=ax)

ax.set_ylim(bottom=0,top=0.003) # Zoom in where the plot is low

ax.set_xlim(left=55,right=95) # Zoom in for the high ages
np.random.seed(2019)

df_2019['cloud_money'] = 0 # Initialize numerical column

df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$1-$99', 'cloud_money'] = np.random.uniform(low=1,high=100,size=df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$1-$99'].shape[0])

df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$100-$999', 'cloud_money'] = np.random.uniform(low=100,high=1000,size=df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$100-$999'].shape[0])

df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$1000-$9,999', 'cloud_money'] = np.random.uniform(low=1000,high=10000,size=df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$1000-$9,999'].shape[0])

df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$10,000-$99,999', 'cloud_money'] = np.random.uniform(low=10000,high=100000,size=df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='$10,000-$99,999'].shape[0])

df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='> $100,000 ($USD)', 'cloud_money'] = np.random.uniform(low=100000,high=200000,size=df_2019.loc[df_2019['Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?']=='> $100,000 ($USD)'].shape[0])
ax = sns.kdeplot(df_2019.loc[~df_2019['vim/emacs_user'],'cloud_money'], shade=True, color='blue', label='Non-Vim/Emacs')

sns.kdeplot(df_2019.loc[df_2019['vim/emacs_user'], 'cloud_money'], shade=True, color='orange', label='Vim/Emacs', ax=ax)
# Some questions are text responses which therefore require a lookup into the 'other_text_responses.csv'

text_responses = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')

text_responses.columns = text_responses.iloc[0]

text_responses=text_responses.drop([0]) # The first row just contains the column names, so we can drop it.
df_2019['What is the primary tool that you use at work or school to analyze data? (Include text response) - Cloud-based data software & APIs (AWS, GCP, Azure, etc.) - Text'] = text_responses['What is the primary tool that you use at work or school to analyze data? (Include text response) - Cloud-based data software & APIs (AWS, GCP, Azure, etc.) - Text'].astype(str).str.upper()
# View the responses sorted by popularity

df_2019.loc[df_2019['vim/emacs_user'], 'What is the primary tool that you use at work or school to analyze data? (Include text response) - Cloud-based data software & APIs (AWS, GCP, Azure, etc.) - Text'].value_counts()
df_2019['uses_cloud_compute'] = ((df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Platform (GCP) '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Amazon Web Services (AWS) '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Alibaba Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Salesforce Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Oracle Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  SAP Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  VMware Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Red Hat Cloud '].notnull()) |

                                 (df_2019['Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice - Other'].notnull())

                                )   
# The percentage of people who use cloud compute, grouped by if they use Vim/Emacs or not.

df_2019.groupby('vim/emacs_user')['uses_cloud_compute'].mean().to_frame('Probability of Using Cloud Compute').reset_index()
# Impute the Other with the freeform text

df_2019['Select the title most similar to your current role (or most recent title if retired): - Other - Text'] = text_responses['Select the title most similar to your current role (or most recent title if retired): - Other - Text'].astype(str).str.upper()
vc_vim = df_2019.loc[df_2019['vim/emacs_user'], 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts(normalize=True)

vc_notvim = df_2019.loc[~df_2019['vim/emacs_user'], 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts(normalize=True)



w = pd.DataFrame(data = [vc_vim, vc_notvim],index = ['Vim/Emacs','Non-Vim/Emacs'])



ax = w.T[['Non-Vim/Emacs']].plot(subplots=True, layout=(1,1),kind='bar',color='blue',linewidth=1,edgecolor='k',legend=True, label='Non-Vim/Emacs',alpha=0.25)

w.T[['Vim/Emacs']].plot(subplots=True, layout=(1,1),kind='bar',color='orange',linewidth=1,edgecolor='k',legend=True, label='Vim/Emacs',alpha=0.25, ax=ax)



plt.gcf().set_size_inches(10,8)

plt.title('Job Title of Vim/Emacs users vs. non-Vim/Emacs users',fontsize=15)

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Job Title',fontsize=15)

plt.ylabel('Percentage of Users',fontsize=15)

plt.show()