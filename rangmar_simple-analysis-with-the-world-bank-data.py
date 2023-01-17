# Import Library!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import warnings
from matplotlib import style

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
%matplotlib inline
survey_df = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
ques_list = survey_df.iloc[0,:].values.tolist()
survey_df = survey_df.iloc[1:, ]

survey_df['Q3'].value_counts()[:32]
worldbank = pd.read_excel('../input/world-bank-datakaggle-survey/WorldBank_Data.xlsx')
worldbank.head()
plt.figure(figsize=(13,7))
sns.regplot(worldbank['GDP_Per_Capita'], worldbank['Count/POP'])
plt.ylabel('Ans_Count/Total Popluation')
for i, v, s in worldbank[['GDP_Per_Capita', 'Count/POP', 'Country Code']].values :
    plt.text(i,v,s)