# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import os
import plotly.offline as pyo

import plotly.graph_objs as go

import plotly.offline as py

from plotly import tools

import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns 

from plotly.offline import iplot

import warnings

warnings.filterwarnings("ignore")

import cufflinks as cf

cf.go_offline()
data = pd.read_csv("../input/stack-overflow-developer-survey-2020/developer_survey_2020/survey_results_public.csv")

sc_data = pd.read_csv("../input/stack-overflow-developer-survey-2020/developer_survey_2020/survey_results_schema.csv")
data.head(9)
sc_data.head(9)
data.shape
data.describe()
data.info()
total_data = data.isnull().sum().sort_values(ascending = False)

percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

missing_data  = pd.concat([total_data, percent], axis=1,keys = ['Total','Percent'])

missing_data
missing_data= data.isnull().sum()

missing_data[0:10]

total_data = np.product(data.shape)

total_data_missing = missing_data.sum()

percent_of_missing_data = (total_data_missing/total_data)*100

print(percent_of_missing_data)
temp = data['MainBranch'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Main Branch of the respondent(%)', hole = 0.5)
temp = data['Hobbyist'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Respondents hobby to code(%)', hole = 0.5)
temp = data["Age"].dropna().value_counts().head(50)

temp.iplot(kind='bar', xTitle = 'Age', yTitle = "Count", title = 'Age of respondents')
temp = data["Age1stCode"].dropna().value_counts().head(30)

temp.iplot(kind='bar', xTitle = 'Age', yTitle = "Count", title = 'Age; when respondents coded for 1st time')
temp = data["Country"].dropna().value_counts().head(30)

temp.iplot(kind='bar', xTitle = 'Country name', yTitle = "Count", title = 'Countries by respondents')
temp = data['Gender'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Developers Gender(%)', hole = 0.5)
temp = data['Employment'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Developers Employment(%)', hole = 0.5)
temp = data['OrgSize'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Size of Organization; Developers working (%)', hole = 0.5)
temp = data['UndergradMajor'].value_counts().head(7)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Undergraduatee major Developers (%)', hole = 0.5)
temp = data["JobSat"].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Job Satisfaction of workers(%)', hole = 0.5)
temp = data["YearsCodePro"].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='years of coding proficiency (%)', hole = 0.5)

temp = data["MainBranch"].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='MainBranch of the Developer (%)', hole = 0.5)
temp = data["JobFactors"].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Job Factors of respondents(%)', hole = 0.5)
temp = data["EdLevel"].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Job Factors of respondents(%)', hole = 0.5)
temp = data["DevType"].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Developer Type(%)', hole = 0.5)
temp = data["Ethnicity"].value_counts().head(10)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Developer Type(%)', hole = 0.5)
temp = data["JobSeek"].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Job Search status(%)', hole = 0.5)
temp = data["LanguageDesireNextYear"].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Language desired by developer for coming years(%)', hole = 0.5)
temp = data["LanguageWorkedWith"].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Language Worked with(%)', hole = 0.5)
temp = data["NEWDevOps"].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Does developers company have a dedicated DevOps person? (%)', hole = 0.5)
temp = pd.DataFrame(data['Sexuality'].dropna().str.split(';').tolist()).stack()

temp =  temp.value_counts().sort_values(ascending=False)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Sexual Orientation(%)', hole = 0.5)
temp = pd.DataFrame(data['NEWLearn'].dropna().str.split(';').tolist()).stack()

temp =  temp.value_counts().sort_values(ascending=False)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='How frequeltly is a respondent learning new language? (%)', hole = 0.5)
temp = data["WorkWeekHrs"].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', 

         title='Respondents work hours in a week(%)', hole = 0.5)
temp = data['WebframeWorkedWith'].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='WebFrame Respondents Working With(%)', hole = 0.5)
temp = data['WebframeDesireNextYear'].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='WebFrame Desired Next Year(%)', hole = 0.5)
temp = data['SOVisitFreq'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='How Frequently does Resopondent visit SOF(%)', hole = 0.5)
temp = data['OpSys'].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Operating System Used(%)', hole = 0.5)
temp = data['DatabaseWorkedWith'].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Database Worked With(%)', hole = 0.5)
temp = data['DatabaseDesireNextYear'].value_counts().head(20)

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='labels',values='values', title='Database Desired Next Year(%)', hole = 0.5)