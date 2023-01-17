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
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
from plotly.offline import iplot
import warnings
warnings.filterwarnings("ignore")
import cufflinks as cf
cf.go_offline()

data = pd.read_csv("../input/survey_results_public.csv")
schema_data = pd.read_csv("../input/survey_results_schema.csv")
data.head()
schema_data.head()
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
temp = data['Hobby'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='% of Developers hobby', hole = 0.7) 
temp = data['OpenSource'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers open source projects(%)', hole = 0.7)
temp = data["Country"].dropna().value_counts().head(20)
temp.iplot(kind='bar', xTitle = 'Country name', yTitle = "Count", title = 'Countries by respondents')
temp = data['Student'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers who are students(%)', hole = 0.7)
temp = data['Employment'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers Employment(%)', hole = 0.7)
temp = data['FormalEducation'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers Formal Education (%)', hole = 0.7)
temp = data['UndergradMajor'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Undergraduatee major Developers (%)', hole = 0.7)
temp = data['CompanySize'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Company Size(%)', hole = 0.7) 
temp = data["JobSatisfaction"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Job Satisfaction of workers(%)', hole = 0.7)
temp = data["YearsCoding"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='years peoples been coding (%)', hole = 0.7)
temp = data["YearsCodingProf"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='years of coding proficiency (%)', hole = 0.7)
temp = data["CareerSatisfaction"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Career Satisfaction of workers(%)', hole = 0.7)
temp = data["HopeFiveYears"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Peoples hope of doing work in next five years (%)', hole = 0.7)
temp = data["JobSearchStatus"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Job Search status(%)', hole = 0.7)
temp = data["LastNewJob"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Last time when new job was taken(%)', hole = 0.7)
temp = data['UpdateCV'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='upadating CV(%)', hole = 0.7)
temp = data['Currency'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Currency(%)', hole = 0.7)
temp = data['StackOverflowRecommend'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='StackOverflow Recommendation(%)', hole = 0.7)
temp = data['StackOverflowVisit'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Peoples visiting StackOverflow(%)', hole = 0.7)
temp = data['StackOverflowHasAccount'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Has a StackOverflow account(%)',hole = 0.7)
temp = data['StackOverflowParticipate'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Participation on StackOverflow (%)',hole = 0.7)
temp = data['StackOverflowJobs'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='StackOverflow Jobs(%)',hole = 0.7)
temp = data['StackOverflowDevStory'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developer story on StackOverflow(%)',hole = 0.7)
temp = data['StackOverflowConsiderMember'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Members of the Stack Overflow(%)', hole = 0.7)
temp = data['UpdateCV'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Upadating CV(%)', hole = 0.7)
temp = data['EthicsChoice'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ethics Choice)(%)', hole = 0.7)
temp = data['EthicsReport'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ethics Reports(%)', hole = 0.7)
temp = data['EthicsResponsible'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ethics responsibility(%)', hole = 0.7)
temp = data['EthicalImplications'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ethical Implicationns(%)', hole = 0.7)
temp = data['OperatingSystem'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Operating system developers(%)', hole = 0.7)
temp = data['WakeTime'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers wake up time(%)', hole = 0.7) 
temp = data['HoursComputer'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers spending time infront of desktop or computer(%)', hole = 0.7)
temp = data['HoursOutside'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers spending time outside(%)', hole = 0.7)
temp = data['SkipMeals'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers skiping meals(%)', hole = 0.7) 
temp = data['ErgonomicDevices'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ergonomic devices(%)', hole = 0.7) 
temp = data['Exercise'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Developers Exercise(%)', hole = 0.7)
temp = data['Age'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Age of the developers(%)', hole = 0.7) 
temp = data["NumberMonitors"].dropna().value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Number of monitors(%)', hole = 0.7) 
temp = data["CheckInCode"].dropna().value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Checkin(%)', hole = 0.7)
temp = pd.DataFrame(data['Gender'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Gender(%)', hole = 0.7)
temp = pd.DataFrame(data['SexualOrientation'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts().sort_values(ascending=False)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Sexual Orientation(%)', hole = 0.7)
temp = data['AdBlockerDisable'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ad Blocker disable(%)', hole = 0.7)
temp = pd.DataFrame(data['AdBlockerReasons'].dropna().str.split(';').tolist()).stack()
temp =  temp.value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Reasons for disabling AdBlocker', hole = 0.7)
temp = data['AdsAgreeDisagree1'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ads agreed or disagreed(%)', hole = 0.7)
temp = data['AdsAgreeDisagree2'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ads agreed or disagreed(%)', hole = 0.7)
temp = data['AdsAgreeDisagree3'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Ads agreed or disagreed(%)', hole = 0.7) 
temp = data['AIDangerous'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='AI technology dangerous aspects (%)', hole = 0.7)
temp = data['AIInteresting'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='AI technology interesting aspects(%)', hole = 0.7) 
temp = data['AIResponsible'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='AI technology responsibility(%)', hole = 0.7)
temp = data['AIFuture'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Future of AI technology(%)', hole = 0.7) 