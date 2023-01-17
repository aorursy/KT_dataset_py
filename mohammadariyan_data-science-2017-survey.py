import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

import seaborn as sns
data = pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding='ISO-8859-1')
plt.figure(figsize=(10,5))

plt.title('language recommendation')

sns.countplot(y=data['LanguageRecommendationSelect'],

              order=data['LanguageRecommendationSelect'].value_counts().index,) 
usefulness=pd.concat([data['LearningPlatformUsefulnessTextbook'].value_counts(),

                     data['LearningPlatformUsefulnessKaggle'].value_counts(),

                     data['LearningPlatformUsefulnessArxiv'].value_counts(),

                     data['LearningPlatformUsefulnessBlogs'].value_counts(),

                     data['LearningPlatformUsefulnessCollege'].value_counts(),

                     data['LearningPlatformUsefulnessCompany'].value_counts(),

                     data['LearningPlatformUsefulnessConferences'].value_counts(),

                     data['LearningPlatformUsefulnessFriends'].value_counts(),

                     data['LearningPlatformUsefulnessNewsletters'].value_counts(),

                     data['LearningPlatformUsefulnessCommunities'].value_counts(),

                     data['LearningPlatformUsefulnessDocumentation'].value_counts(),

                     data['LearningPlatformUsefulnessCourses'].value_counts(),

                     data['LearningPlatformUsefulnessProjects'].value_counts(),

                     data['LearningPlatformUsefulnessPodcasts'].value_counts(),

                     data['LearningPlatformUsefulnessSO'].value_counts(),

                     data['LearningPlatformUsefulnessTradeBook'].value_counts(),

                     data['LearningPlatformUsefulnessTutoring'].value_counts(),

                     data['LearningPlatformUsefulnessYouTube'].value_counts()],keys=['text','kaggle','Arxiv',

                                                                                    'Blogs','College','Company',

                                                                                     'Conferences','Friends','Newsletters',

                                                                                     'Communities','Documentation','Courses',

                                                                                     'Projects','Podcasts','SO',

                                                                                      'TradeBook','Tutoring','YouTube'])

type(usefulness.index) # so we can use unstack command

ufn= usefulness.unstack(level=1)

ufn.plot.barh(figsize=(15,10),fontsize=12)
usefulness=pd.concat([data['JobSkillImportanceBigData'].value_counts(),

                     data['JobSkillImportanceDegree'].value_counts(),

                     data['JobSkillImportanceStats'].value_counts(),

                     data['JobSkillImportanceEnterpriseTools'].value_counts(),

                     data['JobSkillImportancePython'].value_counts(),

                     data['JobSkillImportanceR'].value_counts(),

                     data['JobSkillImportanceSQL'].value_counts(),

                     data['JobSkillImportanceKaggleRanking'].value_counts(),

                     data['JobSkillImportanceMOOC'].value_counts(),

                     data['JobSkillImportanceVisualizations'].value_counts(),

                     data['JobSkillImportanceOtherSelect1'].value_counts(),

                     data['JobSkillImportanceOtherSelect2'].value_counts(),

                     data['JobSkillImportanceOtherSelect3'].value_counts(),

                     ],keys=['big data','degree','stats',

                                                                                    'tools','python','R',

                                                                                     'SQL','kaggle rank','mooc',

                                                                                     'Visualizations','other 1','other 2',

                                                                                     'other 3'])

type(usefulness.index) # so we can use unstack command

ufn= usefulness.unstack(level=1)

ufn.plot.barh(figsize=(15,10),fontsize=12)
data['LearningPlatformSelect'].value_counts()[:10].plot.barh()
plt.subplots(figsize=(10,4))

sns.countplot(y=data['GenderSelect'],order=data['GenderSelect'].value_counts().index

            )
data["LearningDataScienceTime"].value_counts()
for i in data.columns:

    print(i)
data['CurrentJobTitleSelect'].value_counts().plot.barh()
mask = data['CurrentJobTitleSelect'] == 'Statistician' 

statguys=data[mask]
labels=statguys['LanguageRecommendationSelect'].value_counts().plot.barh() # how predictive is that
mask = data['CurrentJobTitleSelect'] == 'Data Scientist'

DSguys=data[mask]
DSguys['LanguageRecommendationSelect'].value_counts().plot.barh()