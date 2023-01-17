# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



data = pd.read_csv('../input/weatherAUS.csv')

data.columns





# Any results you write to the current directory are saved as output.
data.Location.unique()
# Shall we focus our attention on the following:

df = data.loc[(data['Location'] == 'Sydney') | (data['Location'] == 'Perth') |

                        (data['Location'] == 'Adelaide') | (data['Location'] == 'Hobart') |

                        (data['Location'] == 'AliceSprings') | (data['Location'] == 'Melbourne') |

                        (data['Location'] == 'Canberra') | (data['Location'] == 'Darwin') |

                        (data['Location'] == 'Uluru') | (data['Location'] == 'Townsville') |

                        (data['Location'] == 'Brisbane') | (data['Location'] == 'Tuggeranong')

                       ]
df.shape
# dropping NA values, split column Date into two new columns covering Year and Month

df = df.dropna()

df['Year']=[d.split('-')[0] for d in df.Date]

df['Month']=[d.split('-')[1] for d in df.Date]
# Median Monthly MinTemp per Location

MonthlyMedMin = df.groupby(["Location","Month"])["MinTemp"].median().reset_index()

# Median Monthly MaxTemp per Location

MonthlyMedMax = df.groupby(["Location","Month"])["MaxTemp"].median().reset_index()

# Monthly Min per Location

MonthlyMin = df.groupby(["Location","Month"])["MinTemp"].min().reset_index()

# Monthly Min per Location

MonthlyMax = df.groupby(["Location","Month"])["MaxTemp"].max().reset_index()





# Join Tables and change labels.

OutputMed = pd.merge(MonthlyMedMin,MonthlyMedMax,how = 'right',

                     on = ['Location','Month'])

OutputMinMax = pd.merge(MonthlyMin,MonthlyMax,how = 'right',

                     on = ['Location','Month'])



OutputMed.rename(columns = {'MinTemp':'MedMin','MaxTemp':'MedMax'}, inplace = True)



output_month = pd.merge(OutputMed,OutputMinMax,how = 'right',

                     on = ['Location','Month'])

output_month[:8]

# I want an extra column with the average temperature between Min and Max values

output_month['AverageTemp'] = (output_month['MaxTemp'] + output_month['MinTemp'])/2

output_month[:5]
import matplotlib.pyplot as plt

areas = output_month.Location.unique()



for i in areas:



    plt.style.use('ggplot')

    area = output_month.loc[(output_month['Location'] == i)]

    area.plot(x = 'Month',y = ['MedMin','MedMax','MinTemp','MaxTemp','AverageTemp'],

              figsize = (35,10),title = i,fontsize = 25,linewidth = 8.0)

    

    

# Let's check the amount of rain:

# Median Monthly MinTemp per Location

AccuMonthlyRain = df.groupby(["Location","Month"])["Rainfall"].median().reset_index()



AccuMonthlyRain[:10]
# import matplotlib.pyplot as plt

# rainAreas = AccuMonthlyRain.Location.unique()



# for rain in rainAreas:



#     plt.style.use('ggplot')

#     rains = AccuMonthlyRain.loc[(AccuMonthlyRain['Location'] == rain)]

#     rains.plot.bar(x = 'Month',y = 'Rainfall',

#               figsize = (35,10),

#                title = rain,

#                fontsize = 25,

#                linewidth = 8.0)