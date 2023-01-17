# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt # graphing capabilities



import seaborn as sns# for data viz.

import plotly.express as px



from plotly.subplots import make_subplots# for subplots using plotly



import plotly.graph_objects as go





pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
multiple_choice_responses.shape
country=multiple_choice_responses.Q3.value_counts()



#marker_colorscale = 'Blues'

fig = go.Figure(go.Treemap(

    labels = country.index,

    parents=['World']*len(country),

    values = country

))



fig.update_layout(title = 'Country of Survey Participants')

fig.show()
#https://www.worldometers.info/world-population/population-by-country/  (2019)

population = [1366417754, 329064917, 211049527, 126860301, 145872256, 1433783686, 83517045, 67530172, 37411047, 46736776,

200963599, 65129728, 23773876, 83429615, 60550075, 25203198, 37887768, 216565318, 127575529, 43993638, 51225308,

50339443, 270625568, 17097130, 5804337, 163046161, 96462106, 44780677, 36471769, 100388073, 58558270, 10226187, 

52573973, 10473455, 8519377, 8591365, 82913906, 10036379, 18952038, 4882495, 31949777, 32510453, 51225308,

11539328, 9452411, 11694719, 69625582, 108116615, 7436154, 10689209, 43053054, 19364557, 9684679, 5771876,

8955102, 5378857, 4783063, 34268528]
df = pd.DataFrame({'state': country.index, 'value': country.values})

df.drop([2,59], axis = 0, inplace = True)

df['population'] = population

df['proportion'] = df['value']/df['population']



fig = go.Figure(go.Treemap(

    labels = df['state'],

    parents=['World']*58,

    values = df['proportion']

))



fig.update_layout(title = 'Country of Survey Participants')

fig.show()
#sns.distplot(df['proportion'])

sns.distplot(df['proportion'],hist=True)

plt.show()
contengency = pd.crosstab(multiple_choice_responses.Q3.drop([0]), multiple_choice_responses.Q4.drop([0]))

contengency.sort_values(by = ["Bachelor’s degree"], ascending=False).head(10)
import scipy.stats as st

st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(contengency)

print('Khi2_calculate:', st_chi2, 'P value of test:', st_p, 'degree of freedom:', st_dof)
heatData1 = (contengency.values - st_exp)/st_exp

heatData1 = pd.DataFrame(heatData1, columns=contengency.columns, index = contengency.index)

plt.figure(figsize=(20, 20))

sns.heatmap(heatData1, cmap="YlGnBu", annot=True)
B = heatData1[(heatData1["Doctoral degree"]<-0.5) | (heatData1["Doctoral degree"]>0.5)].sort_values(by = ["Doctoral degree"])

height = B["Doctoral degree"].values*100

var = B.index.values

plt.figure(figsize = (15,10))

pos = np.arange(len(var))

bars = plt.barh(pos, height)

plt.yticks(pos, var)

plt.xlabel("deviation from statistical independence")

plt.title('statistical difference (Doctoral degree as high level school versus country)')

plt.tick_params(top = False, left = False, bottom = False, labelbottom = False)

# remove frames

for spine in plt.gca().spines.values():

    spine.set_visible(False)



for bar in bars:

    plt.gca().text(bar.get_width()+1.2, bar.get_y()+bar.get_height()/2, str(int(bar.get_width()))+"%")

    

heatData1.columns
C = heatData1[(heatData1["No formal education past high school"]<=-1) | (heatData1["No formal education past high school"]>1)].sort_values(by = ["No formal education past high school"])

height = C["No formal education past high school"].values*100

var = C.index.values

plt.figure(figsize = (15,10))

pos = np.arange(len(var))

bars = plt.barh(pos, height)

plt.yticks(pos, var)

plt.xlabel("deviation from statistical independence")

plt.title('No formal education past high school')

plt.tick_params(top = False, left = False, bottom = False, labelbottom = False)

# remove frames

for spine in plt.gca().spines.values():

    spine.set_visible(False)



for bar in bars:

    plt.gca().text(bar.get_width()+1.2, bar.get_y()+bar.get_height()/2, str(int(bar.get_width()))+"%")

    
tpuSchool = pd.crosstab(multiple_choice_responses.Q4.drop([0]), multiple_choice_responses.Q22.drop([0]))

tpuSchool.head(10)
st_chi2, st_p, st_dof, st_exp = st.chi2_contingency(tpuSchool)

print('Khi2_calculate:', st_chi2, 'P value of test:', st_p, 'degree of freedom:', st_dof)
heatData2 = (tpuSchool.values - st_exp)/st_exp

heatData2 = pd.DataFrame(heatData2, columns=tpuSchool.columns, index = tpuSchool.index)

plt.figure(figsize=(20, 20))

sns.heatmap(heatData2, cmap="YlGnBu", annot=True)