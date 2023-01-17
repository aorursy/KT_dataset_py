import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
district = pd.read_csv(r'../input/covid19-corona-virus-india-dataset/district_level_latest.csv')
district.head()
district = district[district['state name'] == 'Karnataka']
district
district.drop(['delta_confirmed','delta_deceased','delta_recovered','notes','state name','state code'],axis = 1, inplace = True)
district
district['district'][district['district'] == 'Bengaluru Rural'] = 'Bengaluru_Rural'
district['district'][district['district'] == 'Bengaluru Urban'] = 'Bengaluru_Urban'

district['district'][district['district'] == 'Dakshina Kannada'] = 'Dakshina_Kannada'

district['district'][district['district'] == 'Uttara Kannada'] = 'Uttara_Kannada'
district.drop(301,axis = 0,inplace = True)
district.describe()
size = district['confirmed']
ls = []

for i in district['district']:

    x = i + str(district[district['district'] == i]['confirmed'])

    x = x.split()[:2]

    for i in x:

        if i[0].isalpha():

            d = x.index(i)

            i = i[:-3]

            x[d] = i

    ls.append(x)

label = [' - '.join(i) for i in ls]
fig1, ax1 = plt.subplots(figsize=(10, 5))

fig1.subplots_adjust(0.3,0,1,1)

ax1.pie(size, startangle=90)



ax1.axis('equal')



total = sum(size)

plt.legend(

    loc='upper left',

    prop={'size': 11},

    bbox_to_anchor=(0.0, 1),

    bbox_transform=fig1.transFigure,

    labels = label

)



plt.show()
import plotly.express as px

fig = px.pie(district, values='confirmed', names='district', title='Confirmed cases in Karnataka')

fig.show()
fig = px.pie(district, values='recovered', names='district', title='Recovered cases in Karnataka')

fig.show()
fig = px.pie(district, values='active', names='district', title='Active cases in Karnataka')

fig.show()
import plotly.express as px

fig = px.pie(district, values='deceased', names='district', title='Deceased cases in Karnataka')

fig.show()
sns.jointplot(district['confirmed'],district['recovered'],kind = 'hex')
sns.jointplot(district['confirmed'],district['deceased'],kind = 'hex')
sns.jointplot(district['confirmed'],district['active'],kind = 'hex')
sns.jointplot(district['recovered'],district['active'],kind = 'hex')