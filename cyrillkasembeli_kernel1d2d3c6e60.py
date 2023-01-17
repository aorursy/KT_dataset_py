import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

data=pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

data.shape
#a glimpse of the dataset to know what it entails

data.head(50)
data.drop(["HDI for year"], axis = 1, inplace = True) 

data.shape
data=data.dropna()

data.shape
countries=data['country'].unique().tolist()

len(countries)
generations=data['generation'].unique().tolist()

len(generations)
united_states=data[data['country'].str.contains('United States')]

united_states.head(10)
years =data['year'].unique().tolist()

print(min(years), 'to' ,max(years))
#select the United States as the country of interest and separate male and female statistics

country_name='United States'

gender1 ='male'

gender2 ='female'







mask1=data['country'].str.contains(country_name)

mask2=data['sex'].str.match(gender1)

mask3=data['sex'].str.match(gender2)

mask4=data['year']> 2010

mask5=data['age'].str.match('35-54 years')



#stage1 is the data matching United States for country and male gender(2011-2016)

#stage2 is the data matching United States for country and female gender(2011-2016)

#stage3 is the data matching United States for country and male gender(1985-2016) between 35-54 yrs

#stage4 is the data matching United States for country and female gender(1985-2016) between 35-54 yrs

stage1=data[mask1 & mask2 & mask4]

stage2=data[mask1 & mask3 & mask4]

stage3=data[mask1 & mask2 & mask5]

stage4=data[mask1 & mask3 & mask5]

stage2.head()
stage1.head(7)
# exploration of trends in male suicidal rates over the years 2011 to 2015 grouped by age between 5 and 74 years

#bar plot



bars1= (stage1[stage1['age'].str.contains('5-14 years')])['suicides_no']

bars2= (stage1[stage1['age'].str.contains('15-24 years')])['suicides_no']

bars3= (stage1[stage1['age'].str.contains('25-34 years')])['suicides_no']

bars4= (stage1[stage1['age'].str.contains('35-54 years')])['suicides_no']

bars5= (stage1[stage1['age'].str.contains('55-74 years')])['suicides_no']





# set width of bar

barWidth = 0.15



 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]

r5 = [x + barWidth for x in r4]



# Make the plot

plt.figure(figsize=(10,6))

plt.bar(r1, bars1, color='#FF0000', width=barWidth, edgecolor='white', label='5-14 years')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='15-24 years')

plt.bar(r3, bars3, color='#AFEEEE', width=barWidth, edgecolor='white', label='25-34 years')

plt.bar(r4, bars4, color='#FF7F50', width=barWidth, edgecolor='white', label='35-54 years')

plt.bar(r5, bars5, color='#00008B', width=barWidth, edgecolor='white', label='55-74 years')

 

# Add xticks on the middle of the group bars

plt.xlabel('years', fontweight='bold')

plt.ylabel('No. of Suicides')

plt.title('NUMBER OF MALE SUICIDES IN THE UNITED STATES')

plt.xticks([r + barWidth for r in range(len(bars1))], ['2011', '2012', '2013', '2014', '2015'])

 

# Create legend & Show graphic

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.75))



plt.show()



# exploration of trends in female suicidal rates over the years 2011 to 2015 grouped by age between 5 and 74 years

#bar plot



bars1= (stage2[stage2['age'].str.contains('5-14 years')])['suicides_no']

bars2= (stage2[stage2['age'].str.contains('15-24 years')])['suicides_no']

bars3= (stage2[stage2['age'].str.contains('25-34 years')])['suicides_no']

bars4= (stage2[stage2['age'].str.contains('35-54 years')])['suicides_no']

bars5= (stage2[stage2['age'].str.contains('55-74 years')])['suicides_no']





# set width of bar

barWidth = 0.15



 

# Set position of bar on X axis

r1 = np.arange(len(bars1))

r2 = [x + barWidth for x in r1]

r3 = [x + barWidth for x in r2]

r4 = [x + barWidth for x in r3]

r5 = [x + barWidth for x in r4]



# Make the plot

plt.figure(figsize=(10,6))

plt.bar(r1, bars1, color='#FF0000', width=barWidth, edgecolor='white', label='5-14 years')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='15-24 years')

plt.bar(r3, bars3, color='#AFEEEE', width=barWidth, edgecolor='white', label='25-34 years')

plt.bar(r4, bars4, color='#FF7F50', width=barWidth, edgecolor='white', label='35-54 years')

plt.bar(r5, bars5, color='#00008B', width=barWidth, edgecolor='white', label='55-74 years')

 

# Add xticks on the middle of the group bars

plt.xlabel('years', fontweight='bold')

plt.ylabel('No. of Suicides')

plt.title('NUMBER OF FEMALE SUICIDES IN THE UNITED STATES')

plt.xticks([r + barWidth for r in range(len(bars1))], ['2011', '2012', '2013', '2014', '2015'])

 

# Create legend & Show graphic

plt.legend(loc='lower left', bbox_to_anchor=(1, 0.75))



plt.show()



#analysis of suicide rates between 1985 and 2016  both male and female in age between 35- 54 years

#duble line plots



l_years=np.sort(np.asarray(years, dtype=np.float32))[:-1]

y1= stage3['suicides_no'].values

y2= stage4['suicides_no'].values



plt.figure(figsize=(8,6))

plt.ylim(0, 14000)

plt.plot(l_years,y1,label='Male')

plt.plot(l_years,y2, label='Female')



plt.xlabel('Year')

plt.ylabel('No. 0f Suicides ')

plt.title('SUICIDES IN THE UNITED STATES AGE (35-54) YEARS')



plt.legend()

plt.show
# describe the countries gdp per capita over the years 

#it is the same for both male and female

stage1['gdp_per_capita ($)'].describe()
stage3.head()
stage3['suicides/100k pop'].corr(stage3['gdp_per_capita ($)'])
stage4['suicides/100k pop'].corr(stage4['gdp_per_capita ($)'])
#male scatter plot for correlation between no. of suicides and Gdp per capita 

%matplotlib inline

import matplotlib.pyplot as plt

fig,axis=plt.subplots()

#axis.yaxis.grid(True)



axis.set_title('NO. OF MALE SUICIDES vs. GDP PER CAPITA', fontsize=10)

axis.set_xlabel('No. of suicides', fontsize=10)

axis.set_ylabel('GDP per capita ($)', fontsize=10)

x=stage3['suicides_no']

y=stage3['gdp_per_capita ($)']

axis.scatter(x, y)

plt.show()
#female scatter plot for correlation between no. of suicides and Gdp per capita 

%matplotlib inline

import matplotlib.pyplot as plt

fig,axis=plt.subplots()

#axis.yaxis.grid(True)



axis.set_title('NO. OF FEMALE SUICIDES vs. GDP PER CAPITA', fontsize=10)

axis.set_xlabel('No. of suicides', fontsize=10)

axis.set_ylabel('GDP per capita ($)', fontsize=10)

x=stage4['suicides_no']

y=stage4['gdp_per_capita ($)']

axis.scatter(x, y)

plt.show()
mask6=data['country'].str.contains('United Kingdom')

mask7=data['country'].str.contains('Japan')

mask8=data['country'].str.contains('Italy')

mask9=data['country'].str.contains('Greece')





#stage6 is the data matching United Kingdom for country and male gender(1985-2016) between 35-54 yrs

#stage7 is the data matching Japan for country and male gender(1985-2016) between 35-54 yrs

#stage8 is the data matching Italy for country and male gender(1985-2016) between 35-54 yrs

#stage9 is the data matching Greece for country and male gender(1985-2016) between 35-54 yrs

stage6=data[mask6 & mask2 & mask5]

stage7=data[mask7 & mask2 & mask5]

stage8=data[mask8 & mask2 & mask5]

stage9=data[mask9 & mask2 & mask5]







l_years=np.sort(np.asarray(years, dtype=np.float32))[:-1]

y1= stage3['suicides_no'].values

y6= stage6['suicides_no'].values

y7= stage7['suicides_no'].values

y8= stage8['suicides_no'].values

y9= stage9['suicides_no'].values



plt.figure(figsize=(8,6))

plt.ylim(0, 14000)

plt.plot(l_years,y1,label=' United States')

plt.plot(l_years,y6, label='United Kingdom')

plt.plot(l_years,y7, label='Japan')

plt.plot(l_years,y8, label='Italy')

plt.plot(l_years,y9, label='Greece')



plt.xlabel('Year')

plt.ylabel('No. 0f Suicides ')

plt.title('MALE SUICIDES IN THE AGE (35-54) YEARS')



plt.legend()

plt.show
#stage6 is the data matching United Kingdom for country and female gender(1985-2016) between 35-54 yrs

#stage7 is the data matching Japan for country and female gender(1985-2016) between 35-54 yrs

#stage8 is the data matching Italy for country and female gender(1985-2016) between 35-54 yrs

#stage9 is the data matching Greece for country and female gender(1985-2016) between 35-54 yrs

stage10=data[mask6 & mask3 & mask5]

stage11=data[mask7 & mask3 & mask5]

stage12=data[mask8 & mask3 & mask5]

stage13=data[mask9 & mask3 & mask5]
%matplotlib inline

l_years=np.sort(np.asarray(years, dtype=np.float32))[:-1]

y1= stage4['suicides_no'].values

y10= stage10['suicides_no'].values

y11= stage11['suicides_no'].values

y12= stage12['suicides_no'].values

y13= stage13['suicides_no'].values



plt.figure(figsize=(6,4))

plt.ylim(0, 5000)

plt.plot(l_years,y1,label=' United States')

plt.plot(l_years,y10, label='United Kingdom')

plt.plot(l_years,y11, label='Japan')

plt.plot(l_years,y12, label='Italy')

plt.plot(l_years,y13, label='Greece')



plt.xlabel('Year')

plt.ylabel('No. 0f Suicides ')

plt.title('FEMALE SUICIDES IN THE AGE (35-54) YEARS')



plt.legend()

plt.show