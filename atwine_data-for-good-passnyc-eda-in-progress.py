# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
%matplotlib inline

pd.set_option('display.max_columns', None)  
df = pd.read_csv('../input/2016 School Explorer.csv')
registrations = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
df.head(3)
# There are 1272 schools in the data set
df.shape
df['School Income Estimate'] = df['School Income Estimate'].str.replace(',', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace('$', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace(' ', '')
df['School Income Estimate'] = df['School Income Estimate'].astype(float)
df.plot(kind="scatter", x="Longitude", y="Latitude",
    s=df['School Income Estimate']/1210, c="Economic Need Index", cmap=plt.get_cmap("jet"),label='Schools', title='New York School Population Map',
    colorbar=True, alpha=0.4, figsize=(15,7))
plt.legend()
plt.show()
# Create a function to convert the percentage to a fraction 
def p2f(x):
    return float(x.strip('%'))/100
df['Percent Asian'] = df['Percent Asian'].apply(p2f)
df['Percent Black'] = df['Percent Black'].apply(p2f)
df['Percent Hispanic'] = df['Percent Hispanic'].apply(p2f)
df['Percent White'] = df['Percent White'].apply(p2f)
df['Percent Black / Hispanic'] = df['Percent Black / Hispanic'].apply(p2f)
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[0],
    s=df['Economic Need Index']*100, c="Percent Black", cmap=plt.get_cmap("jet"),label='Schools', title='Black Population Percentage of Schools',
    colorbar=True, alpha=0.3, figsize=(15,7))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[1],
    s=df['Economic Need Index']*100, c="Percent Hispanic", cmap=plt.get_cmap("jet"),label='Schools', title='Hispanic Population Percentage of Schools',
    colorbar=True, alpha=0.3, figsize=(15,7))

plt.legend()
plt.show()
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[0],
    s=df['Economic Need Index']*100, c="Percent Asian", label='Schools', title='Asian Population Percentage of Schools',
    colorbar=True, alpha=0.3, figsize=(15,5))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[1],
    s=df['Economic Need Index']*100, c="Percent White", label='Schools', title='White Population Percentage of Schools',
    colorbar=True, alpha=1, figsize=(15,5))

plt.legend()
plt.show()

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(df['Percent Asian'], kde=False, color="g", ax=axes[0], bins=25).set_title('% Asian Distribution')
axes[0].set_ylabel('School Count')

# Graph Employee Evaluation
sns.distplot(df['Percent White'], kde=False, color="r", ax=axes[1], bins=25).set_title('% White Distribution')
axes[1].set_ylabel('Employee Count')
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

# Graph Employee Average Monthly Hours
sns.distplot(df['Percent Black'], kde=False, color="b", ax=axes[0], bins=35).set_title('% Black Distribution')
axes[0].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(df['Percent Hispanic'], kde=False, color="r", ax=axes[1], bins=35).set_title('% Hispanic Distribution')
axes[1].set_ylabel('Employee Count')

#sns.distplot(df['School Income Estimate'])
df['School Income Estimate'].describe()
median = df['School Income Estimate'].median()
df['School Income Estimate'] = df['School Income Estimate'].fillna(median)
df_income = df[df['School Income Estimate'] < 100000]
sns.distplot(df_income['School Income Estimate'], bins=25)
# Overview of summary (Turnover V.S. Non-turnover)
turnover_Summary = df.groupby('Community School?')
turnover_Summary[['Economic Need Index', 'School Income Estimate', 'Percent Asian','Percent Black', 'Percent Hispanic', 
                  'Percent White', 'Average ELA Proficiency', 'Average Math Proficiency']].mean()
# Let's normalize the School Income Estimate to have better visualization
from sklearn import preprocessing

# Impute the missing Income with the median
median = df['School Income Estimate'].median()
df['School Income Estimate'] = df['School Income Estimate'].fillna(median)

# Create x, where x the 'scores' column's values as floats
x = df[['School Income Estimate']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
school_income_normalized = pd.DataFrame(x_scaled)

df['school_income_normalized'] = school_income_normalized
sns.lmplot(x='school_income_normalized', y='Economic Need Index', data=df,
           fit_reg=False) # No regression line
        
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Community School?'] == 'Yes'),'School Income Estimate'] , color='b',shade=True, label='Community School')
ax=sns.kdeplot(df.loc[(df['Community School?'] == 'No'),'School Income Estimate'] , color='r',shade=True, label='Not Community School')
plt.title('Community School VS Not Community School Income')
registrations.head(3)
pd.DataFrame(registrations.groupby('Year of SHST')['Number of students who registered for the SHSAT'].sum())
school_registrations = registrations.groupby('School name')['Number of students who registered for the SHSAT'].sum().reset_index().sort_values('Number of students who registered for the SHSAT', ascending=False).reset_index(drop=True)
school_registrations.head(10)
f, ax = plt.subplots(figsize=(15, 10))

#Get List of School Names
school_names = list(school_registrations['School name'].values)

y_pos = np.arange(len(school_registrations))
# Create horizontal bars
plt.barh(y_pos, school_registrations['Number of students who registered for the SHSAT'], color='g')
 
# Create names on the y-axis
plt.yticks(y_pos, school_names)
plt.xlabel('SHSAT Registration Count')
plt.ylabel('School name')
plt.title('Total SHSAT Registration Count Per School (2013-2016)')
 
# Show graphic
plt.show()
