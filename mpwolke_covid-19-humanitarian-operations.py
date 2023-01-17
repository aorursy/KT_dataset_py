# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://cdn.cnn.com/cnnnext/dam/assets/200911103045-greece-lesbos-moria-camp-refugees-no-shelter-bell-pkg-intl-hnk-vpx-00000707-super-169.jpg')
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadshumanitariancsv/humanitarian.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'humanitarian.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
#! pip install -q dabl
from colorama import Fore, Style



def count(string: str, color=Fore.RED):

    """

    Saves some work 😅

    """

    print(color+string+Style.RESET_ALL)
def statistics(dataframe, column):

    count(f"The Average value in {column} is: {dataframe[column].mean():.2f}", Fore.RED)

    count(f"The Maximum value in {column} is: {dataframe[column].max()}", Fore.BLUE)

    count(f"The Minimum value in {column} is: {dataframe[column].min()}", Fore.YELLOW)

    count(f"The 25th Quantile of {column} is: {dataframe[column].quantile(0.25)}", Fore.GREEN)

    count(f"The 50th Quantile of {column} is: {dataframe[column].quantile(0.50)}", Fore.CYAN)

    count(f"The 75th Quantile of {column} is: {dataframe[column].quantile(0.75)}", Fore.MAGENTA)
# Print Age Column Statistics

statistics(df, 'Vulnerability')
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(df['Vulnerability'], color='blue')

plt.title(f"Vulnerability Distribution [\u03BC : {df['Vulnerability'].mean():.2f} conditions | \u03C3 : {df['Vulnerability'].std():.2f} conditions]")

plt.xlabel("Vulnerability")

plt.ylabel("Count")

plt.show()
# Print Age Column Statistics

statistics(df, 'Covid19HazardExposure')
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(df['Covid19HazardExposure'], color='red')

plt.title(f"Covid19 Hazard Exposure Distribution [\u03BC : {df['Covid19HazardExposure'].mean():.2f} danger | \u03C3 : {df['Covid19HazardExposure'].std():.2f} danger]")

plt.xlabel("Covid19 Hazard Exposure")

plt.ylabel("Count")

plt.show()
# Print Age Column Statistics

statistics(df, 'IFI COVID19 total')
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('SchoolClosure').size()/df['SchoolClosure'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
# Print Age Column Statistics

statistics(df, 'Risk')
# Count Plot

plt.style.use("classic")

plt.figure(figsize=(10, 8))

sns.countplot(df['Risk'])

plt.xlabel("Risk ")

plt.ylabel("Count")

plt.title("Risk CountPlot")

plt.show()
# Print Age Column Statistics

statistics(df, 'LackOfCopingCapacity')
# Dist plot of Resting Blood Pressure

plt.style.use("classic")

sns.distplot(df['LackOfCopingCapacity'], color='green')

plt.title(f"Lack of Coping Capacity Count [\u03BC : {df['LackOfCopingCapacity'].mean():.2f} risks | \u03C3 : {df['LackOfCopingCapacity'].std():.2f} risks]")

plt.xlabel("Lack of Coping Capacity")

plt.ylabel("Count")

plt.show()
# Print Age Column Statistics

statistics(df, 'PeopleInNeed')
# Dist plot of Resting Blood Pressure

plt.style.use("classic")

sns.distplot(df['PeopleInNeed'], color='orange')

plt.title(f"PeopleInNeed Count [\u03BC : {df['PeopleInNeed'].mean():.2f} risks | \u03C3 : {df['PeopleInNeed'].std():.2f} risks]")

plt.xlabel("People in Need")

plt.ylabel("Count")

plt.show()
plt.style.use("ggplot")

plt.figure(figsize=(18, 9))

sns.boxplot(df['LackOfCopingCapacity'], df['Vulnerability'])

plt.title("Vulnerability & Lack of Coping Capacity")

plt.xlabel("Lack of Coping Capacity")

plt.ylabel("Vulnerability")

plt.show()
plt.style.use("fivethirtyeight")

plt.figure(figsize=(12, 6))

sns.swarmplot(df['Covid19HazardExposure'], df['PeopleInNeed'])

plt.title("People in Need v/s Covid19 Hazard Exposure")

plt.xlabel("Covid19 Hazard Exposure")

plt.ylabel("People in Need")

plt.show()
plt.figure(figsize=(10, 6))

sns.set(style='ticks')

scatter_df = df[["Vulnerability", "Covid19HazardExposure", "LackOfCopingCapacity"]]

sns.pairplot(scatter_df)

plt.show()
# Let's make a barplot of Yearly sales

covid_exposure = df[['Covid19HazardExposure']].stack().value_counts().tolist()

covid_risk = [int(x) for x in dict(df[['Covid19HazardExposure']].stack().value_counts()).keys()]



plt.style.use("ggplot")

sns.barplot(x=covid_risk, y=covid_exposure, palette='Accent_r')

plt.xticks(rotation=55, fontsize=8)

plt.xlabel("Covid19 Hazard Exposure")

plt.ylabel("Covid19 Exposure")

plt.title("Covid19 Hazard Exposure")

plt.show()
# Let's make a barplot of Top-10 Genre by NA Sales 

plt.style.use("ggplot")

sns.barplot(data=df[:10], x='Vulnerability', y='PeopleInNeed', palette='Accent_r')

plt.xticks(rotation=90, fontsize=10)

plt.xlabel("Vulnerability")

plt.ylabel("PeopleInNeed")

plt.title("People in Need Vulnerability")

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.SchoolClosure)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
df[numerical_nan].isna().sum()
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>0 and df[feature].dtypes=='O']

print(categorical_nan)
df[categorical_nan].isna().sum()
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
df[categorical_nan].isna().sum()
#import dabl



#dabl.plot(df, target_col='Vulnerability')