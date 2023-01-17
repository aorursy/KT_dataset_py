#Import lib

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation

import seaborn as sns; sns

import numpy as np

from sklearn import svm

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px

from sklearn.preprocessing import Imputer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_wine

#Import lin

np.set_printoptions(precision=4)

sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import os

os.getcwd()
# Importing Data

Toronto_climate=pd.read_csv("../input/climate-data-toronto-19372018/Toronto_temp.csv")

Toronto_climate.head(50)
# to get number of miising values

Toronto_climate.isnull().sum()
# print the last 20 rows of data

# I did this to check if data has NAN values as first 20 row of data didnt show the NAN

print(Toronto_climate.tail(20))
#get the number of varible in each column

Toronto_climate.count()
#Describtive statistics

Toronto_climate.describe()
#get column names

Toronto_climate.columns
#Rename the date/time column to date

Toronto_climate=Toronto_climate.rename(columns=

              {'Date/Time':

              'Date'})

Toronto_climate.columns
#Removal of , from the values of column Date

Toronto_climate.Date=Toronto_climate.Date.str.replace(',',  '')
#Removal of , from the values of column year

Toronto_climate.Year=Toronto_climate.Year.str.replace(",",  '')
#Removal of , from the values of column season

Toronto_climate.season=Toronto_climate.season.str.replace(",",  '')
#extract column Year', 'Month', 'Day', 'Mean Temp (C)' into a new dataset called Tnum

#these are columns in the Toronto_climate dataset containing only numbers

Tnum=pd.DataFrame(Toronto_climate, columns =[ 'Year', 'Month', 'Day', 'Mean Temp (C)'])

Tnum.head()
# plot between 2 columns of the new extracted data

plt.bar(Tnum['Month'], Tnum['Mean Temp (C)']) 

plt.xlabel("Month") 

plt.ylabel("Mean Temp (C)") 

plt.show() 
# Get a series of unique values in column 'Age' of the dataframe

#this helps us know the number of unique variable

uniqueValues = Toronto_climate['Year'].unique()

 

print('Unique elements in column "Year" ')

print(uniqueValues)
#extracted data with just the month of Febuary

Feb_only =  Toronto_climate.loc[Toronto_climate.Month== 2]

Feb_only
#creating dataset with just the month of Feb containing first year of each decade over a period of 8 decades

Feb_2018=Feb_only.loc[Feb_only.Year== '2018']

Feb_2008=Feb_only.loc[Feb_only.Year== '2008']

Feb_1998=Feb_only.loc[Feb_only.Year== '1998']

Feb_1988=Feb_only.loc[Feb_only.Year== '1998']

Feb_1978=Feb_only.loc[Feb_only.Year== '1978']

Feb_1968=Feb_only.loc[Feb_only.Year== '1968']

Feb_1958=Feb_only.loc[Feb_only.Year== '1958']

Feb_1948=Feb_only.loc[Feb_only.Year== '1948']

Feb_1938=Feb_only.loc[Feb_only.Year== '1938']
#plot the graph of day vs mean temp for the data Feb_2018

Feb_2018.plot("Day",['Mean Temp (C)'],kind='bar')
#create a new dataset called frame containing first year of each decade over a period of 8 decades

frames = [Feb_2008,Feb_1998,Feb_1988,Feb_1978,Feb_1968,Feb_1958,Feb_1948,Feb_1938]

#concatenate dataframes

df = pd.concat(frames, sort=False)

df
#plot a graph of created dataframe

df.plot("Year",['Min Temp (C)'],kind='bar')
#extracted data with just the month of August

Aug_only =  Toronto_climate.loc[Toronto_climate.Month== 8]

Aug_only
#creating dataset with just the month of Aug containing first year of each decade over a period of 8 decades

Aug_2018=Aug_only.loc[Aug_only.Year== '2018']

Aug_2008=Aug_only.loc[Aug_only.Year== '2008']

Aug_1998=Aug_only.loc[Aug_only.Year== '1998']

Aug_1988=Aug_only.loc[Aug_only.Year== '1998']

Aug_1978=Aug_only.loc[Aug_only.Year== '1978']

Aug_1968=Aug_only.loc[Aug_only.Year== '1968']

Aug_1958=Aug_only.loc[Aug_only.Year== '1958']

Aug_1948=Aug_only.loc[Aug_only.Year== '1948']

Aug_1938=Aug_only.loc[Aug_only.Year== '1938']
#create a new dataset called frame containing first year of each decade over a period of 8 decades

framesA = [Aug_2008,Aug_1998,Aug_1988,Aug_1978,Aug_1968,Aug_1958,Aug_1948,Aug_1938]

#concatenate dataframes

dfA = pd.concat(framesA, sort=False)

dfA
#plot a graph of created dataframe

dfA.plot("Year",['Min Temp (C)'],kind='bar')
#create a new dataset containg both df and dfA

dF = [df,dfA]

#concatenate dataframes

DF = pd.concat(dF, sort=False)

DF
#checking if there is a relationship between winter and summer season

sns.catplot(x="Year", y="Mean Temp (C)", hue="season", kind="point", data=DF);

sns.catplot(x="Year", y="Mean Temp (C)", hue="season", kind="bar", data=DF);
#create a dataset containg just feb 2018 and aug 2018

Main = [Feb_2018,Aug_2018]

#concatenate dataframes

Main_Feb = pd.concat(Main, sort=False)

Main_Feb
#plot 2 different type of  graph for created dataframe

sns.catplot(

    data=Main_Feb,

    x='Day',

    y='Mean Temp (C)',

    hue='season',

    kind='bar',

    height=5, # make the plot 5 units high

    aspect=3) 



sns.catplot(

    data=Main_Feb,

    x='Day',

    y='Mean Temp (C)',

    hue='season',

    kind='point',

    height=5, # make the plot 5 units high

    aspect=3) 
#plot boxplot of created dataframe

sns.catplot(

    data=Toronto_climate,

    x='Day',

    y='Mean Temp (C)',

    hue='season',

    kind='box',

    height=5, # make the plot 5 units high

    aspect=3)
# get a general description 

Feb_only.describe()
#LOGISTIC REGRESSION TO PREDICT A PARTICULAR SEASON IF ALL OTHER 

#VARIABLE [Year,Month,Day,Mean Temp (C),Max Temp (C),Min Temp (C),Total Rain (mm),Total Snow (cm),Total Precip (mm),] ARE GIVEN
#dealing with missing values 

#get a new copy of dataset

Newtoronto=Toronto_climate
#Apply the drop() functon to the new dataset also assign it to a newname 

NT=Newtoronto.drop('Date',axis=1)

NT
# fill missing values with mean column values using fillna() function

NT.fillna(Newtoronto.mean(), inplace=True)

# count the number of NaN values in each column
#get the number of missing values

NT.isnull().sum()
NT.values
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
#Assign values to X and Y

X = NT.iloc[:, 0:8].values

y = NT.iloc[:, 9].values



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Feature Scaling to Dataset

from sklearn.preprocessing import StandardScaler

np.seterr(divide='ignore', invalid='ignore')

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Implement LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 2)

X_train = lda.fit_transform(X_train, y_train)

X_test = lda.transform(X_test)
#Removal of IndexError: single positional indexer is out-of-bounds

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# Train Logistic Regression with LDA

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
# Predict Results of Regression with LDA

y_pred = classifier.predict(X_test)
# Confusion Matrix 

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
#labelled confusion matrix

df_confusion = pd.crosstab(y_test, y_pred)

print(df_confusion )
#Normalized confusion matrix

df_conf_norm = df_confusion / df_confusion.sum(axis=1)

print(df_conf_norm)
#Accuracy of regression

y_pred =  classifier.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format( classifier.score(X_test, y_test)))
#plot of confusion matrix

from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Wistia):

    plt.matshow(df_confusion, cmap=cmap) # imshow

    #plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(df_confusion.columns))

    plt.xticks(tick_marks, df_confusion.columns, rotation=45)

    plt.yticks(tick_marks, df_confusion.index)

    #plt.tight_layout()

    plt.ylabel(df_confusion.index.name)

    plt.xlabel(df_confusion.columns.name)



plot_confusion_matrix(df_confusion)
#Compute precision, recall, F-measure and support

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
#LINEAR REGRESSION to predict Total Precip from Month, Day and Mean Temp


# Pairs plot visualizing the relationships between our three selected predictors and the response



ax_3pairs = sns.pairplot(NT, x_vars=['Month','Day','Mean Temp (C)'], 

                         y_vars='Total Precip (mm)', height=5, aspect=0.7, kind='reg')



ax_3pairs.fig.suptitle('Total Precip (mm) vs. Month, Day, and Mean Temp (C)', y=1.03)
# Scatter plot for the relationship between Total Precip and Month

ax_scatter = sns.scatterplot(NT['Month'], Toronto_climate['Total Precip (mm)'])

ax_scatter.set_title("Total Precip (mm) vs. Month")
# Residual plot between Calories and Rating

ax_residual = sns.residplot(NT['Day'], NT['Total Precip (mm)'], lowess=True, color="r")

ax_residual.set_title("Total Precip (mm) vs. Day Residual Plot")
#x is the predictor variable for the response variable y

import scipy

scipy.stats.linregress(NT['Total Precip (mm)'], NT['Day'])