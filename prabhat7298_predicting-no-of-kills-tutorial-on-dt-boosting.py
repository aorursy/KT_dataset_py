import numpy as np  # Library for array processing , Linear algebra
import pandas as pd  # Library for data processing, data manipulation
import matplotlib.pyplot as plt  # Library for data visualisation
import seaborn as sns  # Library for different plots

from sklearn.model_selection import train_test_split  # To split data into training and validation data
from sklearn.metrics import mean_squared_error  # Evaluation metric

from subprocess import check_call # for running command line process

sns.set(style="whitegrid", color_codes=True) 
sns.set(font_scale=1)

from IPython.display import display 
pd.options.display.max_columns = None  # To display all columns in the notebook
from IPython.display import Image as PImage # To display all images inside the notebook

# Displaying graphs in the notebook itself
%matplotlib inline 

import warnings
warnings.filterwarnings('ignore')  # Doesn't display warnings
### START CODE HERE ###
# Read and store the data in a dataframe 'data' to be used for furthur processing (1 line of code)
data = pd.read_csv("pubg_kills.csv")
### END CODE HERE ###
# Display first five rows of the dataset
data.head()
# Similarily data.tail() shows last five rows of the data

### START CODE HERE ###
# Display the last five rows of the data (1 line of code)
data.tail()
### END CODE HERE ###
# Dimensions of the data
# Number of rows, Number of columns(features)
print(data.shape)
# print all the columns/features in the data
#length of dataset
len(data)
#To access a column player_survive_time
data['player_survive_time'].head()
#To access multiple columns
data[['party_size','player_kills']].head(4)
#To access a multiple rows
data.iloc[3:6]
#to change the date format
data['date'] =  pd.to_datetime(data['date'], format='%Y-%m-%dT%H:%M:%S+0000')
#extracting the weekday from date
data['Day'] = pd.DatetimeIndex(data['date']).weekday
weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
# Extracting hour from time
# creating new variable hour from the time variable 
data['Hour'] = pd.DatetimeIndex(data['date']).hour
# display first three rows of the data
data.head(3)
del data['date']  # As we have already extracted the useful info i.e. Weekday and Hour
del data['match_mode']  # Because all the matches were played in TPP (Third-Person Perspective) mode
del data['team_id']  # Because we already have match_id and player_name to uniquely identify an instance
# determining data types of the variable
data.dtypes
# continious variable analysis
data.describe()
# plot given numerical variable with respect to other variables
cont_vars = ['player_dbno', 'player_dist_walk', 'player_dmg', 'player_kills']
sns.pairplot(data[cont_vars])
#Plotting histogram for 'player_kills' variable
sns.distplot(data['player_kills'], color="purple", kde=False)
plt.title("Distribution of Number of Kills")
plt.ylabel("Number of Occurences")
plt.xlabel("Number of Kills");
#frequency of each value in weekday column
weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
dict(data.Day.value_counts())
#Plotting histogram for 'Day' variable
week_data = {'Mon': 14155, 'Tue': 13860, 'Wed': 13183, 'Thu': 11611, 'Fri': 14458, 'Sat': 16443, 'Sun': 16290}
names = list(week_data.keys())
values = list(week_data.values())

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axs[0].bar(names, values)
axs[1].plot(names, values)
fig.suptitle('Categorical Plotting')
#for more information -> https://chartio.com/resources/tutorials/what-is-a-box-plot/
sns.boxplot("game_size", data=data, showfliers=False)
plt.title("Distribution of game_size")
plt.xlabel("Number of Teams in Game");
# selecting categorical variables from the data
categorical_variables = ['party_size', 'Day', 'Hour']
#print categorical variables
print(categorical_variables)
# unique values count in each categorical variable
data[categorical_variables].apply(lambda x: len(x.unique()))
#frequency count of each categorical variable
for var in categorical_variables:
    print(var)
    print(data[var].value_counts())
    print('\n')
#display in pie chart
labels = data['party_size'].unique()
sizes = data['party_size'].value_counts().values
explode=[0.1,0,0]
parcent = 100.*sizes/sizes.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]

colors = ['yellowgreen', 'gold', 'lightblue']
patches, texts= plt.pie(sizes, colors=colors,explode=explode,
                        shadow=True,startangle=90)
plt.legend(patches, labels, loc="best")

plt.title("Party Size Classification")
plt.show()
#scatter plot
plt.scatter(np.sqrt(data["player_dmg"]), data["player_dbno"])
# to display title above the plot
plt.title("Hitpoints Dealt Vs Down but not out ")
# to label y-axis
plt.ylabel("No. of DBNO's")
# to label x-axis
plt.xlabel("Hitploints Dealt by the Player");
# correlation between variables 
# heat map
corrMatrix = data[["game_size", "player_assists", "player_dbno",
                   "player_dist_ride", "player_dist_walk", "player_dmg",
                   "player_survive_time", "team_placement", "player_kills"]].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(9, 9))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');
# sns.boxplot(x, y, argument to hide outliers)
sns.boxplot(data["party_size"], data["player_survive_time"], showfliers=False)
# title for the plot
plt.title("Survival Time vs Team Size")
plt.ylabel("Survival Time")
plt.xlabel("Team Size");
crosstable = pd.crosstab(data.Day, data.party_size)
crosstable
# Plotting stacked bar plot
crosstable.plot(kind='bar',stacked='True')
# Detecting missing values
data.isnull().sum()
#box plot
sns.boxplot("player_survive_time", data=data, showfliers=True)
plt.title("Distribution of Survival Time")
plt.xlabel("Survival Time");
#Treating outliers
# Removing Outliers
Q1 = data['player_survive_time'].quantile(.25)
Q3 = data['player_survive_time'].quantile(.75)
IQR = Q3-Q1
lower_value = IQR-1.5*Q1
upper_value = IQR+1.5*Q3
# print range lower_value and upper_value
lower_value, upper_value
#replacing outlier with meadian value the data
def outlier_imputer(x):
    if x < lower_value or x > upper_value:
        return data['player_survive_time'].median()
    else:
        return x
result = data['player_survive_time'].apply(outlier_imputer)  # This would take a lil bit time to run
sns.boxplot(result, showfliers=True)
plt.title("Distribution of Survival Time")
plt.xlabel("Survival Time");
#depenent_variable -> which we are going to predict
#independent_variable -> helps in predicting dependent_variable
dependent_variable = 'player_kills'
independent_variable = ['game_size', 'party_size', 'player_assists', 'player_dbno', 'player_dist_ride', 'Hour', 
                        'player_dist_walk', 'player_dmg', 'player_survive_time', 'team_placement', 'Day']
independent_variable
#library to split data
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=.2, shuffle=True, random_state=42)
train.head()
print(len(data))
print(len(train))
print(len(test))
# Predicting by using mode
np.round(train['player_kills'].mean())  # train['player_kills'].mean() = 0.887
test['prediction'] = 1.0
test.head()
# Analysing the prediction
from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(test['prediction'], test[dependent_variable]))
np.round(RMSE)  # RMSE = 1.616
# Importing machine learning library
from sklearn.linear_model import LinearRegression
# Creating machine learning model
model1 = LinearRegression()
# Training our model
model1.fit(train[independent_variable], train[dependent_variable])
# Get coeffecients
model1.coef_
# Get intercept
model1.intercept_
# Predicting on test data
prediction = model1.predict(test[independent_variable])
# Accuracy on training dataset
np.sqrt(mean_squared_error(model1.predict(train[independent_variable]), train[dependent_variable]))
# Accuracy on testing dataset
np.sqrt(mean_squared_error(model1.predict(test[independent_variable]), test[dependent_variable]))
# Importing Decision Tree Classifier
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
# Training our model
model2.fit(train[independent_variable], train[dependent_variable])
# Get Predictions
prediction = model2.predict(test[independent_variable])
# Accuracy on testing dataset
np.sqrt(mean_squared_error(prediction, test[dependent_variable]))
# create a Graphviz file
from sklearn.tree import export_graphviz
with open("tree1.dot", 'w') as f:
    f = export_graphviz(model2, out_file=f, feature_names=independent_variable)
    
#Convert .dot to .png to allow display in web notebook
#Please install graphviz before this conda install python-graphviz
#check_call(['dot','-Tpng','tree.dot','-o','tree.png'])

# Annotating chart with PIL
#img = Image.open("tree.png")
#img.save('sample-out.png')
#PImage("sample-out.png")
from sklearn.ensemble import GradientBoostingRegressor
model3 = GradientBoostingRegressor()
# Training our model
model3.fit(train[independent_variable], train[dependent_variable])
feat_importances = pd.Series(model3.feature_importances_, index=train[independent_variable].columns)
feat_importances.nsmallest(len(independent_variable)).plot(kind='barh')
# Get Predictions
prediction = model3.predict(test[independent_variable])
# Accuracy on testing dataset
np.sqrt(mean_squared_error(prediction, test[dependent_variable]))

