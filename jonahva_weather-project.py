import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats

import matplotlib.ticker as plticker
import matplotlib.dates as mdates

from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['font.size'] = 10
%matplotlib inline
df = pd.read_csv('../input/weatherAUS.csv')
df.head()
# Looks like certain columns have lots of NANs in it.
df.info()
# Largely skewed data towards no rain tomorrow. Might cause a lot of false negatives.
df['RainTomorrow'].value_counts(normalize=True)
df.describe().T
# Converting Rain today/tomorrow to binary for classification and checking correlation. 
# Also changing date to a datetime to explore data over time.
df['RainTomorrow'] = df['RainTomorrow'].map(dict(Yes=1, No=0))
df['RainToday'] = df['RainToday'].map(dict(Yes=1, No=0))
df['Date'] = pd.to_datetime(df['Date'])
# Exploring the data to see if the large NANs are random or not.
sns.catplot(x="Evaporation", y="Location", kind="bar", data=df, height=8, aspect=1)
plt.title('Average Evaporation (mm) for each location')
plt.show()
# Looks like some of the locations dont input the data.
sns.catplot(x="Sunshine", y="Location", kind="bar", data=df, height=8, aspect=1)
plt.title('Average Sunshine hours for each location')
plt.show()
# Checking Cloud levels as they should be a range from 0-8
print('3pm cloud level max: ', df['Cloud3pm'].max())
print('9am cloud level max: ', df['Cloud9am'].max())
# Dropping these rows as the cloud level cant be above 8.
df = df.drop(df[(df['Cloud3pm'] > 8) | (df['Cloud9am'] > 8)].index)
# Going to drop the NAN rows instead of removing the columns as I think those are important features to keep.
# We have enough data to still make a good prediction. If it turns out to be not important, will go back and drop features.
# An option in future could be to get averages from the weather stations close to each other to fill the data.
# Downside of this method is that im getting rid of a lot of locations but I dont think that is too important.
df.dropna(inplace=True)
# Exploring the wind gust speed to see how different locations differ. Maybe useful but probably not.
# Grouping the wind speeds so I can order them in a readable way.
result = df.groupby(['Location'])['WindGustSpeed'].aggregate(np.average).reset_index().sort_values('WindGustSpeed').iloc[::-1]
sns.catplot(x="WindGustSpeed", y="Location", kind="bar", data=df, height=7, aspect=1, order=result['Location'])
plt.title('Average wind gust speed for locations')
plt.show()
# Exploring rainfall over time, probably not relavant to the end goal but it is interesting to look at.
timedf = pd.DataFrame(data = df[['Rainfall','Date']])
timedf = df.set_index('Date')

# Due to the large ammounts of data Will get a quarterly rolling mean with a small window.
rolling_mean = timedf['Rainfall'].resample('Q').sum().rolling(window=2, center=True).mean()

# Getting exponentially weighted mean to generate a smoother trend line.
exp_weighted_mean = timedf['Rainfall'].resample('Q').sum().ewm(span=10).mean()

ax = rolling_mean.plot(lw=2.5, figsize=(14,7), color='orange')
exp_weighted_mean.plot(ax=ax, lw=2.5, color='navy')

ax.set_ylabel('Rainfall')
ax.set_title('Rainfall rolling mean(o) vs Rainfall exp rolling mean(b) over time')
plt.show()
df['WindDir3pm'].value_counts().head()
df['WindDir9am'].value_counts().head()
# Creating dataframes for the value counts of wind direction.
df_9am = pd.DataFrame(df['WindDir9am'].value_counts())
df_3pm = pd.DataFrame(df['WindDir3pm'].value_counts())

# Setting same index to later merge.
df_3pm['direction'] = df_3pm.index
df_9am['direction'] = df_9am.index

# Reordering the directions.
reorderlist = ['N','NNW','NW','WNW',
               'W','WSW','SW','SSW',
               'S','SSE','SE','ESE',
               'E','ENE','NE','NNE']

df_9am = df_9am.reindex(reorderlist)

# Merging the two dataframes together
df_dir = df_9am.merge(df_3pm)

df_dir.head()
# Creating a radar chart to visualise the difference in wind directions based on time.

# Creating the angles for the circle.
angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
angles = np.concatenate((angles,[angles[0]]))

# Wind direction values reformatted to fit radar chart.
dir_am = df_dir['WindDir9am'].values
dir_am = np.concatenate((dir_am,[dir_am[0]]))

dir_pm = df_dir['WindDir3pm'].values
dir_pm = np.concatenate((dir_pm,[dir_pm[0]]))

# Creating labels for the chart.
labels = df_dir['direction'].values

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, polar=True)

# Sets the grid location to a circle.
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_theta_zero_location("N")

# Plots the am directions.
ax.plot(angles, dir_am, linewidth=2)
ax.fill(angles, dir_am, alpha=0.25)

# Plots the pm directions.
ax.plot(angles, dir_pm, linewidth=2)
ax.fill(angles, dir_pm, alpha=0.25)

ax.set_title('Wind dir 9am(b) vs 3pm(o)', fontsize=16)


# Setting the minimum value so the radar is more readable.
ax.set_rmin(1500)
ax.grid(True)
# Initial correlation map to check for important and unneeded  features.
fixed_corr = np.round(df.corr(), decimals = 2)

fix, ax = plt.subplots(figsize=(12,12))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(fixed_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(fixed_corr, mask=mask, ax=ax, annot=True, annot_kws={'size': 11})

# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=12)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=12)
ax.set_title('Correlation Heatmap')

plt.show()
# Using zscore to get rid of the few outliers.
print(df.shape)
z = np.abs(stats.zscore(df._get_numeric_data()))
df = df[(z < 3).all(axis=1)]
print(df.shape)
def plotdate(feature, *size):
    # Creates a date plot with the given feature.
    # Optional parameter if I want to make a bigger graph.
    if 'big' in size:
        fig, ax = plt.subplots(figsize=(15,10))
    else:
        fig, ax = plt.subplots(figsize=(12,7))
    
    plt.title("{} over time".format(feature))
    ax.grid(True, which='both', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('{}'.format(feature))
    plt.plot_date(df['Date'], df[feature],  alpha=0.8)
    
# Plotting to get a deeper understanding of what seems to be the best features so far.
plotdate('Sunshine')
plotdate('Evaporation')
plotdate('Rainfall')
# Going to add difference columns for each time column. e.g. (humidity9am - humidity3pm = difference)
df_diff = df.copy()

# Grabs each column that has 3pm in it and returns the column name without the 3pm so it can also get the 9am column.
cols = [c[:-3] for c in df.columns if '3pm' in c]

# remove the wind direction as its a direction and cant really get the difference.
cols.remove('WindDir')

#Loops through each column and creates the difference.
for col in cols:
    df_diff['{}Diff'.format(col)] =  df['{}3pm'.format(col)] - df['{}9am'.format(col)]

def remover(dfname, cols, parlist):
    # Removes the columns from the dataframe based on a list of parameters. 
    for c in cols:
        for p in parlist:
            if p in c:
                dfname.drop(columns= c, inplace=True)
remover(df_diff, df_diff.columns, ['3pm', '9am'])
fixed_corr = np.round(df_diff.corr(), decimals = 2)

fix, ax = plt.subplots(figsize=(12,12))

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(fixed_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Plot the heatmap with seaborn.
# Assign the matplotlib axis the function returns. This will let us resize the labels.
ax = sns.heatmap(fixed_corr, mask=mask, ax=ax, annot=True, annot_kws={'size': 12})

# Resize the labels.
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=14)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=14)

ax.set_title('Correlation Heatmap')
plt.show()
# Create tempdiff and remove all 9am columns from main dataframe
df['TempDiff'] = df['Temp3pm'] - df['Temp9am']
remover(df,df.columns,['9am'])
# Dropping RISK_MM as to not leak information.
# Min and max temp are similar to the other temp columns and I have created the temp diff which I believe will work well.
# Wind/ wind gust direction I don't believe will help due to rain not coming from a certain direction.
# Date I might later add in as a monthly or seasonal column.
# Location probably won't lead to much findings with rain being common most areas.
cols = ['RISK_MM', 'MinTemp', 'MaxTemp','WindGustDir','Date', 'Location','WindDir3pm']
df.drop(columns=cols, inplace=True)
def kdeplot(feature, feature_format):
    # Function to create a KDE plot.
    # To visualise the difference in distribution of data for rain and not rain.
    # Given option to set the format so can plot with RainToday and RainTomorrow.
    plt.figure(figsize=(9, 4))
    plt.title("KDE for {}".format(feature))
    plt.ylabel('Probability Density')
    plt.xlabel('{} level'.format(feature))
    
    ax0 = sns.kdeplot(df[df[feature_format] == 0][feature].dropna(), color= 'orange', label= '{}: No'.format(feature_format))
    ax1 = sns.kdeplot(df[df[feature_format] == 1][feature].dropna(), color= 'navy', label= '{}: Yes'.format(feature_format))
    
kdeplot('Sunshine','RainToday')
kdeplot('Sunshine','RainTomorrow')
kdeplot('Humidity3pm','RainToday')
kdeplot('Humidity3pm','RainTomorrow')
kdeplot('TempDiff','RainToday')
kdeplot('TempDiff','RainTomorrow')
# Assign dataframe to X
X = df.copy()

# Rain Tomorrow is what we want to predict so assigned to Y.
y = df['RainTomorrow'].values

# Dropping RainTomorrow from X to not leak answers.
X.drop(columns='RainTomorrow', inplace=True)

# Standardize X.
ss = StandardScaler()
Xs = ss.fit_transform(X)

# Split the data into train and test.
X_train, X_test, y_train, y_test = train_test_split(Xs, y, random_state=1)
# Adding a class weight as I want the model to better predict rainy days.
# Choosing to just use random forest to better understand the model.
rfc = RandomForestClassifier(max_depth = 15, n_estimators = 99, class_weight='balanced')
rfc.fit(X_train,y_train)
print('Training Accuracy:',round(rfc.score(X_train,y_train),3))
print('Test Accuracy:',round(rfc.score(X_test,y_test),3))
rfc_params = {
    'max_depth':range(12,16),
    'n_estimators':range(99,105)
}

rfc_gd = GridSearchCV(rfc,rfc_params,cv=5, verbose=True, n_jobs=-1)
rfc_gd.fit(X_train,y_train)

print(rfc_gd.best_params_)
print(rfc_gd.score(X_train,y_train))
print(rfc_gd.score(X_test,y_test))
rfc = rfc_gd.best_estimator_
rfc.fit(X_train,y_train)
print('Test Accuracy:',round(rfc.score(X_test,y_test),3))
df_cm = pd.DataFrame(confusion_matrix(y_test, rfc.predict(X_test)),
                     index=['no rain', 'rain'], columns=['no rain', 'rain'])
plt.figure(figsize=(10,7))
sns.heatmap(df_cm, annot=True,fmt='g', cmap='Blues')

plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test, rfc.predict(X_test), target_names=['no rain', 'rain']))
# Get the prediction probabilities to compare to test data.
y_pred_prob = rfc.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# # Create Plot.
plt.figure(figsize=(10,7))
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.legend(loc="lower right")
plt.show()
feature_importances = pd.DataFrame(rfc.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
sns.barplot(x=feature_importances['importance'],y=feature_importances.index)
plt.title('random forest important features ranked')
plt.show()
