%matplotlib notebook



import time



# Linear Algebra

import numpy as np



# Data Processing

import pandas as pd



# Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud



# Stats

from scipy import stats



# Algorithms

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_absolute_error, classification_report, roc_auc_score, roc_curve

from sklearn.preprocessing import LabelEncoder



# Classifiers

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



# String matching

import difflib



# Set random seed for reproducibility

np.random.seed(0)



# Stop unnecessary Seaborn warnings

import warnings

warnings.filterwarnings('ignore')

sns.set()  # Stylises graphs
df = pd.read_csv('../input/advertising/advertising.csv')
df.head()
df.info()
df[df.columns].isnull().sum() * 100 / df.shape[0]
df.duplicated().sum()
qual_cols = set(df.select_dtypes(include = ['object']).columns)

print(f'Qualitative Variables: {qual_cols}')
qual_cols = qual_cols - {'Timestamp'}

print(f'Qualitative Variables: {qual_cols}')
df[qual_cols].describe()
quant_cols = set(df.columns) - set(qual_cols)

print(f'Quantitative Variables: {quant_cols}')
df[quant_cols].describe()
pd.crosstab(df['Country'], columns='count').sort_values('count', ascending=False).head(10)
pd.crosstab(df['Country'], df['Clicked on Ad']).sort_values(1, ascending=False).head(10)
# Convert Timestamp to a more appropreate data type

df['Timestamp'] = pd.to_datetime(df['Timestamp'])



df['Month'] = df['Timestamp'].dt.month

df['Day'] = df['Timestamp'].dt.day

df['Hour'] = df['Timestamp'].dt.hour

df['Weekday'] = df['Timestamp'].dt.dayofweek
region_df = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')
region_df.head()
region_df = region_df[['name', 'region', 'sub-region']]
def label_region(row):

    # Match all rows that directly equal one another

    matches = region_df[region_df['name'] == row['Country']]



    if matches.empty:

        # If no matches, check if all csv countries contain the country from a given row 

        matches = region_df[[row['Country'] in country for country in region_df['name']]]

    

    if matches.empty:

        # If still no matches, check if all csv countries contain the first word of a country from a given row 

        matches = region_df[

            [

                row['Country'].split(' ')[0] in country

                for country in region_df['name']

            ]

        ]

        

        if len(matches) > 1:

            # If there was more than one match, we're not intrested

            matches = pd.DataFrame()

    

    if matches.empty:

         # If still no matches, fuzzyily get matches

        matches = difflib.get_close_matches(row['Country'], region_df['name'], cutoff=0.8)

        

        if matches:

            return region_df[region_df['name'] == matches[0]][['region', 'sub-region']].iloc[0]

        else:

            matches = pd.DataFrame()



    if not matches.empty:

        return matches[['region', 'sub-region']].iloc[0]

    else:

        return [np.nan, np.nan] 



df[['region', 'sub-region']] = df.apply(label_region, axis=1)
df[df.isna().any(axis=1)]
df = df.dropna()

df.head()
plt.figure(figsize=(20, 10))

sns.countplot(x='Clicked on Ad', data=df)

plt.title("The Number of People that Clicked on Ads")

plt.xlabel("Clicked on Ad")

plt.xticks([0, 1], ('False', 'True'))

plt.ylabel("Count")

plt.show()
plt.figure(figsize=(10, 10))

sns.pairplot(

    df,

    hue ='Clicked on Ad',

    vars=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'],

    diag_kind='kde',

    palette='bright'

)

plt.show()
analysis_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']



for col in analysis_cols:

    vars_ = []



    for clicked in [0, 1]:

        var = np.var(

            df[df['Clicked on Ad'] == clicked][col],

            ddof=1

        )

        vars_.append(var)



        if clicked:

            print(f'Sample variance for {col} of clicked: {var}')

        else:

            print(f'Sample variance for {col} of non clicked: {var}')

            

        

    print(f'Differences in Variance: {round(abs(vars_[0] - vars_[1]), 2)}\n')
for col in analysis_cols:

    means_ = []



    for clicked in [0, 1]:

        mean = np.mean(

            df[df['Clicked on Ad'] == clicked][col]

        )

        means_.append(mean)



        if clicked:

            print(f'Mean for {col} of clicked: {mean}')

        else:

            print(f'Mean for {col} of non clicked: {mean}')

            

        

    print(f'Differences in Mean: {round(abs(means_[0] - means_[1]), 2)}\n')
alpha = 0.05



for col in analysis_cols:



    for clicked in [0, 1]:

        k2, p = stats.normaltest(df[df['Clicked on Ad'] == clicked][col])



        if clicked:

            print(f'Results for clicked {col}:')

        else:

            print(f'Results for nonclicked {col}:')



        print(f'\tStatistic: {k2}')

        print(f'\tpvalue: {p}')

        

        if p < alpha:

            print('The null hypothesis can be rejected.\n')

        else:

            print('The null hypothesis cannot be rejected.\n')
alpha = 0.05



for col in analysis_cols:

        clicked = df[df['Clicked on Ad'] == 1][col]

        non_clicked = df[df['Clicked on Ad'] == 0][col]



        w, p = stats.mannwhitneyu(x=clicked, y=non_clicked, alternative='two-sided')

        

        print(f'Results for {col}: ')

        print(f'\tStatistic: {w}')

        print(f'\tpvalue: {p}')

        

        if p < alpha:

            print('The null hypothesis can be rejected.\n')

        else:

            print('The null hypothesis cannot be rejected.\n')
# Computer correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(20, 20))



# Generate a custom diverging colourmap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(

    corr, mask=mask, cmap=cmap, vmax=.3, center=0,

    square=True, linewidths=.5, cbar_kws={"shrink": .5},

    annot=True

)



ax.set_title('Correlation Heatmap of the Variables')



plt.show()
# SET DATA 



month_counts = pd.crosstab(df["Clicked on Ad"], df["Month"])



# for i in range(1, 12):

#     if i not in month_counts:

#         month_counts[i] = 0

        

# CREATE BACKGROUND

months = [

    'Jan', 'Feb', 'Mar', 'Apr',

    'May', 'Jun', 'Jul'

]



# Angle of each axis in the plot

angles = [(n / 7) * 2 * np.pi for n in range(8)]  # Seven months in data



subplot_kw = {

    'polar': True

}



fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=subplot_kw)

ax.set_theta_offset(np.pi / 2)

ax.set_theta_direction(-1)

ax.set_rlabel_position(0)



plt.xticks(angles[:-1], months)

plt.yticks(color="grey", size=7)



# ADD PLOTS



# PLOT 1

month_counts_nonclicked = month_counts.iloc[0].tolist()

month_counts_nonclicked += month_counts_nonclicked[:1]  # Properly loops the circle back



ax.plot(angles, month_counts_nonclicked, linewidth=1, linestyle='solid', label="Didn't Click Ad")

ax.fill(angles,  month_counts_nonclicked, alpha=0.1)



# PLOT 2

month_counts_clicked = month_counts.iloc[1].tolist()

month_counts_clicked += month_counts_clicked[:1]  # Properly loops the circle back



ax.plot(angles, month_counts_clicked, linewidth=1, linestyle='solid', label="Clicked Ad")

ax.fill(angles,  month_counts_clicked, 'orange', alpha=0.1)



plt.title("Counts of People that Click and Didn't Click Ads for Each Month")

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))



plt.show()
# SET DATA 



weekday_counts = pd.crosstab(df["Clicked on Ad"], df["Weekday"])



# CREATE BACKGROUND

weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']



# Angle of each axis in the plot

angles = [(n / 7) * 2 * np.pi for n in range(8)]  # Seven months in data



subplot_kw = {

    'polar': True

}



fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=subplot_kw)

ax.set_theta_offset(np.pi / 2)

ax.set_theta_direction(-1)

ax.set_rlabel_position(0)



plt.xticks(angles[:-1], weekdays)

plt.yticks(color="grey", size=7)



# ADD PLOTS



# PLOT 1

weekday_counts_nonclicked = weekday_counts.iloc[0].tolist()

weekday_counts_nonclicked += weekday_counts_nonclicked[:1]  # Properly loops the circle back



ax.plot(angles, weekday_counts_nonclicked, linewidth=1, linestyle='solid', label="Didn't Click Ad")

ax.fill(angles,  weekday_counts_nonclicked, alpha=0.1)



# PLOT 2

weekday_counts_clicked = weekday_counts.iloc[1].tolist()

weekday_counts_clicked += weekday_counts_clicked[:1]  # Properly loops the circle back



ax.plot(angles, weekday_counts_clicked, linewidth=1, linestyle='solid', label="Clicked Ad")

ax.fill(angles,  weekday_counts_clicked, 'orange', alpha=0.1)



plt.title("Counts of People that Click and Didn't Click Ads for Each Weekday")

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))



plt.show()
# SET DATA 



hour_counts = pd.crosstab(df["Clicked on Ad"], df["Hour"])



# CREATE BACKGROUND



# Angle of each axis in the plot

angles = [(n / 24) * 2 * np.pi for n in range(25)]  # Seven months in data



subplot_kw = {

    'polar': True

}



fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=subplot_kw)

ax.set_theta_offset(np.pi / 2)

ax.set_theta_direction(-1)

ax.set_rlabel_position(0)



plt.xticks(angles[:-1], list(range(25)))

plt.yticks(color="grey", size=7)



# ADD PLOTS



# PLOT 1

hour_counts_nonclicked = hour_counts.iloc[0].tolist()

hour_counts_nonclicked += hour_counts_nonclicked[:1]  # Properly loops the circle back



ax.plot(angles, hour_counts_nonclicked, linewidth=1, linestyle='solid', label="Didn't Click Ad")

ax.fill(angles,  hour_counts_nonclicked, alpha=0.1)



# PLOT 2

hour_counts_clicked = hour_counts.iloc[1].tolist()

hour_counts_clicked += hour_counts_clicked[:1]  # Properly loops the circle back



ax.plot(angles, hour_counts_clicked, linewidth=1, linestyle='solid', label="Clicked Ad")

ax.fill(angles,  hour_counts_clicked, 'orange', alpha=0.1)



plt.title("Counts of People that Click and Didn't Click Ads for Each Hour")

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))



plt.show()
text = ' '.join(topic_line for topic_line in df['Ad Topic Line'])

world_cloud = WordCloud(width=1000, height=1000).generate(text)



plt.figure(figsize=(20, 10))

plt.imshow(world_cloud, interpolation='bilinear')

plt.axis('off')

plt.tight_layout()

plt.show()
exten_qual_cols = [

    'Daily Time Spent on Site', 'Age',

    'Area Income', 'Daily Internet Usage'

]



outliers_df = pd.DataFrame(columns=df.columns)



for col in exten_qual_cols:

    stat = df[col].describe()

    print(stat)

    IQR = stat['75%'] - stat['25%']

    upper = stat['75%'] + 1.5 * IQR

    lower = stat['25%'] - 1.5 * IQR

    

    outliers = df[(df[col] > upper) | (df[col] < lower)]



    if not outliers.empty:

        print(f'\nOutlier found in: {col}')

        outliers_df = pd.concat([outliers_df, outliers])

    else:

        print(f'\nNo outlier found in: {col}')



    print(f'\nSuspected Outliers Lower Bound: {lower}')

    print(f'Suspected Outliers Upper Bound: {upper}\n\n')



print(f'Number of outlier rows: {len(outliers_df)}')



del outliers
outliers_df.head(10)
X = df.copy()



drop_cols = ['Ad Topic Line', 'City', 'Timestamp']



for col in drop_cols:

    X.drop([col], axis=1, inplace=True)
encode_cols = ['Country', 'region', 'sub-region']

le = LabelEncoder()



for col in encode_cols:

    X[col] = le.fit_transform(X[col])
y = X['Clicked on Ad']

X.drop(['Clicked on Ad'], axis=1, inplace=True)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)



print(f'X Train Shape: {X_train.shape}')

print(f'X Validation Shape: {X_valid.shape}')

print(f'y Train Shape: {y_train.shape}')

print(f'y Validation Shape: {y_valid.shape}')
scores = {}



for n_estimators in range(2, 100):

    RF_model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

    RF_model.fit(X_train, y_train)

    RF_predictions = RF_model.predict(X_valid)

    RF_mae = mean_absolute_error(RF_predictions, y_valid)

    scores[n_estimators] = RF_mae
plt.figure(figsize=(10, 4))

plt.title("Mean Absolute Error with Number of Estimators of a Random Forest")

plt.xlabel("Number of Estimators")

plt.ylabel("Mean Absolute Error")

plt.plot(scores.keys(), scores.values())

plt.show()
best_n_estimators = []



for n_estimators, score in scores.items():

    if score == min(scores.values()):

        best_n_estimators.append(n_estimators)



print(f"Best Number of Estimators: {min(best_n_estimators)}")
rf_clf = RandomForestClassifier(n_estimators=min(best_n_estimators), random_state=0)



rf_time = time.time()

rf_clf.fit(X_train, y_train)

rf_time = time.time() - rf_time



rf_auc = roc_auc_score(y_valid, rf_clf.predict(X_valid))



score_train = rf_clf.score(X_train, y_train)

print('Training Accuracy : ' + str(score_train))



score_valid = rf_clf.score(X_valid, y_valid)

print('Validation Accuracy : ' + str(score_valid))



print()

print(f'AUC: {rf_auc}')

print(f'Time Elapsed: {rf_time} seconds')

print(classification_report(y_valid, rf_clf.predict(X_valid)))
svc_lin_scores = {}

c = np.linspace(0.0069, 0.0072, 10)



for C in c:

    svc_lin_clf = SVC(random_state=0, kernel='linear', C=C)

    svc_lin_clf.fit(X_train, y_train)

    svc_lin_scores[C] = svc_lin_clf.score(X_train, y_train)
plt.figure(figsize=(20, 5))

plt.title("Precision of Linear SVC With Penalty Parameter C")

plt.ylabel("Precision")

plt.xlabel("C")

plt.plot(svc_lin_scores.keys(), svc_lin_scores.values())

plt.show()
svc_lin_clf = SVC(random_state=0, kernel='linear', C=0.007, probability=True)



svc_lin_time = time.time()

svc_lin_clf.fit(X_train, y_train)

svc_lin_time = time.time() - svc_lin_time



svc_lin_auc = roc_auc_score(y_valid, svc_lin_clf.predict(X_valid))



score_train = svc_lin_clf.score(X_train, y_train)

print('Training Accuracy : ' + str(score_train))



score_valid = svc_lin_clf.score(X_valid, y_valid)

print('Validation Accuracy : ' + str(score_valid))



print()

print(f'AUC: {svc_lin_auc}')

print(f'Time Elapsed: {svc_lin_time} seconds')

print(classification_report(y_valid, svc_lin_clf.predict(X_valid)))
knn_scores = {}



for k in range(1, 30):

    knn_clf = KNeighborsClassifier(k)

    knn_clf.fit(X_train, y_train)

    knn_scores[k] = knn_clf.score(X_train, y_train)
plt.figure(figsize=(20, 5))

plt.title("Precision of k Nearest Neighbors Classifier With k Nearest Neighbors")

plt.ylabel("Precision")

plt.xlabel("k Nearest Neighbors")

plt.plot(knn_scores.keys(), knn_scores.values())

plt.show()
knn_clf = KNeighborsClassifier(3)



knn_time = time.time()

knn_clf.fit(X_train, y_train)

knn_time = time.time() - knn_time



knn_auc = roc_auc_score(y_valid, knn_clf.predict(X_valid))



score_train = knn_clf.score(X_train, y_train)

print('Training Accuracy : ' + str(score_train))



score_valid = knn_clf.score(X_valid, y_valid)

print('Validation Accuracy : ' + str(score_valid))



print()

print(f'AUC: {knn_auc}')

print(f'Time Elapsed: {knn_time} seconds')

print(classification_report(y_valid, knn_clf.predict(X_valid)))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_valid, rf_clf.predict_proba(X_valid)[:,1])

svc_lin_fpr, svc_lin_tpr, svc_lin_thresholds = roc_curve(y_valid, svc_lin_clf.predict_proba(X_valid)[:,1])

knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_valid, knn_clf.predict_proba(X_valid)[:,1])



plt.figure(figsize=(10, 10))



# Plot Random Forest ROC

plt.plot(rf_fpr, rf_tpr, label=f'Random Forest AUC: {round(rf_auc, 3)}')

plt.plot(svc_lin_fpr, svc_lin_tpr, label=f'Linear Support Vector Classifier: {round(svc_lin_auc, 3)}')

plt.plot(knn_fpr, knn_tpr, label=f'k-Nearest Neighbors: {round(knn_auc, 3)}')





# Plot Base Rate ROC

plt.plot([0,1], [0,1],label='Base Rate')



plt.xlim([-0.005, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Graph')

plt.legend(loc="lower right")

plt.show()
def pred_time(model, data):

    times = []

    

    for _ in range(1000):

        pred_time = time.time()

        model.predict(data)

        times.append(time.time() - pred_time)



    return np.mean(times)



models = {

    'Random Forest': (rf_clf, rf_time, rf_auc),

    'Linear Support Vector Classifier': (svc_lin_clf, svc_lin_time, svc_lin_auc),

    'k Nearest Neighbors': (knn_clf, knn_time, knn_auc)

}



for name, model in models.items():

    print(f'Model: {name}')

    print(f'Model Fitting Time: {round(model[1], 4)} seconds')

    print(f'Prediction Time: {round(pred_time(model[0], X_valid), 4)} seconds')

    print(f'Model AUC: {round(model[2], 4)}\n\n')
columns = X.columns

train = pd.DataFrame(np.atleast_2d(X_train), columns=columns)
feature_importances = pd.DataFrame(rf_clf.feature_importances_,

                                   index = train.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances = feature_importances.reset_index()

feature_importances.head(10)
plt.figure(figsize=(13, 7))

sns.barplot(

    x="importance", y='index',

    data=feature_importances[0:10], label="Total"

)

plt.title("Random Forest Variable Importance")

plt.ylabel("Variable")

plt.xlabel("Importance")

plt.show()