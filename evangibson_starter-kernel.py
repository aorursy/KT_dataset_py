# Import libraries

import seaborn as sns



#Set Seaborn Style



sns.set(style="whitegrid")



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Bring training and test data into environment

df_train = pd.read_csv("../input/proxymeanstest-cr/train.csv")

df_test = pd.read_csv("../input/proxymeanstest-cr/test.csv")
# Let's look at the shape of our dataframes to confirm the number of columns and rows

print(df_train.shape)

print(df_test.shape)
#Let's display the heads of both dataframes to make sure they imported correctly

pd.set_option('display.max_columns', 200)

df_train.head()
df_test.head()
# Summary Statistics

df_train.describe()
df_test.describe()
# In order to continue, we should understand how many unique households are in the data set vs. the number of individuals

print(df_train.idhogar.nunique())

print(df_test.idhogar.nunique())
# Creating smaller dataframe that removes missing values so that we can use seaborn

df_1 = df_train[pd.notnull(df_train['v2a1'])]



sns.distplot(df_1.v2a1)
#The outliers cause the scaling to function strangely. I'd like to have a clean look at what's going on at the lower incomes.

df_2 = df_train[df_train.v2a1 < 750000]



#Boxplot for income by target group with upper outliers removed

sns.boxplot(y="v2a1", x="Target", data=df_2)
escuela_1 = sns.countplot(x="escolari", data = df_train)
counts_tar = sns.countplot(x="Target", data = df_train)
# Import libraries

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier  

from sklearn import metrics

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



#We'll need this to ignore the warnings due to the df manipulation

import warnings

warnings.filterwarnings('ignore')

# Remove object type columns

df_train_1 = df_train.select_dtypes(exclude=['object'])



#These columns contain NA values

# In order to understand data collection issues, I should make sure I understand where presumably 

listo = df_train_1.columns[df_train_1.isna().any()].tolist()
# By imputing, we can regress data anomalies to the mean

# What does this "assumtpion" mean?



# It means that we assume missing values come from a place of collection shortfalls.

imp=SimpleImputer(missing_values=np.nan, strategy="mean")



for i in listo:

    imp.fit(df_train_1[[i]])

    df_train_1[[i]] = imp.fit_transform(df_train_1[[i]])

    

#Perform the same for the test data

for i in listo:

    imp.fit(df_test[[i]])

    df_test[[i]] = imp.fit_transform(df_test[[i]])
# divide into attributes and labels (sklearn syntax-ism)

X = df_train_1.drop('Target', axis=1)  

y = df_train_1['Target']



#80/20 split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# Fit our classifier using a random state to allow duplication of results

classifier = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=321)  

classifier.fit(X_train, y_train)
# Evaluating our model

y_pred = classifier.predict(X_test)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print("Macro F1 Score is ", f1_score(y_test,y_pred, average ='macro'))

print("Weighted F1 Score is ", f1_score(y_test,y_pred, average ='weighted'))



# Casting feature importances to a dataframe for easy sorting and 

feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': classifier.feature_importances_})

feature_importances.head()
# Defining a function so that we can visualize feature importances

# Why might feature importances interest us?

def plot_feature_importances(df, n = 10, threshold = None):



    plt.style.use('fivethirtyeight')

    

    # Sort features with most important at the head

    df = df.sort_values('importance', ascending = False).reset_index(drop = True)

    

    # Normalize the feature importances to add up to one and calculate cumulative importance

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    

    plt.rcParams['font.size'] = 12

    

    # Bar plot of n most important features

    df.loc[:n, :].plot.barh(y = 'importance_normalized', 

                            x = 'feature', color = 'darkgreen', 

                            edgecolor = 'k', figsize = (12, 8),

                            legend = False, linewidth = 2)



    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 

    plt.title(f'{n} Most Important Features', size = 18)

    plt.gca().invert_yaxis()

    

    

    if threshold:

        # Cumulative importance plot

        plt.figure(figsize = (8, 6))

        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')

        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 

        plt.title('Cumulative Feature Importance', size = 18);

        

        # Number of features needed for threshold cumulative importance

        # This is the index (will need to add 1 for the actual number)

        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        

        # Add vertical line to plot

        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')

        plt.show();

        

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 

                                                                                  100 * threshold))

    

    return df



norm_fi = plot_feature_importances(feature_importances, threshold=0.95)
#Precision, recall, and other metrics.

# What do all of these metrics mean?

# What do they tell us about the performance of the model?

print(metrics.classification_report(y_pred, y_test))
#Visualizing the confusion matrix for our model

mat = confusion_matrix(y_test, y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');
df_train_2 = df_train_1.drop(['Target'], axis=1)

features = df_train_2.columns
#Let's make some predictions



sample_pred = classifier.predict(df_test[features])



df = pd.DataFrame(sample_pred)

df_test['Target'] = df

df_final = df_test[['ID', 'Target']]

df_final.to_csv("/kaggle/working/predictions_starter.csv", index = False)

df_final.head()