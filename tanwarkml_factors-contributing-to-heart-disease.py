# We first import important packages for our analysis

import pandas as pd # pandas and numpy are for data analysis and manipulation

import numpy as np

import matplotlib.pyplot as plt # matplotlib and seaborn allow for great visualization

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv') # make sure the dataset in the same directory as the notebook

df.head()
def get_shape(df, name = 'dataframe'):

    dims = len(df.shape)

    if dims == 1:

        print('The {} has {} rows and {} column.'.format(name, df.shape[0], 1))

    elif dims == 2:

        print('The {} has {} rows and {} columns.'.format(name, df.shape[0], df.shape[1]))

    

get_shape(df)
df.isnull().sum() # Number of NA values by column
old_names = df.columns # saving old column names which might be used later



df.columns = ['Age', 'Sex', 'Chest pain type', 'Resting blood pressure', 'Serum cholestrol',

             'Fasting blood sugar > 120 mg/dl', 'Resting ECG', 'Max heart rate', 'Exercise enduced angina',

             'Exercise enduced ST depression', 'Slope of ST', 'No. of major vessels', 'Thalassemia',

             'Diagnosis']



old_df = df.copy() # saving a copy of older dataframe



df['Sex'] = df['Sex'].map({0: 'Female', 1: 'Male'})



# Angina is chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood

df['Chest pain type'] = df['Chest pain type'].map({0: 'typical angina',

                                                  1: 'atypical angina',

                                                  2: 'non-anginal pain',

                                                  3: 'asymptomatic'})

df['Fasting blood sugar > 120 mg/dl'] = df['Fasting blood sugar > 120 mg/dl'].map({0: 'No',

                                                                                  1: 'Yes'})



# An electrocardiogram (ECG or EKG) records the electrical signal from your heart to check for different heart conditions.

# Electrodes are placed on your chest to record your heart's electrical signals, which cause your heart to beat.



df['Resting ECG'] = df['Resting ECG'].map({0: 'Normal',

                                          1: 'ST-T wave abnormality',

                                          2: 'Left ventricular hypertrophy'})



df['Exercise enduced angina'] = df['Exercise enduced angina'].map({0: 'No',

                                                                  1: 'Yes'})



df['Slope of ST'] = df['Slope of ST'].map({0: 'Up-sloping',

                                          1: 'Flat',

                                          2: 'Down-sloping'})



# The description on the website doesn't not provide the right mapping.

# So I've used generic mapping i.e. Type 1, 2, 3, and 4





# Thalassemia is a blood disorder which the body makes an abnormal form or inadequate amount of hemoglobin.



df['Thalassemia'] = df['Thalassemia'].map({0: 'Type 1',

                                          1: 'Type 2', 

                                          2: 'Type 3',

                                          3: 'Type 4'})  

df['Diagnosis'] = df['Diagnosis'].map({0: 'Negative',

                                      1: 'Positive'})

df.head()
from sklearn.model_selection import train_test_split



X = df.drop('Diagnosis', axis = 1)

y = df['Diagnosis']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 147,

                                                   shuffle = True, stratify = y)



get_shape(X_train, 'X_train datafram')

get_shape(X_test, 'X_test dataframe')

get_shape(y_train, 'y_train dataframe')

get_shape(y_test, 'y_test dataframe')
# We'll recombine are train data for visualization purpose. 

df_train = pd.concat([X_train, y_train], axis = 1)
# It's always good to check if predictor variable is distributed equally across different classes.

# It is often difficult to model with skewed classes.



plt.figure(figsize = (3, 4))

sns.countplot(x = df_train['Diagnosis'], color = "#FFCE00")

plt.ylabel('Counts');



# we'll use counts to set x position for text

counts = df_train['Diagnosis'].value_counts() 

# creating percentage text

percentage_text = ['{:0.0f}%'.format(x) for x in counts*100/df_train.shape[0]]

# setting each text using plt.text

for pos in range(len(percentage_text)):

    plt.text(pos-0.10, counts[pos]-10, percentage_text[pos],

            color = 'black')



# It seems like classes are almost equally split
figure, axes = plt.subplots(1, 2, figsize = (14, 5))



sns.boxplot(x = df['Resting blood pressure'], y = df['Diagnosis'], 

            color = '#6bc6fa', ax = axes[0])

axes[0].set_title('Resting Blood pressure by diagnosis')



sns.boxplot(x = df['Serum cholestrol'], y = df['Diagnosis'], 

            color = '#6bc6fa', ax = axes[1])

axes[1].set_title('Serum Cholestrol by diagnosis')

figure.tight_layout(pad = 5);
figure, axes = plt.subplots(1, 2, figsize = (14, 5))



sns.countplot(x = df_train['No. of major vessels'], hue = df_train['Diagnosis'], 

            palette = ['#FF0055', '#6bc6fa'], ax = axes[0])

axes[0].set_title('Diagnosis by major blood vessel')

axes[0].legend(loc = 'upper right')



sns.boxplot(y = df_train['Resting blood pressure'], x = df_train['No. of major vessels'], 

            color = '#6bc6fa', ax = axes[1])

axes[1].set_title('Resting blood pressure by diagnosis')

figure.tight_layout(pad = 5);
args = {'color': ['#FF0266', 'grey']}

scatters = sns.FacetGrid(row = 'Sex', col = 'Chest pain type', hue = 'Diagnosis', 

                         margin_titles = True, hue_kws = args, data = df_train)

scatters.map(plt.scatter, 'Age', 'Max heart rate');

scatters.add_legend()

plt.subplots_adjust(top=0.9) # padding for title

scatters.fig.suptitle("Heart Disease by Sex and Chest pain type");
figure, axes = plt.subplots(1, 3, figsize = (15, 3))



sns.distplot(df_train['Age'], ax = axes[0])

axes[0].set_title('Age distribution for all patients')

sns.distplot(df_train[df_train['Diagnosis'] == 'Positive']['Age'], ax = axes[1])

axes[1].set_title('Age distribution patients tested positive')

sns.distplot(df_train[df_train['Diagnosis'] == 'Negative']['Age'], ax = axes[2])

axes[2].set_title('Age distribution patients tested negative')

plt.setp(axes, ylim = (0, 0.07));
# Optional: Creating a custom diverging colormap

# I do not like diverging colormaps provided by matplotlib as none of them provide

# same color at the end of the spectrum. For correlation, it doesn't matter whether the values

# are positive or negative. We are interested in values at the end of the spectrum

# Hence, I'm creating a custom diverging colormap with same colors at the end.

# However, this is totally optional and you can use default colormaps.

from matplotlib import cm

from matplotlib.colors import ListedColormap



bupu = cm.get_cmap('BuPu', 128)



bupu_divergence = np.vstack((bupu(np.linspace(1, 0, 128)),

                       bupu(np.linspace(0, 1, 128))))



cust_cmap = ListedColormap(bupu_divergence, name='PurpleDiverging')



plt.figure(figsize = (10, 5))

sns.heatmap(old_df.corr(), annot = True, fmt = '0.2f', 

            vmin = -1, vmax = 1, cmap = cust_cmap);
figure, axes = plt.subplots(2, 2, figsize = (15, 11))

plt.subplots_adjust(hspace = 0.3)





sns.countplot(x = df_train['Diagnosis'], hue = df_train['Exercise enduced angina'], 

              palette = ['#FF0055', '#6bc6fa'], hue_order = ['Yes', 'No'], ax = axes[0,0])

axes[0,0].set_title("Diagnosis by exercise induced agina")



# Percentage labels

# diag_by_angina_labels = df.groupby(['Diagnosis', 'Exercise enduced angina'])['Age'].count()/df.shape[0]

# diag_by_angina_percentage_labels = ['{:0.0%}'.format(x) for x in diag_by_angina_labels]

# xticks = [-0.25, 0.15, 0.75, 1.15]

# for pos, xtick in zip(range(len(diag_by_angina_labels)), xticks):

#     axes[0,0].text(xtick, diag_by_angina_labels[pos], diag_by_angina_percentage_labels[pos] )





sns.countplot(x = df_train['Chest pain type'], hue = df_train['Exercise enduced angina'], 

              palette = ['#FF0055', '#6bc6fa'], hue_order = ['Yes', 'No'], ax = axes[0,1])

axes[0,1].set_title("Exercise induced agina by chest pain type")



sns.boxplot(x = df_train['Exercise enduced ST depression'], y = df_train['Diagnosis'], 

              color = '#6bc6fa', ax = axes[1,0])

axes[1,0].set_title("Diagnosis by exercise enduced ST depression")



sns.boxplot(x = df_train['Chest pain type'], y = df_train['Exercise enduced ST depression'],  

              color = '#6bc6fa', ax = axes[1,1])

axes[1,1].set_title("Exercise enduced ST depression by chest pain type");
# Create dummy variables

X_train = pd.get_dummies(X_train, drop_first=True)

X_test = pd.get_dummies(X_test, drop_first=True)



# Minmax scaling

from sklearn.preprocessing import MinMaxScaler



minmaxer = MinMaxScaler()

train_minmax = minmaxer.fit_transform(X_train)

test_minmax = minmaxer.transform(X_test) # I had to use fit again as train and test don't have the same size!
# Algorithms and model selection libraries



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import GridSearchCV
def base_model(models, X, y):

    for model_name, model in  models.items():

        crossval = cross_val_score(model, X, y, cv = 5, scoring = 'roc_auc')

        

        print("\n{} with Cross Validation \n".format(model_name),

              ['{:0.3%}'.format(x) for x in crossval], "\nMean Score \n",

              '{:0.3%}'.format(np.mean(crossval)))

        

seed = 30

models = {'Logistic': LogisticRegression(random_state = seed),

          'KNN': KNeighborsClassifier(n_neighbors=20),

          'Gaussian': GaussianNB(),

          'SVC': SVC(random_state = seed),

          'Decision Tree': DecisionTreeClassifier(random_state = seed),

          'Random Forest': RandomForestClassifier(random_state = seed),

          'Gradient Boosting': GradientBoostingClassifier(random_state = seed)   

         }



base_model(models, train_minmax, y_train)
rf_params = {'n_estimators': [10, 100, 300, 500, 1000],

            'min_samples_leaf': [1, 10, 25, 50]}



rf_grid = GridSearchCV(RandomForestClassifier(random_state=0), rf_params, cv = 7, scoring = 'roc_auc')

rf_grid.fit(train_minmax, y_train)
print("Best Validation Score: {:0.2%} ".format(rf_grid.best_score_))

print("Test Score: {:0.2%}".format(rf_grid.score(test_minmax, y_test)))

print("Best Parameters", rf_grid.best_params_)

print('\n\nChosen Model\n', rf_grid.best_estimator_)
# Visualizing scores across all parameters

scores = rf_grid.cv_results_['mean_test_score'].reshape(4, 5)

sns.heatmap(scores, cmap = 'BuPu', annot=True, fmt = '0.0%', 

           xticklabels = rf_params['n_estimators'], yticklabels=rf_params['min_samples_leaf']);



plt.xlabel('N Estimators')

plt.ylabel('Min Samples Leaf')

plt.yticks(rotation = 0);
%%time

rf_params = {'n_estimators': [10, 15, 20, 25, 50 , 75, 100],

            'min_samples_leaf': [1, 5, 10, 15, 20, 25, 30, 50]}



rf_finer = GridSearchCV(RandomForestClassifier(random_state = 0), rf_params, cv = 7, scoring = 'roc_auc')

rf_finer.fit(train_minmax, y_train)
print("Best Validation Score: {:0.2%} ".format(rf_finer.best_score_))

print("Test Score: {:0.2%}".format(rf_finer.score(test_minmax, y_test)))

print("\n\nBest Parameters", rf_finer.best_params_)

print('Chosen Model\n', rf_finer.best_estimator_)
# Visualizing scores across all parameters

scores = rf_finer.cv_results_['mean_test_score'].reshape(8, 7)

sns.heatmap(scores, cmap = 'BuPu', annot=True, fmt = '0.0%', 

           xticklabels = rf_params['n_estimators'], yticklabels=rf_params['min_samples_leaf']);



plt.xlabel('N Estimators')

plt.ylabel('Min Samples Leaf')

plt.yticks(rotation = 0);


plt.figure(figsize = (3, 4))

sns.countplot(x = y_test, color = "#FFCE00")

plt.ylabel('Counts');



# we'll use counts to set x position for text

total = y_test.shape[0]

test_counts = y_test.value_counts() 

# creating percentage text

test_percentage = ['{:0.0f}%'.format(x) for x in test_counts*100/total]



print('We have {} samples in our test set. {} ({}) samples are positive' \

      ' and {} ({}) samples are negative.'.format(total, test_counts[0], test_percentage[0], test_counts[1], test_percentage[1]))



# setting each text using plt.text

for pos in range(len(test_percentage)):

    plt.text(pos-0.10, test_counts[pos]-2.5, test_percentage[pos],

            color = 'black')



# It seems like classes are almost equally split
# from sklearn.metrics import confusion_matrix

# from sklearn.metrics import classification_report



from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve



rf_prec, rf_rec, rf_thresh = precision_recall_curve(y_test,

                                                    pos_label = 'Positive',

                                                    probas_pred = rf_finer.best_estimator_.predict_proba(test_minmax)[:,1])



optimal_idx = np.argmin(np.abs(rf_rec - rf_prec))



plt.plot(rf_prec, rf_rec, color = 'grey')

plt.plot(rf_prec[optimal_idx], rf_rec[optimal_idx], '^', 

         markersize  = 10, color = '#FF0055')



plt.legend(('Precision Recall Curve', 'Optimal Threshold'))

plt.xlabel('Precision')

plt.ylabel('Recall')

plt.xticks(np.arange(0.75, 1.01, 0.05))

plt.yticks(np.arange(0, 1.1, 0.1));
# ROC Curve

rf_tpr, rf_fpr, rf_thresh = roc_curve(y_test,

                                      pos_label = 'Positive',

                                      y_score = rf_finer.best_estimator_.predict_proba(test_minmax)[:, 1])



roc_optimal_idx = np.argmax(rf_fpr - rf_tpr)

# optimal_threshold = thresholds[optimal_idx]



plt.plot(rf_tpr, rf_fpr, color = 'grey')

plt.plot(rf_tpr[roc_optimal_idx], rf_fpr[roc_optimal_idx], marker = '^', color = '#FF0055')

plt.legend(('ROC Curve', 'Optimal Threshold'))

plt.xlabel("False Positive Rate")

plt.ylabel('True Positive Rate');
plt.figure(figsize = (10, 7))



features = X_train.columns

importance = rf_finer.best_estimator_.feature_importances_ 

indices = np.argsort(importance)



percentage_labels = ['{:0.0%}'.format(x) for x in importance[indices]]



plt.barh(y = range(len(indices)), color = "#FFCE00",

        width = importance[indices])

plt.yticks(range(len(indices)), features[indices])



for pos, index in zip(range(len(indices)), indices):

    plt.text(importance[index], pos , percentage_labels[pos])
# Base level predictions with loo

logistic_loo = cross_val_score(LogisticRegression(), train_minmax, y_train, cv = LeaveOneOut())

knn_loo = cross_val_score(KNeighborsClassifier(n_neighbors=20), X_train, y_train, cv = LeaveOneOut())

gaussian_loo = cross_val_score(GaussianNB(), train_minmax, y_train, cv = LeaveOneOut())

dtree_loo = cross_val_score(DecisionTreeClassifier(max_depth = 5), X_train, y_train, cv = LeaveOneOut())

svc_loo = cross_val_score(SVC(), train_minmax, y_train, cv = LeaveOneOut())

rf_loo = cross_val_score(RandomForestClassifier(max_depth = 5), X_train, y_train, cv = LeaveOneOut())

gb_loo = cross_val_score(GradientBoostingClassifier(max_depth = 5), X_train, y_train, cv = LeaveOneOut())



print("Mean Score of Logistic with LOO \n",

     '{:0.3%}'.format(np.mean(logistic_loo)))



print("\nMean Score of KNN with LOO \n",

     '{:0.3%}'.format(np.mean(knn_loo)))



print("\nMean Score of Naive Bayes with LOO \n",

     '{:0.3%}'.format(np.mean(gaussian_loo)))



print("\nMean Score of Decision Tree with LOO \n",

     '{:0.3%}'.format(np.mean(dtree_loo)))



print("\nMean Score of SVC with LOO \n",

     '{:0.3%}'.format(np.mean(svc_loo)))



print("\nMean Score of Random Forest with LOO \n",

     '{:0.3%}'.format(np.mean(rf_loo)))



print("\nMean Score of Gradient Boosting with LOO \n",

     '{:0.3%}'.format(np.mean(gb_loo)))