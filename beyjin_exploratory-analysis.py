import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt
heart_disease = pd.read_csv('../input/heart.csv')
heart_disease.head()
heart_disease.info()
heart_disease.describe()
heart_disease.rename(columns = {

    'cp' : 'chest_pain_type',

    'trestbps' : 'resting_blood_pressure',

    'chol' : 'cholesteral',

    'fbs' : 'fasting_blood_sugar_higher_120',

    'restecg' : 'resting_cardiographic_results',

    'thalach' : 'max_heartrate',

    'exang' : 'induced_angina',

    'ca' : 'num_maj_vessels_flourosopy',

    'thal' : 'blood_characterization'

}, inplace = True)
print(plt.hist(heart_disease['age'], histtype = 'step'))
def define_cluster_groups(dataset, column, interval_indicator = 5):

    new_column = column + '_cluster'

    # create cluster

    cluster = 0

    dataset[new_column] = 0

    while(cluster * interval_indicator < max(dataset[column])):

        cluster_value = cluster * interval_indicator + 1 # +1 because else we would overwrite the last age value per cluster

        min_cluster_value = min(dataset[column]) + cluster_value

        max_cluster_value = max(dataset[column]) + cluster_value

        if cluster == 0:

             dataset.at[(dataset[column] >= (min_cluster_value - 1)) & 

                        (dataset[column] <= (max_cluster_value - 1)),

                       new_column] = cluster

        else:

            dataset.at[(dataset[column] >= min_cluster_value) & 

                       (dataset[column] <= max_cluster_value),

                       new_column] = cluster

        cluster += 1
## create age_cluster

define_cluster_groups(heart_disease, 'age', interval_indicator = 5)
print(heart_disease['age_cluster'].value_counts())
### check the first rows for validation of our function

heart_disease[['age', 'age_cluster']].head(10)
heart_disease.groupby(['age_cluster', 'target'])['age'].count().reset_index()
sns.barplot(x = 'age_cluster', y = 'age', hue = 'target', data = heart_disease.groupby(['age_cluster', 'target'])['age'].count().reset_index())

## since I´m doing a count --- 'age' can be replaced with any other column
heart_disease = heart_disease.loc[(heart_disease['age_cluster'] > 0) &

                                  (heart_disease['age_cluster'] < 8), :].reset_index().drop('index', axis = 1)
plt.hist(heart_disease['max_heartrate'], histtype = 'step')
## create heart rate cluster

define_cluster_groups(heart_disease, 'max_heartrate', interval_indicator = 13)

# for clarification: I´m using the interval_indicator of the step hist -- you can see the values in the second printed out array 
sns.barplot(x = 'max_heartrate_cluster', y = 'age', hue = 'target', data = heart_disease.groupby(['max_heartrate_cluster', 'target'])['age'].count().reset_index())

sns.jointplot(x = 'age_cluster', y = 'max_heartrate_cluster', data = heart_disease.loc[heart_disease['target'] == 1])
heart_disease.info()
sns.jointplot(x = 'resting_blood_pressure', y = 'max_heartrate', data = heart_disease[heart_disease['target']==1], kind = 'kde')
sns.pairplot(heart_disease[['age', 'resting_blood_pressure', 'cholesteral', 'max_heartrate', 'oldpeak', 'target']], hue = 'target')
heart_disease[['age', 'resting_blood_pressure', 'cholesteral', 'max_heartrate', 'oldpeak']].corr()
heart_disease.loc[heart_disease['target'] == 1, ['age', 'resting_blood_pressure', 'cholesteral', 'max_heartrate', 'oldpeak']].corr()
heart_disease.loc[heart_disease['target'] == 0, ['age', 'resting_blood_pressure', 'cholesteral', 'max_heartrate', 'oldpeak']].corr()
sns.jointplot(x = 'resting_blood_pressure', y = 'cholesteral', data = heart_disease[heart_disease['target'] == 1], kind = 'kde')
sex  = sns.FacetGrid(heart_disease, col="sex", hue="target")

sex.map(plt.scatter, "cholesteral", "resting_blood_pressure", alpha=.7)

sex.add_legend();
sex  = sns.FacetGrid(heart_disease, col="sex", hue="target")

sex.map(plt.scatter, "max_heartrate", "resting_blood_pressure", alpha=.7)

sex.add_legend();