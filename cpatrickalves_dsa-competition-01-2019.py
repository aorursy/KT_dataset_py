# Importing packages

import numpy as np 

import pandas as pd 

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

%matplotlib inline



# Loading the data

data = pd.read_csv('../input/dataset_treino.csv')

test_data = pd.read_csv('../input/dataset_teste.csv')

data.head(5)
# General statistics

data.info()
# If the result is False, there is no missing value

data.isnull().values.any()
data.describe()
# Compute the number of occurrences of a zero value 

features = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

for c in features:

    counter = len(data[data[c] == 0])    

    print('{} - {}'.format(c, counter))    
# Removing observations with zero value

data_cleaned = data.copy()   

for c in ['glicose', 'pressao_sanguinea', 'bmi']:

    data_cleaned = data_cleaned[data_cleaned[c] != 0]



data_cleaned.shape
data_cleaned.describe()
fig, axes = plt.subplots(2,4, figsize=(20,8))



x,y = 0,0

for i, column in enumerate(data_cleaned.columns[1:-1]):    

    sns.boxplot(x=data_cleaned[column], ax=axes[x,y])

    if i < 3:

        y += 1

    elif i == 3: 

        x = 1

        y = 0

    else:

        y += 1
# Compute the Z-Score for each columns



print(data_cleaned.shape)



z = np.abs(stats.zscore(data_cleaned))    

data_cleaned = data_cleaned[(z < 3).all(axis=1)]   



print(data_cleaned.shape)    
fig, axes = plt.subplots(2,4, figsize=(20,8))



x,y = 0,0

for i, column in enumerate(data_cleaned.columns[1:-1]):    

    sns.boxplot(x=data_cleaned[column], ax=axes[x,y], palette="Set2")

    if i < 3:

        y += 1

    elif i == 3: 

        x = 1

        y = 0

    else:

        y += 1
data_cleaned.classe.value_counts().plot(kind='bar');
data_cleaned.classe.value_counts(normalize=True)
from imblearn.over_sampling import SMOTE



# Select the columns with features

features = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

X = data_cleaned[features]

# Select the columns with labels

Y = data_cleaned['classe']



smote = SMOTE(sampling_strategy=1.0, k_neighbors=4)

X_sm, y_sm = smote.fit_sample(X, Y)



print(X_sm.shape[0] - X.shape[0], 'new random picked points')

data_cleaned_oversampled = pd.DataFrame(X_sm, columns=data.columns[1:-1])

data_cleaned_oversampled['classe'] = y_sm

data_cleaned_oversampled['id'] = range(1,len(y_sm)+1)



for c in ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'idade']:

    data_cleaned_oversampled[c] = data_cleaned_oversampled[c].apply(lambda x: int(x))



data_cleaned_oversampled.classe.value_counts().plot(kind='bar');
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn import tree



# Select the columns with features

features = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

X = data_cleaned_oversampled[features]

# Select the columns with labels

Y = data_cleaned_oversampled['classe']



# Perform the training and test 100 times with different seeds and compute the mean accuracy.

# Save results

acurrances = []

for i in range(100):    

    # Spliting Dataset into Test and Train

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)



    # Create and train the model

    clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=i, max_depth=4)

    clf.fit(X_train,y_train)



    # Performing predictions with test dataset

    y_pred = clf.predict(X_test)

    # Computing accuracy    

    acurrances.append(accuracy_score(y_test, y_pred)*100)



print('Accuracy is ', np.mean(acurrances))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix



# Select the columns with features

features = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']



# For the LR the oversampled database decreased the model accuracy, so I choose do not use it.

X = data_cleaned[features]

# Select the columns with labels

Y = data_cleaned['classe']



# Perform the training and test 100 times with different seeds and compute the mean accuracy.

# Save results

acurrances = []

for i in range(100):    

    # Spliting the data

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

    LR_model=LogisticRegression(class_weight={1:1.15})

    LR_model.fit(X_train,y_train)



    # Testing

    y_pred=LR_model.predict(X_test)

    acurrances.append(accuracy_score(y_test, y_pred)*100)

    

    # Print only the last

    if i == 99:

        pass

        #print(classification_report(y_test,y_pred))

        #print(confusion_matrix(y_true=y_test, y_pred=y_pred))



print('Accuracy is ', np.mean(acurrances))
# Replacing the zeros by the mean

data_cleaned_no_zeros = data_cleaned.copy()



for c in ['grossura_pele', 'insulina']:    

    feature_avg =data_cleaned[data_cleaned[c]>0][[c]].mean()

    data_cleaned[c]=np.where(data_cleaned[[c]]!=0,data_cleaned[[c]],feature_avg)
# Select the columns with features

features = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

X = data_cleaned_no_zeros[features]

# Select the columns with labels

Y = data_cleaned_no_zeros['classe']



# Perform the training and test 100 times with different seeds and compute the mean accuracy score.

# Save results

acurrances = []

for i in range(100):    

    # Spliting the data

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)

    LR_model=LogisticRegression(class_weight={1:1.1})

    LR_model.fit(X_train,y_train)



    # Testing

    y_pred=LR_model.predict(X_test)

    acurrances.append(accuracy_score(y_test, y_pred)*100)

    

    # Print only the last

    if i == 99:

        print(classification_report(y_test,y_pred))

        print(confusion_matrix(y_true=y_test, y_pred=y_pred))



print('Accuracy is ', np.mean(acurrances))

# Create and train the model with all data

model=LogisticRegression(class_weight={1:1.1})

model.fit(data_cleaned[features],data_cleaned['classe'])



# Get the kaggle test data

X_test = test_data[features]

# Make the prediction 

prediction = model.predict(X_test)



# Add the predictions to the dataframe 

test_data['classe'] = prediction



# Create the submission file

test_data.loc[:,['id', 'classe']].to_csv('submission.csv', encoding='utf-8', index=False)