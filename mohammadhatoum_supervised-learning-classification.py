import warnings

warnings.filterwarnings("ignore")



# Load the diabetes dataset

import pandas as pd

train_df = pd.read_csv("../input/titanic/train.csv")

test_df = pd.read_csv("../input/titanic/test.csv")
print(train_df.columns.values)
print(test_df.columns.values)
train_df.head()
test_df.head()
train_df.info()
test_df.info()
train_df.describe(include=['O'])
test_df.describe(include=['O'])
train_df.columns[train_df.isnull().any()]
import string

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if str.find(big_string, substring) != -1:

            return substring

    return np.nan



title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                'Don', 'Jonkheer']

train_df['Title']=train_df['Name'].map(lambda x: substrings_in_string(x, title_list))

test_df['Title']=test_df['Name'].map(lambda x: substrings_in_string(x, title_list))



#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

    

train_df['Title']=train_df.apply(replace_titles, axis=1)

test_df['Title']=test_df.apply(replace_titles, axis=1)





#Drop the columns 'Name', 'PassengerId' and 'Ticket'

train_df = train_df.drop(['Name','PassengerId','Ticket'],axis=1)

test_df = test_df.drop(['Name','PassengerId','Ticket'],axis=1)
train_df['Family_Size']=train_df['SibSp']+train_df['Parch']

test_df['Family_Size']=test_df['SibSp']+test_df['Parch']
import numpy as np

from scipy.stats import mode



for df in [train_df, test_df]:

    

    meanAge=np.mean(df.Age)

    df.Age=df.Age.fillna(meanAge)

    bins = (-1, 0,  50, 100)

    group_names = ['Unknown', 'Under_50', 'More_Than_50']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    

    modeEmbarked = mode(df.Embarked)[0][0]

    df.Embarked = df.Embarked.fillna(modeEmbarked)

    

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories
# Extract the training and test data

y = train_df['Survived']

X = train_df.drop('Survived',axis=1)



from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size= 0.2, random_state=0)



# View the shape (structure) of the data

print(f"Training features shape: {X_train.shape}")

print(f"Testing features shape: {X_val.shape}")

print(f"Training label shape: {y_train.shape}")

print(f"Testing label shape: {y_val.shape}")
train_df.describe(include=['O'])
# Print top 10 records before transformation

X_train[0:10]


from sklearn.preprocessing import OrdinalEncoder

encoder_sex = OrdinalEncoder()

X_train['Sex'] = encoder_sex.fit_transform(X_train['Sex'].values.reshape(-1, 1))

X_val['Sex'] = encoder_sex.transform(X_val['Sex'].values.reshape(-1, 1))



encoder_cabin = OrdinalEncoder()

X_train['Cabin'] = encoder_cabin.fit_transform(X_train['Cabin'].values.reshape(-1, 1))

X_val['Cabin'] = encoder_cabin.transform(X_val['Cabin'].values.reshape(-1, 1))





encoder_embarked = OrdinalEncoder()

X_train['Embarked'] = encoder_embarked.fit_transform(X_train['Embarked'].values.reshape(-1, 1))

X_val['Embarked'] = encoder_embarked.transform(X_val['Embarked'].values.reshape(-1, 1))



encoder_title = OrdinalEncoder()

X_train['Title'] = encoder_title.fit_transform(X_train['Title'].values.reshape(-1, 1))

X_val['Title'] = encoder_title.transform(X_val['Title'].values.reshape(-1, 1))





from sklearn.preprocessing import LabelEncoder

features = ['Fare',  'Age']





for feature in features:

        le = LabelEncoder()

        le = le.fit(X_train[feature])

        X_train[feature] = le.transform(X_train[feature])

        X_val[feature] = le.transform(X_val[feature])

        

    

# Print top 10 records after transformation

X_train[0:10]
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
names = ["Kernel SVM", "Naive Bayes", "K Nearest Neighbor",

         "Decision Tree"]
classifiers = [

    SVC(kernel = 'rbf',gamma='scale'),

    GaussianNB(),

    KNeighborsClassifier(3),

    DecisionTreeClassifier(max_depth=5)]
# iterate over classifiers

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train) 

    y_pred = clf.predict(X_val)

# Here we will add the error and evaluation metrics
from sklearn.metrics import accuracy_score





data = []

# iterate over classifiers

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    

    y_pred = clf.predict(X_val)

    print(f"Accuracy for {name} : {accuracy_score(y_val, y_pred)*100.0}")

    data.append(accuracy_score(y_val, y_pred)*100.0)



models = pd.DataFrame({

    'Model': names,

    'Score': data})

models.sort_values(by='Score', ascending=False)

from sklearn.metrics import classification_report



# iterate over classifiers

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    

    y_pred = clf.predict(X_val)

    

    print(f"Classification Report for {name}")

    print(classification_report(y_val, y_pred))

    print('_'*60)
# I will use the code from : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax



class_names = np.array([0,1])



np.set_printoptions(precision=2)





# iterate over classifiers

for name, clf in zip(names, classifiers):

    clf.fit(X_train, y_train)

    

    y_pred = clf.predict(X_val)

    

    print(f"Confusion Matrix for {name}")

    # Plot non-normalized confusion matrix

    plot_confusion_matrix(y_val, y_pred, classes=class_names,

                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix

    plot_confusion_matrix(y_val, y_pred, classes=class_names, normalize=True,

                          title='Normalized confusion matrix')

    plt.show()

    print('_'*60)