#Now let's open it with pandas
import pandas as pd
from pandas import Series,DataFrame
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
import time
import random


# Set up the Titanic csv file as a DataFrame
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_ytest = pd.read_csv('../input/gender_submission.csv')
titanic_train.head()
titanic_test.head()
titanic_ytest.head()

result = pd.merge(titanic_ytest,titanic_test, how='inner', on=['PassengerId'])
result=result.append(titanic_train)
result.head()


titanic=result
titanic.head()
titanic.isna().any()
# Actual replacement of the missing value using median value.
titanic = titanic.fillna((titanic.median()))
titanic.head()
titanic.isna().any()
titanic.describe()
#Correlation with Quality with respect to attributes
titanic.corrwith(titanic.Survived).plot.bar(
        figsize = (20, 10), title = "Correlation with quality", fontsize = 15,
        rot = 45, grid = True)
## Correlation Matrix
sns.set(style="white")

# Compute the correlation matrix
corr = titanic.corr()
corr.head()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
titanic.columns

#Assigning and dividing the dataset
X = titanic.drop(columns=['Survived','Embarked','Cabin','Name','Ticket'],axis=1)
y=titanic['Survived']
X.isna().any()
y.head()
data=titanic.drop(columns=['Survived','Embarked','Cabin','Name','Ticket'],axis=1)
data.head()
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encoding the Dependent Variable
labelencoder_y = LabelEncoder()

X['Sex'] = labelencoder_y.fit_transform(X['Sex'])



X.head()
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Encoding the Dependent Variable
labelencoder_y = LabelEncoder()

data['Sex'] = labelencoder_y.fit_transform(data['Sex'])

data.head()

features_label = data.columns[1:]
features_label

#Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
importances = classifier.feature_importances_
indices = np. argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i+1, 30, features_label[i],importances[indices[i]]))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]),importances[indices], color="green", align="center")
plt.xticks(range(X.shape[1]),features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_test2 = pd.DataFrame(sc.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
pd.DataFrame(explained_variance)
#### Model Building ####

### Comparing Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l1')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results)
## SVM (rbf)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)
## Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)
#we ttok the highest accuracy model svm rbf classifier
## K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,
                             cv = 100)
print("SVM Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))
# Round 1: SVM tuning
parameters = [{ 'C': [1,  100],'kernel': ['linear']},
              { 'C': [1,  100],'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]
# Make sure classifier points to the RF model
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)

t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
# Predicting Test Set
y_pred = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (n=100, GSx2 + Gini)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
results
import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(cm,classes=[0,1])
sns.set(rc={'figure.figsize':(6,6)})
plt.show()

#Let's see how our model performed
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Let's import what we'll need for the analysis and visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Let's first check gender
sns.countplot('Sex',data=result)
sns.set(rc={'figure.figsize':(6,6)})
# Now let's seperate the genders by classes, remember we can use the 'hue' arguement here!
sns.countplot('Pclass',data=result,hue='Sex')
# We'll treat anyone as under 16 as a child, and then use the apply technique with a function to create a new column


# First let's make a function to sort through the sex 
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex
    

# We'll define a new column called 'person', remember to specify axis=1 for columns and not index
result['person'] = result[['Age','Sex']].apply(male_female_child,axis=1)
# Let's see if this worked, check out the first ten rows
result[0:10]
# Let's try the factorplot again!
sns.countplot('Pclass',data=result,hue='person')
# Quick way to create a histogram using pandas
result['Age'].hist(bins=70)
# We could also get a quick overall comparison of male,female,child
result['person'].value_counts()
# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

# Set the figure equal to a facetgrid with the pandas dataframe as its data source, set the hue, and change the aspect ratio.
fig = sns.FacetGrid(result, hue="Sex",aspect=4)

# Next use map to plot all the possible kdeplots for the 'Age' column by the hue choice
fig.map(sns.kdeplot,'Age',shade= True)

# Set the x max limit by the oldest passenger
oldest = result['Age'].max()

#Since we know no one can be negative years old set the x lower limit at 0
fig.set(xlim=(0,oldest))

#Finally add a legend
fig.add_legend()
# We could have done the same thing for the 'person' column to include children:

fig = sns.FacetGrid(result, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = result['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
# Let's do the same for class by changing the hue argument:
fig = sns.FacetGrid(result, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = result['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
# Let's get a quick look at our dataset again
result.head()
# First we'll drop the NaN values and create a new object, deck
deck = result['Cabin'].dropna()
# Quick preview of the decks
deck.head()
# So let's grab that letter for the deck level with a simple for loop

# Set empty list
levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    

# Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.countplot('Cabin',data=cabin_df,palette='winter_d')
# Redefine cabin_df as everything but where the row was equal to 'T'
cabin_df = cabin_df[cabin_df.Cabin != 'T']
#Replot
sns.countplot('Cabin',data=cabin_df,palette='summer')
# Now we can make a quick factorplot to check out the results, note the x_order argument, used to deal with NaN values
sns.countplot('Embarked',data=result,hue='Pclass',order=['C','Q','S'])
# Let's start by adding a new column to define alone

# We'll add the parent/child column with the sibsp column
result['Alone'] =  result.Parch + result.SibSp
result['Alone']
# Look for >0 or ==0 to set alone status
result['Alone'].loc[result['Alone'] >0] = 'With Family'
result['Alone'].loc[result['Alone'] == 0] = 'Alone'

# Note it's okay to ignore an  error that sometimes pops up here. For more info check out this link
url_info = 'http://stackoverflow.com/questions/20625582/how-to-deal-with-this-pandas-warning'
# Let's check to make sure it worked
result.head()
# Now let's get a simple visualization!
sns.countplot('Alone',data=result,palette='Blues')
# Let's start by creating a new column for legibility purposes through mapping (Lec 36)
result["Survivor"] = result.Survived.map({0: "no", 1: "yes"})

# Let's just get a quick overall view of survied vs died. 
sns.countplot('Survivor',data=result,palette='Set1')
# Let's use a factor plot again, but now considering class
sns.pointplot('Pclass','Survived',data=result)
# Let's use a factor plot again, but now considering class and gender
sns.pointplot('Pclass','Survived',hue='person',data=result)
# Let's use a linear plot on age versus survival
sns.lmplot('Age','Survived',data=result)
# Let's use a linear plot on age versus survival using hue for class seperation
sns.lmplot('Age','Survived',hue='Pclass',data=result,palette='winter')
# Let's use a linear plot on age versus survival using hue for class seperation
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=result,palette='winter',x_bins=generations)
sns.lmplot('Age','Survived',hue='Sex',data=result,palette='winter',x_bins=generations)






