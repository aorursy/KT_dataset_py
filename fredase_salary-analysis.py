##Before anything, set your working directory
#import os
#os.chdir("C:/Users/Ezinne/Desktop/Adult Salary")

#os.getcwd()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

# Import sys and warnings to ignore warning messages 
import sys
import warnings

%matplotlib inline

pd.set_option("display.max.columns", None)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
## We read in the excel data file

Salary = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

Salary.head(5)
print(len(Salary))
print(Salary.shape)
##We investigate further to know the various data types of the respective columns

Salary.info()
##Obtain summary statistics of the data

print(Salary.describe())
print(Salary.describe(include = np.object))

##Or you can use print(Salary.describe(include = 'all')) to list summary statistics of all variables at a go
Salary.isnull().sum()
Salary.columns
##Noticed earlier the presence of a special character, so we investigate this further

Salary.isin(['?']).sum(axis=0)
Salary['workclass'] = Salary['workclass'].replace('?', np.nan)
Salary['occupation'] = Salary['occupation'].replace('?', np.nan)
Salary['native-country'] = Salary['native-country'].replace('?', np.nan)
Salary.isnull().sum()
##drop all the resulting null values
Salary.dropna(how = 'any', inplace = True)

print(Salary.shape)
## For convenience we shall rename some columns

Salary.rename(columns = {'educational-num': 'education rank', 'marital-status': 'marital status',
                        'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'hours-per-week': 'HPW',
                        'native-country': 'country', 'fnlwgt': 'Final Weight'}, inplace = True)
Salary.columns
##To find out the value count of all the variables in the data we run a loop through the data

#for c in Salary.columns:
#    print("----%s----" %c)
#    print(Salary[c].value_counts())
##First we start with a visualization of all continuous variables in the data

sns.pairplot(Salary)
print(Salary['age'].value_counts())

Salary['age'].hist(figsize = (6, 6))
Salary['age'].skew()
##Next we look at the 'Final Weight' variable

print(Salary['Final Weight'].value_counts())

Salary['Final Weight'].hist(figsize = (5,5))
plt.show()
print(Salary['workclass'].value_counts( sort = False, normalize = True))

plt.figure(figsize=(8,8))

total = float(len(Salary['income']))

a = sns.countplot(x='workclass',data = Salary)

a.set_xticklabels(a.get_xticklabels(), rotation = 90)

for f in a.patches:
    height = f.get_height()
    a.text(f.get_x() + f.get_width()/2., height+3, '{:1.2f}'.format((height/total)*100),ha="center")

plt.title("WorlClass Distribution")
plt.show()
print(Salary['education'].value_counts(normalize = True))


plt.figure(figsize=(20,5))


tot = float(len(Salary))

a1 = sns.countplot(Salary['education'], palette = 'Set1')
a1.set_xticklabels(a1.get_xticklabels(), rotation = 90)

for s in a1.patches:
    height = s.get_height()
    a1.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title("Education Level Distribution")
plt.show()  
print(Salary['marital status'].value_counts(normalize = True))


plt.figure(figsize=(15,5))



tot = float(len(Salary))

ax = sns.countplot(Salary['marital status'], palette = 'Set1')

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

for s in ax.patches:
    height = s.get_height()
    ax.text(s.get_x() + s.get_width()/2.,height + 3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title("Marital Status Distribution")
plt.show()

print(Salary['occupation'].value_counts(normalize = True))


plt.figure(figsize=(20,5))


tot = float(len(Salary))

ay = sns.countplot(Salary['occupation'], palette = 'Set2')
ay.set_xticklabels(ay.get_xticklabels(), rotation = 90)

for s in ay.patches:
    height = s.get_height()
    ay.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title('Occupation variable Distribution')
plt.show()

print(Salary['relationship'].value_counts(normalize = True))


plt.figure(figsize=(10,8))


tot = float(len(Salary))

az = sns.countplot(Salary['relationship'], palette = 'Set1')
#az.set_xticklabels(az.get_xticklabels(), rotation = 90)

for s in az.patches:
    height = s.get_height()
    az.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title('Distribution of "Relationship" variable')
plt.show()

print(Salary['race'].value_counts(normalize = True))


plt.figure(figsize=(10,10))


tot = float(len(Salary))

aj = sns.countplot(Salary['race'], palette = 'Set1')
#aj.set_xticklabels(aj.get_xticklabels(), rotation = 90)

for s in aj.patches:
    height = s.get_height()
    aj.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title("Distribution of the race variable")
plt.show()

print(Salary['gender'].value_counts(normalize = True))


plt.figure(figsize=(6,6))


tot = float(len(Salary))

ap = sns.countplot(Salary['gender'])
#ap.set_xticklabels(ap.get_xticklabels(), rotation = 90)

for s in ap.patches:
    height = s.get_height()
    ap.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title('Distribution by Gender')
plt.show()

print(Salary['income'].value_counts(normalize = True))


plt.figure(figsize=(6,6))


tot = float(len([Salary]))

ab = sns.countplot(Salary['income'])
#ab.set_xticklabels(ab.get_xticklabels(), rotation = 90)

for s in ab.patches:
    height = s.get_height()
    ab.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')
    
plt.title("Income Distribution")
plt.show()

fig = plt.figure(figsize = (6,6))
sns.boxplot(x = 'income', y = 'age', data = Salary, palette = "Accent").set_title("Income distribution by Age")

##Alternatively we can represent the above plot using a violin plot

sns.violinplot(x = 'income', y = 'age', data = Salary, size = 6)
plt.title('Violin Plot of Age by Income')
##Just for confirmation sake, we check the median age across income brackets

print(Salary.loc[Salary['income'] == '<=50K', 'age'].median())
print(Salary.loc[Salary['income'] == '>50K', 'age'].median())
print(pd.crosstab(Salary['income'], Salary['workclass']))

fig = plt.figure(figsize = (12,10))
ad = sns.countplot(x = 'workclass', hue = 'income', data = Salary, palette = "Spectral")

#ad.set_xticklabels(ad.get_xticklabels(), rotation = 90)

for s in ad.patches:
    height = s.get_height()
    ad.text(s.get_x()+s.get_width()/2.,height+3,'{:1.2f}'.format((height/total)*100),ha='center')

plt.title('Income split by WorkClass')
plt.show()
print(pd.crosstab(Salary['income'], Salary['relationship']))

plt.figure(figsize = (10, 10))
sns.countplot(x = 'relationship', hue = 'income', data = Salary, palette = "gist_earth").set_title('Income by Relationship')
plt.figure(figsize = (20, 8))

sns.catplot(y = 'race', hue = 'income', col = 'gender', data = Salary, kind = 'count', palette = 'twilight')
#plt.title('Income by Race across Gender')
plt.figure(figsize = (10,10))

sns.catplot(y = 'education', hue = 'income', data = Salary, kind = 'count', col = 'gender', palette = 'pastel')

plt.figure(figsize = (8,8))
sns.catplot(y = 'marital status', hue = 'income', col = 'gender', data = Salary, palette = 'prism', kind = 'count')
plt.figure(figsize = (10,10))
sns.catplot(y = 'occupation', hue = 'income', col = 'gender', kind = 'count', data = Salary, palette = 'Set3_r')
Salary.drop(['education rank', 'country', 'relationship'], axis = 1, inplace = True)

Salary.head(5)
### We proceed to code the various categorical variables
ed = set(Salary['education'])
wc = set(Salary['workclass'])
ms = set(Salary['marital status'])
occ = set(Salary['occupation'])
#rel = set(Salary['relationship'])
gen = set(Salary['gender'])
inc = set(Salary['income'])
race = set(Salary['race'])
print(ed)

Salary['education'] = Salary['education'].map({'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3,
                                              '9th': 4, '10th': 5, '11th': 6, '12th': 7, 'HS-grad': 8,
                                              'Some-college': 9, 'Assoc-voc': 10, 'Assoc-acdm': 11,
                                              'Bachelors': 12, 'Masters': 13, 'Doctorate': 14, 'Prof-school': 15}).astype(int)
print(wc)

Salary['workclass'] = Salary['workclass'].map({'Without-pay': 0, 'Self-emp-not-inc': 1, 'Self-emp-inc': 2,
                                              'Local-gov': 3, 'State-gov': 4, 'Federal-gov': 5, 'Private': 6}).astype(int)
print(ms)

Salary['marital status'] = Salary['marital status'].map({'Never-married': 0, 'Separated': 1, 'Divorced': 2,
                                                        'Widowed': 3, 'Married-spouse-absent': 4, 'Married-civ-spouse': 5,
                                                        'Married-AF-spouse': 6}).astype(int)
print(occ)

Salary['occupation'] = Salary['occupation'].map({'Other-service': 0, 'Craft-repair': 1, 'Priv-house-serv': 2,
                                                'Handlers-cleaners': 3, 'Farming-fishing': 4, 'Adm-clerical': 5,
                                                'Transport-moving': 6, 'Machine-op-inspct': 7, 'Sales': 8, 'Armed-Forces': 9,
                                                'Tech-support': 10, 'Protective-serv': 11, 'Exec-managerial': 12,
                                                'Prof-specialty': 13}).astype(int)
#print(rel)

#Salary['relationship'] = Salary['relationship'].map({'Not-in-family': 0, 'Other-relative': 1, 'Unmarried': 2,
#                                                    'Own-child': 3, 'Wife': 4, 'Husband': 5}).astype(int)
print(gen)

Salary['gender'] = Salary['gender'].map({'Female': 0, 'Male': 1}).astype(int)
print(inc)

Salary['income'] = Salary['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
print(race)

Salary['race'] = Salary['race'].map({'Other': 0, 'Amer-Indian-Eskimo': 1, 'Asian-Pac-Islander': 2, 'Black': 3,
                                    'White': 4}).astype(int)
Salary.head(5)
Salary.tail(5)
##Plot another correlation matrix including our new coded variables
corr_matrix = Salary.corr()

f, ax = plt.subplots(figsize = (12,10))
k = 12 ##The number of variables to be used for the heatmap
cols = corr_matrix.nlargest(k, 'income')['income'].index ##The 'income' variable is used as index as it is compared against others
cm = np.corrcoef(Salary[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10},
                yticklabels = cols.values, xticklabels = cols.values) ##annot prints the values inside the matrix

plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

##Split the data into train and test sets while stratifying our target variable(because its imbalanced)

X = Salary.drop('income', axis = 1)
y = Salary['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30, stratify = y)

print("Size of the train dataset: ", len(X_train))
print("Size of the test dataset: ", len(X_test))
##In fitting our logistic model we take into account that this is an unbalanced dataset,
logmodel = LogisticRegression(solver = 'lbfgs', max_iter = 200)

logmodel.fit(X_train, y_train)
pred = logmodel.predict(X_test)

print(confusion_matrix(y_test, pred))

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logmodel.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logmodel.score(X_test, y_test)))
##Plot the Confusion Matrix
import itertools
matrix = confusion_matrix(y_test, pred)

plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')

plt.imshow(matrix, interpolation = 'nearest', cmap = plt.cm.Blues)
plt.colorbar()

fmt = 'd'

thresh = matrix.max()/2.
for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
    plt.text(j, i, format(matrix[i, j], fmt), 
            horizontalalignment = 'center', color = 'white' if matrix[i, j] > thresh else 'black')
    
class_names = ['Class-0', 'Class-1']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation = 90)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True Label', size = 10)
plt.xlabel('Predicted Label', size = 10)
plt.show()
print(classification_report(pred, y_test))

from sklearn import metrics 

pred_prob = logmodel.predict_proba(X_test)

y_preds = pred_prob[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_preds)
auc_score = metrics.auc(fpr, tpr)
#plt.pred_prob()
plt.figure(figsize = (10,10))
plt.title('ROC Curve')
plt.plot(fpr, tpr, label = 'AUC = {:.2f}'.format(auc_score))
plt.plot([0, 1], [0, 1], 'r--')

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
optimal_idx = np.argmax(np.abs(tpr - fpr))
optimal_threshold = _[optimal_idx]
optimal_threshold
gmeans = np.sqrt(tpr * (1 - fpr))
ix = np.argmax(gmeans)
print('Best Threshold = %f, G-mean = %.3f' % (_[ix], gmeans[ix]))
y_preds = pred_prob[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_preds)

ix = np.argmax(gmeans)
print('Best Threshold = %f, G-mean = %.3f' % (_[ix], gmeans[ix]))

##Plot the roc curve for the model
plt.figure(figsize = (8, 8))
plt.plot([0,1], [0,1], linestyle = '--', color = 'red', label = 'No Skill')
plt.plot(fpr, tpr, marker = '.',color = 'yellow', label = 'Logistic')
plt.scatter(fpr[ix], tpr[ix], marker = 'o', color = 'black', label = 'Best')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
##Tune the hyper-parameters by cross validation
n_splits = 15 ##specify the number of splits

kfold = KFold(n_splits, random_state = 20) ##splits the data set into n folds for evaluation

result = cross_val_score(logmodel, X, y, cv = kfold, scoring = 'accuracy')

##The accuracy of the k-fold cross-validation can be obtained from the mean of the results
print('Accuarcy: %.3f (%.3f)' % (result.mean(), result.std()))
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(max_depth = 10)
tree_model.fit(X_train,y_train)
prediction = tree_model.predict(X_test)


print(confusion_matrix(y_test, prediction))

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(tree_model.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(tree_model.score(X_test, y_test)))
from sklearn import tree
feature_names = ['age', 'workclass', 'Final Weight', 'education', 'marital status', 'occupation',
                'race', 'gender', 'capital gain', 'capital loss', 'HPW']

cn = ['<=50K', '>50K']

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8), dpi = 300)
tree.plot_tree(tree_model, feature_names = feature_names, class_names = cn, filled = True)
plt.show()
print(classification_report(prediction, y_test))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state =0, n_jobs = -1, n_estimators = 20, class_weight = 'balanced').fit(X_train, y_train)

pred2 = clf.predict(X_test)

print(confusion_matrix(y_test, pred2))

print('Accuracy of RandomForest classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RandomForest classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print(classification_report(pred2, y_test))
##Apply kfolds cross_validation
n_splits = 10
kfold1 = KFold(n_splits, random_state = 20)
result1 = cross_val_score(clf, X, y, cv = kfold1, scoring = 'accuracy')
print('Accuarcy: %.3f (%.3f)' % (result1.mean(), result1.std()))
important = clf.feature_importances_

feature_importance = np.array(important)
feature_names = np.array(feature_names)

data = {'feature_names': feature_names, 'feature_importance': important}
df = pd.DataFrame(data)

df.sort_values(by = ['feature_importance'], ascending = False, inplace = True)
#fig, ax = plt.subplots()
plt.figure(figsize = (10, 8))
#plt.bar([x for x in range(len(important))], important)
sns.barplot(x = df['feature_importance'], y = df['feature_names'], palette = 'twilight')
plt.ylabel('Feature')
plt.xlabel('Realtive Importance')
#ax.set_xticklabels(feature_names, minor = False)
plt.title("Feature Importance in Random Classifier")
plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn_class = KNeighborsClassifier(n_neighbors = 16).fit(X_train, y_train)
knn_pred = knn_class.predict(X_test)

print(confusion_matrix(knn_pred, y_test))

print(classification_report(knn_pred, y_test))
accrate = list()

for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors = i)
    score = cross_val_score(knn, X, y, cv = 10)
    accrate.append(score.mean())


plt.figure(figsize = (10,10))
plt.plot(range(1, 30), accrate, color = 'green', linestyle = 'dashed', marker = 'o',
        markerfacecolor = 'red', markersize = 10)

plt.title('Accuarcy Vs Different K values')
plt.xlabel('K')
plt.ylabel('Accuarcy Rate')
