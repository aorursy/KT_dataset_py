import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import imblearn

#read the csv data from env and store it
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

# drop cols that will not be used
df.drop('Firstname', axis=1, inplace=True)
df.drop('Lastname', axis=1, inplace=True)
df.drop('PassengerId', axis=1, inplace=True)

#transform category col to numeric (M=1, F=0, C=1, P=0)
df['Sex'].replace('M', 1, inplace=True)
df['Sex'].replace('F', 0, inplace=True)
df['Category'].replace('C', 1, inplace=True)
df['Category'].replace('P', 0, inplace=True)
df.sample(5)
sur_counts = df['Survived'].value_counts()
death_toll_rate = sur_counts[0] / (sur_counts[0] + sur_counts[1])
print(f'The naive baseline is: {death_toll_rate * 100:.2f}%')

sur_counts.plot(kind='bar', title='Classes')
plt.show()
#check if there is a correlation between survived people and their country, 
#examine the two major countries(Sweden and Estonia) where most of the passengers and crew come from
sweden_counts = df.Survived[df['Country'] == 'Sweden'].value_counts()
estonia_counts = df.Survived[df['Country'] == 'Estonia'].value_counts()
finland_counts = df.Survived[df['Country'] == 'Finland'].value_counts()

country_df = pd.DataFrame({'country' : ['Sweden', 'Estonia', 'Finland'], 'Dead' : [sweden_counts[0], estonia_counts[0], finland_counts[0]], 'Survived' : [sweden_counts[1], estonia_counts[1], finland_counts[1]]})
country_df.plot(kind='bar', stacked=True, x='country')
plt.title("Country regard survive rate")
plt.xlabel("Country")
plt.ylabel("death toll")
plt.show()
country = df.Country.unique()
# other_country = country[(country != 'Estonia') & (country != 'Sweden')].values
#remove Estonia and Sweden
other_country = country[2:]
df.Country = df.Country.replace(to_replace=other_country, value='Other')
df = pd.get_dummies(df, columns = ['Country'])
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

#https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets#t3
def plot_2d_space(X, y, label, axis=plt):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        axis.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    if axis == plt:
        axis.title(label)
    else:
        axis.title.set_text(label)
    axis.legend(loc='upper right')  
    
f = plt.figure(figsize=(10,10))
pca = PCA(n_components=2)
X = pca.fit_transform(df.loc[:, df.columns != 'Survived'])
y = df.Survived
plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')

fig, axis = plt.subplots(2,2,figsize=(15,15))
        
# Under sample randomsampler
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

# print('resampled dataset shape:', Counter(y_rus))
plot_2d_space(X_rus, y_rus, 'Random under-sampling', axis=axis[0,0])

#Tomkel links

tl = TomekLinks(sampling_strategy='majority')
X_tl, y_tl= tl.fit_sample(X, y)

plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling', axis=axis[0,1])

#over sample using SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_sample(X, y)

# print('resampled dataset shape:', Counter(y_sm))

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling', axis=axis[1,0])

#over sample using combined SMOTE and Tomek links

smt = SMOTETomek(sampling_strategy='auto')
X_smt, y_smt = smt.fit_sample(X, y)

# print('resampled dataset shape:', Counter(y_smt))

plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links', axis=axis[1,1])
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#standardize the Age column 
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
X = df.loc[:, df.columns != 'Survived']
y = df.Survived
# Under sample randomsampler
X_rus, y_rus = rus.fit_resample(X, y)


#Tomkel links
X_tl, y_tl= tl.fit_sample(X, y)

#over sample using SMOTE
X_sm, y_sm = smote.fit_sample(X, y)

#over sample using combined SMOTE and Tomek links
X_smt, y_smt = smt.fit_sample(X, y)

#logreg with SMOTE strategy
logreg = LogisticRegression(random_state=0, C=1)
logreg.fit(X_sm, y_sm)
pred_y = logreg.predict(X_sm)
print('1st Confusion Matrix:')
print(confusion_matrix(y_sm, pred_y))
print('\nAccuracy score: ', accuracy_score(y_sm, pred_y))

# #logreg with random under sampling
logreg.fit(X_rus, y_rus)
pred_y = logreg.predict(X_rus)
print('\n2nd Confusion Matrix:')
print(confusion_matrix(y_rus, pred_y))
print('Accuracy score: ', accuracy_score(y_rus, pred_y))

#logreg with tomkel links sampling strategy
logreg.fit(X_tl, y_tl)
pred_y = logreg.predict(X_tl)
print('\n3rd Confusion Matrix: \n')
print(confusion_matrix(y_tl, pred_y))
print('\nAccuracy score: ', accuracy_score(y_tl, pred_y))
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)

#with SMOTE sampling strategy
gbrt.fit(X_sm, y_sm)
pred_y = gbrt.predict(X_sm)
print('Confusion Matrix: \n')
print(confusion_matrix(y_sm, pred_y))
print('Accuracy score: ', accuracy_score(y_sm, pred_y))

#with Tomkel Link sampling strategy
gbrt.fit(X_tl, y_tl)
pred_y = gbrt.predict(X_tl)
print('Confusion Matrix: \n')
print(confusion_matrix(y_tl, pred_y))
print('\nAccuracy score: ', accuracy_score(y_tl, pred_y))

#try gbrt with SMOTE w/ Tomek links sampling strategy
gbrt.fit(X_smt, y_smt)
pred_y = gbrt.predict(X_smt)
print('Confusion Matrix: \n')
print(confusion_matrix(y_smt, pred_y))
print('Accuracy score: ', accuracy_score(y_smt, pred_y))

#try random under sampling strategy
gbrt.fit(X_rus, y_rus)
pred_y = gbrt.predict(X_rus)
print('Confusion Matrix: \n')
print(confusion_matrix(y_rus, pred_y))
print('Accuracy score: ', accuracy_score(y_rus, pred_y))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state = 0)

#with SMOTE sampling strategy
forest.fit(X_sm, y_sm)
pred_y = forest.predict(X_sm)
print('Confusion Matrix: \n')
print(confusion_matrix(y_sm, pred_y))
print('\nAccuracy score: ', accuracy_score(y_sm, pred_y))

#with Tomkel Link sampling strategy
forest.fit(X_tl, y_tl)
pred_y = forest.predict(X_tl)
print('Confusion Matrix: \n')
print(confusion_matrix(y_tl, pred_y))
print('\nAccuracy score: ', accuracy_score(y_tl, pred_y))

#with SMOTE+Tomkel Link sampling strategy
forest.fit(X_smt, y_smt)
pred_y = forest.predict(X_smt)
print('Confusion Matrix: \n')
print(confusion_matrix(y_smt, pred_y))
print('\nAccuracy score: ', accuracy_score(y_smt, pred_y))

#with random under sampling strategy
forest.fit(X_rus, y_rus)
pred_y = forest.predict(X_rus)
print('Confusion Matrix: \n')
print(confusion_matrix(y_rus, pred_y))
print('\nAccuracy score: ', accuracy_score(y_rus, pred_y))

#without resampling
forest.fit(X, y)
pred_y = forest.predict(X)
print('Confusion Matrix: \n')
print(confusion_matrix(y, pred_y))
print('\nAccuracy score: ', accuracy_score(y, pred_y))
from sklearn.model_selection import GridSearchCV

params = {'bootstrap': [True, False],
          'max_depth': [40, 50, 60, None],
          'max_features': ['auto'],
          'min_samples_leaf': [1, 2, 4],
          'min_samples_split': [2, 5, 10],
          'n_estimators': [20, 100, 200]}

gs_cv = GridSearchCV(RandomForestClassifier(random_state = 0, n_jobs = -1), params, cv=5, scoring='accuracy')
gs_cv.fit(X_smt, y_smt)

print(f'best params: {gs_cv.best_estimator_}')
gs_cv.best_score_
forest = RandomForestClassifier(bootstrap=False, max_depth=50, n_estimators=200,
                       n_jobs=-1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, random_state = 42)

forest.fit(X_smt, y_smt)
pred_y = forest.predict(X_smt)
print(f'train accuracy: {accuracy_score(y_smt, pred_y)}')

forest.fit(X_train, y_train)
pred_y = forest.predict(X_test)
print(f'test accuracy: {accuracy_score(y_test, pred_y)}')

print(confusion_matrix(y_test, pred_y))