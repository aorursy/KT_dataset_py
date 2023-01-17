import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LinearRegression, Lasso

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor

from sklearn.linear_model import LogisticRegression





np.random.seed(2)



import warnings

warnings.filterwarnings("ignore")



plt.style.use('ggplot')
df = pd.read_csv("../input/student-alcohol-consumption/student-mat.csv")
df.head()
df.info()
df.columns
plt.figure(figsize=(6, 4))

plt.hist(df.G3)

plt.title('Grade distribution', fontsize=12)

plt.show()
plt.figure(figsize=(5, 4))

sns.countplot(x = 'school', hue = 'reason', data = df )

plt.title('Reason of selecting school', fontsize=12)

plt.show()
plt.figure(figsize=(12,4))

sns.countplot(x = 'G3', hue = 'reason', data = df )

plt.title('Reason of selecting school vs Grade', fontsize=12)

plt.show()



g = sns.FacetGrid(df, col='school', height=4)

g = g.map(sns.distplot,  "G3").add_legend()
df.internet.value_counts()
g = sns.FacetGrid(df, col='internet', height=4)

g = g.map(sns.distplot,  "G3").add_legend()

g.fig.subplots_adjust(top=0.9)

g.fig.suptitle('Internet availability vs Grade', fontsize=12)

plt.show()
plt.figure(figsize=(8, 4))

sns.swarmplot(x ='G3',y='absences', data = df, palette = sns.color_palette("Set1"))

plt.title('Number of absences vs Grade', fontsize=12)

plt.show()
plt.figure(figsize=(8, 4))

sns.countplot(x = 'G3', hue = 'paid', data = df )

plt.title('Extra classes vs Grade', fontsize=12)

plt.show()



g = sns.FacetGrid(df, col='paid', height=4)

g = g.map(sns.distplot,  "G3").add_legend()
g = sns.FacetGrid(df, col='Dalc', height=3, col_wrap=3)

g = g.map(sns.distplot,  "G3").add_legend()
g = sns.FacetGrid(df, col='Walc', height=3, col_wrap=3)

g = g.map(sns.distplot,  "G3").add_legend()
binary_encoder = lambda x: 1 if x == 'yes' else 0



df.schoolsup = df.schoolsup.apply(binary_encoder)

df.famsup = df.famsup.apply(binary_encoder)

df.paid = df.paid.apply(binary_encoder)

df.activities = df.activities.apply(binary_encoder)

df.nursery = df.nursery.apply(binary_encoder)

df.higher = df.higher.apply(binary_encoder)

df.internet = df.internet.apply(binary_encoder)

df.romantic = df.romantic.apply(binary_encoder)



sex_encoder = lambda x: 1 if x == 'F' else (0 if x == 'M' else x)

df.sex = df.sex.apply(sex_encoder)



address_encoder = lambda x: 1 if x == 'U' else (0 if x == 'R' else x)

df.address = df.address.apply(address_encoder)



famsize_encoder = lambda x: 1 if x == 'GT3' else (0 if x == 'LE3' else x)

df.famsize = df.famsize.apply(famsize_encoder)



pstatus_encoder = lambda x: 1 if x == 'T' else (0 if x == 'A' else x)

df.Pstatus = df.Pstatus.apply(pstatus_encoder)



school_encoder = lambda x: 1 if x == 'GP' else (0 if x == 'MS' else x)

df.school = df.school.apply(school_encoder)



def job_encoder(val):

    if val == 'teacher':

        val = 0

    elif val == 'health':

        val = 1

    elif val == 'services':

        val = 2

    elif val == 'at_home':

        val = 3

    elif val == 'other':

        val = 4

    else:

        val = val

    return val



            

df.Mjob = df.Mjob.apply(job_encoder)

df.Fjob = df.Fjob.apply(job_encoder)



def reason_encoder(val):

    if val == 'course':

        val = 0

    elif val == 'home':

        val = 1

    elif val == 'reputation':

        val = 2

    elif val == 'other':

        val = 3

    else:

        val = val

    return val



df.reason = df.reason.apply(reason_encoder)



def gaurdian_encoder(val):

    if val == 'mother':

        val = 0

    elif val == 'father':

        val = 1

    elif val == 'other':

        val = 2

    else:

        val = val

    return val



df.guardian = df.guardian.apply(gaurdian_encoder)

df.info()
corr = df.corr()



plt.figure(figsize=(14, 12))

sns.heatmap(corr)
def scale_and_split(df, test_sizre=0.3):

    

    target = df[['G3']]

    features = df.drop('G3', axis = 1)

    labels = list(target.G3.unique())

    

#     scaler = StandardScaler()

#     features = scaler.fit_transform(features)

    

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)

    

    return X_train, X_test, y_train, y_test, labels
def evaluate_classifier(model, df):

    

    X_train, X_test, y_train, y_test, labels = scale_and_split(df)

    

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print('Cross validation score - ', scores.mean()*100)

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    accuracy = accuracy_score(y_test, y_pred) 

    print('Test accuracy - ',accuracy*100)
def evaluate_regressor(model, df):

    

    X_train, X_test, y_train, y_test, labels = scale_and_split(df)

    

    scores = cross_val_score(model, X_train, y_train, cv=5)

    print('Cross validation score - ', scores.mean()*100)

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print('MSE - ', mse)

    print('R2 Score - ', r2*100)
lr = LogisticRegression()

dtc = DecisionTreeClassifier(criterion='gini', max_depth=12, random_state=42)

rfc = RandomForestClassifier(n_estimators=100, max_depth=12 ,random_state=42)

adc = AdaBoostClassifier()



print('\nEvaluation results - Logistic Regression')

evaluate_classifier(lr, df)



print('\nEvaluation results - Decision Tree Classifier')

evaluate_classifier(dtc, df)



print('\nEvaluation results - Random Forest Classifier')

evaluate_classifier(rfc, df)



print('\nEvaluation results - Adaboost Classifier')

evaluate_classifier(adc, df)
lr = LinearRegression()

lasso = Lasso()

dtr = DecisionTreeRegressor( random_state=42)

rfr = RandomForestRegressor( random_state=42)

adr = AdaBoostRegressor()



print('\nEvaluation results - Linear Regression')

evaluate_regressor(lr, df)



print('\nEvaluation results - Decision Tree Regressor')

evaluate_regressor(dtr, df)



print('\nEvaluation results - Random Forest Regressor')

evaluate_regressor(rfr, df)



print('\nEvaluation results - Adaboost Regressor')

evaluate_regressor(adr, df)