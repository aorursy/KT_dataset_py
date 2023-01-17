#Helper function to plot training vs test data accuracy



def plot_training_vs_test(training, test):

    objects = ('Training', 'Test')

    performance = [training, test]

    y_pos = np.arange(len(objects))

    

    plt.bar(y_pos,performance, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)

    axes = plt.gca()

    axes.set_ylim([0,1])

    axes.tick_params(axis='x', colors='white')

    axes.tick_params(axis='y', colors='white')

    plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import seaborn as sns

from sklearn import model_selection, metrics, linear_model, datasets, feature_selection, preprocessing

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')





df = pd.read_csv("../input/speed-dating-experiment/Speed Dating Data.csv", encoding="ISO-8859-1")

manualtest = pd.read_csv("../input/manualdata/Manual_Test.csv", encoding="ISO-8859-1")

do_manual_test = True

genderToCompare = int(input("Enter what gender to model for (0 = female  1 = male  * = both): "))



all_fields_to_extract = [

    'int_corr',

    'samerace',

    'attr',

    'sinc',

    'intel',

    'fun',

    'amb',

    'shar',

    'like',

    'age',

    'field_cd',

    'race',

    'income',

    'imprace',

    'imprelig',

    'age_o',

    'dec'

]

features_to_include = ['int_corr', 

                       'samerace',

                       'attr' ,

                       'sinc',

                       'intel',

                       'fun',

                       'amb',

                       'shar',

                       'age',

                       'field_cd',

                       'race',

                       'imprace',

                       'agediff'

                      ]   

#                       'imprelig'                       

    

if(genderToCompare == 0):

    df = df.loc[df.gender == 0, :].loc[:, all_fields_to_extract]

    print('female chosen')

elif(genderToCompare == 1):

    df = df.loc[df.gender == 1, :].loc[:, all_fields_to_extract]

    print('male chosen')

else:

    df = df.loc[:,all_fields_to_extract]

    print('both chosen')

   

df['agediff'] = df['age_o'] - df['age']

df['income'] = df['income'].str.replace(',', '')

manualtest['income'] = manualtest['income'].str.replace(',', '')

manualtest['agediff'] = manualtest['age_o'] - manualtest['age']



df.sample(5)
df = df.replace([np.inf, -np.inf], np.nan)

#Average out the NaNs

#df = df.fillna(df.mean())

#Drop NaN records

df = df.dropna()
feature_rank = feature_selection.mutual_info_classif(df[features_to_include], df['dec'])

feature_rank_df = pd.DataFrame(list(zip(features_to_include, feature_rank)), columns=['Feature', 'Score'])

feature_rank_df.sort_values(by='Score', ascending = False).head()

#sns.pairplot(df, hue='dec')

fig, ax = plt.subplots(figsize=(20,20))

sns.heatmap(df.corr(), annot = True)
X_trn, X_tst, Y_trn, Y_tst = train_test_split(df[features_to_include], df['dec'], test_size=0.4)

print('Size of training: ', len(Y_trn))

print('Size of testing: ', len(Y_tst))
logreg = sk.linear_model.LogisticRegression()

logreg.fit(X_trn[features_to_include],Y_trn)



print('accuracy on training data',round(logreg.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(logreg.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(logreg.score(X_trn[features_to_include], Y_trn), logreg.score(X_tst[features_to_include], Y_tst))



if(do_manual_test):

    preds = logreg.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])

from sklearn.svm import LinearSVC



SVM_model = LinearSVC()

SVM_model.fit(X_trn[features_to_include], Y_trn)



print('accuracy on training data',round(SVM_model.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(SVM_model.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(SVM_model.score(X_trn[features_to_include], Y_trn), SVM_model.score(X_tst[features_to_include], Y_tst))



if(do_manual_test):

    preds = SVM_model.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])
from sklearn.svm import SVC



kernel_SVM_model = SVC(kernel='rbf')

kernel_SVM_model.fit(X_trn[features_to_include], Y_trn)



print('accuracy on training data',round(kernel_SVM_model.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(kernel_SVM_model.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(kernel_SVM_model.score(X_trn[features_to_include], Y_trn), kernel_SVM_model.score(X_tst[features_to_include], Y_tst))



if(do_manual_test):

    preds = kernel_SVM_model.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])
from sklearn.svm import SVC



#kernel_SVM_model = SVC(kernel='poly')

#kernel_SVM_model.fit(X_trn[features_to_include], Y_trn)



#print('accuracy on training data',round(kernel_SVM_model.score(X_trn[features_to_include], Y_trn),2),'%')

#print('accuracy on test data',round(kernel_SVM_model.score(X_tst[features_to_include], Y_tst),2),'%')



#plot_training_vs_test(kernel_SVM_model.score(X_trn[features_to_include], Y_trn), kernel_SVM_model.score(X_tst[features_to_include], Y_tst))
from sklearn.svm import SVC



kernel_SVM_model = SVC(kernel='sigmoid', C=3.0, coef0=1.0, probability=True)

kernel_SVM_model.fit(X_trn[features_to_include], Y_trn)



print('accuracy on training data',round(kernel_SVM_model.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(kernel_SVM_model.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(kernel_SVM_model.score(X_trn[features_to_include], Y_trn), kernel_SVM_model.score(X_tst[features_to_include], Y_tst))



if(do_manual_test):

    preds = kernel_SVM_model.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])
from sklearn import tree

DT_model = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=50)

DT_model.fit(X_trn[features_to_include], Y_trn)



print('accuracy on training data',round(DT_model.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(DT_model.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(DT_model.score(X_trn[features_to_include], Y_trn), DT_model.score(X_tst[features_to_include], Y_tst))



if(do_manual_test):

    preds = DT_model.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])
from sklearn.ensemble import RandomForestClassifier



#bestScore = 0.0

#bestX = 1

#bestY = 1

#for x in range(1,100):

#    for y in range(1,15):

#        random_forest_model = RandomForestClassifier(n_estimators=x, max_depth=y)

#        random_forest_model.fit(X_trn[features_to_include], Y_trn)

#        if(round(random_forest_model.score(X_tst[features_to_include], Y_tst),2) > bestScore):

#            bestScore = round(random_forest_model.score(X_tst[features_to_include], Y_tst),2)

#            bestX = x

#            bestY = y

#print('best estimators: ', bestX, ' best max depth: ', bestY)



random_forest_model = RandomForestClassifier(n_estimators=5, max_depth=3)

random_forest_model.fit(X_trn[features_to_include], Y_trn)

print('accuracy on training data',round(random_forest_model.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(random_forest_model.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(random_forest_model.score(X_trn[features_to_include], Y_trn), random_forest_model.score(X_tst[features_to_include], Y_tst))



df_feat_importances = pd.DataFrame(list(zip(features_to_include,random_forest_model.feature_importances_)), columns=['Feature','Importance'])

df_feat_importances.sort_values(by='Importance', inplace=True)

plt.figure(figsize=[6,8])

plt.barh(df_feat_importances['Feature'],df_feat_importances['Importance'])



if(do_manual_test):

    preds = random_forest_model.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])
from sklearn.ensemble import GradientBoostingClassifier



gbm_model = GradientBoostingClassifier(n_estimators=50, max_depth=8, min_samples_leaf=75)

gbm_model.fit(X_trn[features_to_include], Y_trn)



print('accuracy on training data',round(gbm_model.score(X_trn[features_to_include], Y_trn),2),'%')

print('accuracy on test data',round(gbm_model.score(X_tst[features_to_include], Y_tst),2),'%')



plot_training_vs_test(gbm_model.score(X_trn[features_to_include], Y_trn), gbm_model.score(X_tst[features_to_include], Y_tst))



df_feat_importances_gbm = pd.DataFrame(list(zip(features_to_include,gbm_model.feature_importances_)), columns=['Feature','Importance'])

df_feat_importances_gbm.sort_values(by='Importance', inplace=True)

plt.figure(figsize=[6,8])

plt.barh(df_feat_importances_gbm['Feature'],df_feat_importances_gbm['Importance'])



if(do_manual_test):

    preds = gbm_model.predict(manualtest[features_to_include])

    print(preds[0] == manualtest.dec[0])

    print('Prediction of manual data: ')

    print(preds[0])

    print('Actual: ')

    print(manualtest.dec[0])
from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import numpy as np



sc = StandardScaler()

X = sc.fit_transform(df.drop('dec', axis=1))

y = df['dec'].values

y_cat = to_categorical(y)





X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, y_cat, test_size=0.2)



from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



model = Sequential()

model.add(Dense(32, input_shape=(17,), activation='tanh'))

model.add(Dense(32, activation='tanh'))

model.add(Dense(32, activation='tanh'))

model.add(Dense(2, activation='softmax'))

model.compile(Adam(lr=0.05),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_trn, Y_trn, epochs=20, verbose=2, validation_split=0.1)





y_pred = model.predict(X_tst)



y_test_class = np.argmax(y_tst, axis=1)

y_pred_class = np.argmax(y_pred, axis=1)



from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



accuracy_score(y_test_class, y_pred_class)

print(classification_report(y_test_class, y_pred_class))

confusion_matrix(y_test_class, y_pred_class)