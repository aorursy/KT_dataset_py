import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

pd.options.display.float_format = '{:,.4f}'.format

sns.set(style="whitegrid")
df = pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

print("Shape of DataSet:", df.shape[0], 'rows |', df.shape[1], 'columns')

df.head()
df.info()
df.describe().T
def eda_categ_feat_desc_plot(series_categorical, title = ""):

    """Generate 2 plots: barplot with quantity and pieplot with percentage. 

       @series_categorical: categorical series

       @title: optional

    """

    series_name = series_categorical.name

    val_counts = series_categorical.value_counts()

    val_counts.name = 'quantity'

    val_percentage = series_categorical.value_counts(normalize=True)

    val_percentage.name = "percentage"

    val_concat = pd.concat([val_counts, val_percentage], axis = 1)

    val_concat.reset_index(level=0, inplace=True)

    val_concat = val_concat.rename( columns = {'index': series_name} )

    

    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)

    if(title != ""):

        fig.suptitle(title, fontsize=18)

        fig.subplots_adjust(top=0.8)



    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])

    for index, row in val_concat.iterrows():

        s.text(row.name, row['quantity'], row['quantity'], color='black', ha="center")



    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),

                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],

                             title="Percentage Plot")



    ax[1].set_ylabel('')

    ax[0].set_title('Quantity Plot')



    plt.show()
def eda_horiz_plot(df, x, y, title, figsize = (8,5), palette="Blues_d", formating="int"):

    """Using Seaborn, plot horizonal Bar with labels

    !!! Is recomend sort_values(by, ascending) before passing dataframe

    !!! pass few values, not much than 20 is recommended

    """

    f, ax = plt.subplots(figsize=figsize)

    sns.barplot(x=x, y=y, data=df, palette=palette)

    ax.set_title(title)

    for p in ax.patches:

        width = p.get_width()

        if(formating == "int"):

            text = int(width)

        else:

            text = '{.2f}'.format(width)

        ax.text(width + 1, p.get_y() + p.get_height() / 2, text, ha = 'left', va = 'center')

    plt.show()
def eda_categ_feat_desc_df(series_categorical):

    """Generate DataFrame with quantity and percentage of categorical series

    @series_categorical = categorical series

    """

    series_name = series_categorical.name

    val_counts = series_categorical.value_counts()

    val_counts.name = 'quantity'

    val_percentage = series_categorical.value_counts(normalize=True)

    val_percentage.name = "percentage"

    val_concat = pd.concat([val_counts, val_percentage], axis = 1)

    val_concat.reset_index(level=0, inplace=True)

    val_concat = val_concat.rename( columns = {'index': series_name} )

    return val_concat
def eda_numerical_feat(series, title="", with_label=True, number_format=""):

    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)

    print(series.describe())

    if(title != ""):

        f.suptitle(title, fontsize=18)

    sns.distplot(series, ax=ax1)

    sns.boxplot(series, ax=ax2)

    if(with_label):

        describe = series.describe()

        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 

              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],

              'Q3': describe.loc['75%']}

        if(number_format != ""):

            for k, v in labels.items():

                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',

                         size=8, color='white', bbox=dict(facecolor='#445A64'))

        else:

            for k, v in labels.items():

                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',

                     size=8, color='white', bbox=dict(facecolor='#445A64'))

    plt.show()
def check_balanced_train_test_binary(x_train, y_train, x_test, y_test, original_size, labels):

    """ To binary classification

    each paramethes is pandas.core.frame.DataFrame

    @total_size = len(X) before split

    @labels = labels in ordem [0,1 ...]

    """

    train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)

    test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)



    prop_train = train_counts_label/ len(y_train)

    prop_test = test_counts_label/ len(y_test)



    print("Original Size:", '{:,d}'.format(original_size))

    print("\nTrain: must be 80% of dataset:\n", 

          "the train dataset has {:,d} rows".format(len(x_train)),

          'this is ({:.2%}) of original dataset'.format(len(x_train)/original_size),

                "\n => Classe 0 ({}):".format(labels[0]), train_counts_label[0], '({:.2%})'.format(prop_train[0]), 

                "\n => Classe 1 ({}):".format(labels[1]), train_counts_label[1], '({:.2%})'.format(prop_train[1]),

          "\n\nTest: must be 20% of dataset:\n",

          "the test dataset has {:,d} rows".format(len(x_test)),

          'this is ({:.2%}) of original dataset'.format(len(x_test)/original_size),

                  "\n => Classe 0 ({}):".format(labels[0]), test_counts_label[0], '({:.2%})'.format(prop_test[0]),

                  "\n => Classe 1 ({}):".format(labels[1]),test_counts_label[1], '({:.2%})'.format(prop_test[1])

         )
def class_report(y_target, y_preds, name="", labels=None):

    if(name != ''):

        print(name,"\n")

    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred, target_names=labels))
print("No Missing Data:")



print("\t", df.isnull().sum().max(), "invalid Data")
# No duplicate Rows

print("Check duplicated rows")

print("\t", df.duplicated(subset=None, keep='first').sum(), 'rows Duplicates')
df['PassengerId'].value_counts()
df.query('PassengerId in [463, 767]')
eda_categ_feat_desc_plot(df['Country'], 'Analysis "Contry" Distribution')
eda_categ_feat_desc_df(df['Country'])
df_names = eda_categ_feat_desc_df(df['Firstname'])

print("unique first names: ", df_names.shape[0] )

df_names.head()
df_names = eda_categ_feat_desc_df(df['Lastname'])

print("unique last names: ", df_names.shape[0] )

df_names.head()
eda_categ_feat_desc_plot(df['Sex'], 'Analysis "Sex" Distribution')
eda_numerical_feat(df['Age'], '"Age" Distribution', with_label=True)
# Min Ages and Max Ages

top_int = 5



pd.concat([df.sort_values(by="Age").head(top_int), df.sort_values(by="Age").tail(top_int)])
eda_categ_feat_desc_plot(df['Category'], 'Analysis "Category" Distribution')
eda_categ_feat_desc_plot(df['Survived'], 'Analysis "Survived" Distribution')
print(list(df.columns))
sns.pairplot(df, hue="Survived", corner=True)
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 5), sharex=False)

f.suptitle('Age x Survived', fontsize=18)



sns.boxplot(y="Age", x="Survived", data=df, ax=ax1)

sns.violinplot(y="Age", x="Survived", data=df, ax=ax2)

ax1.set_title("BoxPlot")

ax2.set_title("ViolinPlot")



plt.show()
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 5), sharex=False)

f.suptitle('Age x Survived by Sex', fontsize=18)



sns.boxplot(y="Age", x="Survived", hue="Sex", data=df, ax=ax1)

sns.violinplot(y="Age", x="Survived", hue="Sex", data=df, ax=ax2)

ax1.set_title("BoxPlot")

ax2.set_title("ViolinPlot")



plt.show()
f, ax1 = plt.subplots(figsize=(10, 5))



sns.countplot(x="Sex", hue="Survived", data=df, ax=ax1)

ax1.set_title("Sex x Survived", size=15)



plt.show()
df1 = df.groupby(["Survived","Sex"]).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)

df1
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)

f.suptitle('Percentage of Survivors by Sex', fontsize=18)



alist = df1['Quantity'].tolist()



df1.query('Sex == "M"').plot.pie(y='Quantity', figsize=(10, 5), autopct='%1.2f%%', 

                                 labels = ['Not Survived = ' + str(alist[1]), 'Survived = ' + str(alist[3]) ],

                                 title="% of male survivors " + "(Total = " + str(alist[1] + alist[3]) + ")",

                                 ax=ax1, labeldistance=None)



df1.query('Sex == "F"').plot.pie(y='Quantity', figsize=(10, 5), autopct='%1.2f%%', 

                                 labels = ['Not Survived = ' + str(alist[0]), 'Survived =' + str(alist[2]) ],

                                title="% of female survivors " + "(Total = " + str(alist[0] + alist[2]) + ")", 

                                 ax=ax2, labeldistance=None)



plt.show()
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)

f.suptitle('Percentage of Sex that Survivors', fontsize=18)



alist = df1['Quantity'].tolist()



df1.query('Survived == 0').plot.pie(y='Quantity', figsize=(10, 5), autopct='%1.2f%%',

                                    labels = ['Female = ' + str(alist[0]), 'Male = ' + str(alist[1]) ],

                                    title="% to sex of deaths " + "(Total = " + str(alist[0] + alist[1]) + ")",

                                    ax=ax1, labeldistance=None)



df1.query('Survived == 1').plot.pie(y='Quantity', figsize=(10, 5), autopct='%1.2f%%',

                                    labels = ['Female = ' + str(alist[2]), 'Male = ' + str(alist[3]) ],

                                    title="% to sex of survivors "  + "(Total = " + str(alist[2] + alist[3]) + ")",

                                    ax=ax2, labeldistance=None)



plt.show()
df2 = df.groupby(["Survived","Category"]).count().reset_index().rename({"Firstname": "Quantity"}, axis=1)

df2
f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)

f.suptitle('Percentage of Categories that Survived', fontsize=18)



alist = df2['Quantity'].tolist()



df2.query('Category == "P"').plot.pie(y='Quantity', figsize=(10, 5), autopct='%1.2f%%',

                                      labels = ['Not Survived = ' + str(alist[1]), 'Survived = ' + str(alist[3]) ],

                                      title="% of class P that survived " + "(Total = " + str(alist[1] + alist[3]) + ")",

                                      ax=ax1, labeldistance=None)



df2.query('Category == "C"').plot.pie(y='Quantity', figsize=(10, 5), autopct='%1.2f%%',

                                      labels = ['Not Survived = ' + str(alist[0]), 'Survived = ' + str(alist[2]) ],

                                      title="% of class C that survived " + "(Total = " + str(alist[0] + alist[2]) + ")",

                                      ax=ax2, labeldistance=None)



plt.show()
replace_list = {"Sex":{"M":0,"F":1}, "Category":{"P":0,"C":1}}



df_corr = df.replace(replace_list).drop(['PassengerId', 'Country','Firstname','Lastname'], axis = 1)

df_corr.head(3)
corr = df_corr.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



f, ax1 = plt.subplots(figsize=(8,6))

sns.heatmap(corr, cmap='coolwarm_r', 

            annot=True, annot_kws={'size':15}, ax=ax1, mask=mask)



ax1.set_title("Correlation", fontsize=14)



plt.show()
eda_categ_feat_desc_plot(df['Survived'], 'Remember, is a Unbalanced DataSet: Survived Rate')
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
# x = df.loc[:,["Sex","Age","Category"]]

# y = df.loc[:,["Survived"]]



# # OneHotEncoding

# x = pd.get_dummies(x)



# # Scaling Age

# sc = StandardScaler()  # Normal Distribution: mean 0, min -1 and max 1

# x['Age'] = sc.fit_transform(x['Age'].values.reshape(-1,1))



# x.head(3)
x = df.loc[:,["Sex","Age","Category","Country"]]

y = df.loc[:,["Survived"]]



x.head(1)
# Separate Country in Estonia, Sweden and Others

x['Country'] = x['Country'].apply(lambda x: 'Estonia' if x == 'Estonia' 

                                  else ('Sweden' if x == 'Sweden' else 'Others'))



sc = StandardScaler()  # Normal Distribution: mean 0, min -1 and max 1

x['Age'] = sc.fit_transform(x['Age'].values.reshape(-1,1))



x = pd.get_dummies(x, drop_first=True) # Remove EstoniaColumn



x.head(3)
# x = df.loc[:,["Sex","Age","Category","Country", "Lastname"]]

# y = df.loc[:,["Survived"]]



# x['Country']= [1 if el =='Estonia' or el =='Sweden' else 0 for el in x['Country']] 



# sc = StandardScaler()  

# x['Age'] = sc.fit_transform(x['Age'].values.reshape(-1,1))



# encode = LabelEncoder()

# x['Sex']=encode.fit_transform(x['Sex'])

# x['Lastname']=encode.fit_transform(x['Lastname'])

# x['Category'] =encode.fit_transform(x['Category'])



# x.head(3)
from sklearn.model_selection import KFold, StratifiedKFold



kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)



for train_index, test_index in kfold.split(x, y):

    x_train, x_test = x.iloc[train_index].values, x.iloc[test_index].values

    y_train, y_test = y.iloc[train_index].values, y.iloc[test_index].values



check_balanced_train_test_binary(x_train, y_train, x_test, y_test, len(df), ['Death', 'Survives'])
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE

from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss

from imblearn.combine import SMOTEENN, SMOTETomek # over and under sampling

from imblearn.metrics import classification_report_imbalanced



imb_models = {

    'ADASYN': ADASYN(),

    'SMOTE': SMOTE(random_state=42),

    'SMOTEENN': SMOTEENN("minority", random_state=42),

    'SMOTETomek': SMOTETomek(tomek=TomekLinks(sampling_strategy='majority')),

    'RandomUnderSampler': RandomUnderSampler()

}



imb_strategy = "SMOTE"



if(imb_strategy != "None"):

    print("train dataset before", x_train.shape[0])

    print("imb_strategy:", imb_strategy)



    imb_tranformer = imb_models[imb_strategy]

    

    # x_train, y_train | xsm_train, ysm_train

    xsm_train, ysm_train = imb_tranformer.fit_sample(x_train, y_train)



    print("train dataset before", xsm_train.shape[0],

          'generate', xsm_train.shape[0] - x_train.shape[0] )

else:

    print("Dont correct unbalanced dataset")
# use: x_train, y_train, x_test, y_test



# Classifier Libraries

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC



# Ensemble Classifiers

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



# Others Linear Classifiers

from sklearn.linear_model import SGDClassifier, RidgeClassifier

from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier



# xboost

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



# scores

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score



# neural net of sklearn

from sklearn.neural_network import MLPClassifier



# others

import time

import operator
# def neural nets

mlp = MLPClassifier(verbose = False, max_iter=1000, tol = 0.000010,

                    solver = 'adam', hidden_layer_sizes=(100), activation='relu')



# def classifiers



nn_classifiers = {

    "Multi Layer Perceptron": mlp

}



linear_classifiers = {

    "SGDC": SGDClassifier(),

    "Ridge": RidgeClassifier(),

    "Perceptron": Perceptron(),

    "PassiveAggressive": PassiveAggressiveClassifier()

}



gboost_classifiers = {

    "XGBoost": XGBClassifier(),

    "LightGB": LGBMClassifier(),

}



classifiers = {

    "Naive Bayes": GaussianNB(),

    "Logisitic Regression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Machine": SVC(),

    "Decision Tree": DecisionTreeClassifier()

}



ensemble_classifiers = {

    "AdaBoost": AdaBoostClassifier(),

    "GBoost": GradientBoostingClassifier(),

    "Bagging": BaggingClassifier(),

    "Random Forest": RandomForestClassifier(),

    "Extra Trees": ExtraTreesClassifier()    

}



all_classifiers = {

    "Simple Models": classifiers,

    "Ensemble Models": ensemble_classifiers,

    "GBoost Models": gboost_classifiers,

    "NeuralNet Models": nn_classifiers,

    "Others Linear Models": linear_classifiers,

}



metrics = {

    'cv_scores': {},

    'acc_scores': {},

    'f1_mean_scores': {},

}
format_float = "{:.4f}"



is_print = False # True/False



time_start = time.time()



print("Fit Many Classifiers")



for key, classifiers in all_classifiers.items():

    if (is_print):

        print("\n{}\n".format(key))

    for key, classifier in classifiers.items():

        t0 = time.time()

        # xsm_train, ysm_train || x_train, y_train

        classifier.fit(x_train, y_train) 

        t1 = time.time()

        # xsm_train, ysm_train || x_train, y_train

        training_score = cross_val_score(classifier, x_train, y_train, cv=5) 

        y_pred = classifier.predict(x_test)

        cv_score = round(training_score.mean(), 4) * 100

        acc_score = accuracy_score(y_test, y_pred)

        f1_mean_score = f1_score(y_test, y_pred, average="macro") # average =  'macro' or 'weighted'

        if (is_print):

            print(key, "\n\tHas a training score of", 

                  cv_score, "% accuracy score on CrossVal with 5 cv ")

            print("\tTesting:")

            print("\tAccuracy in Test:", format_float.format(acc_score))

            print("\tF1-mean Score:", format_float.format(f1_mean_score)) 

            print("\t\tTime: The fit time took {:.2} s".format(t1 - t0), '\n')

        metrics['cv_scores'][key] = cv_score

        metrics['acc_scores'][key] = acc_score

        metrics['f1_mean_scores'][key] = f1_mean_score

        

time_end = time.time()

        

print("\nDone in {:.5} s".format(time_end - time_start), '\n')

        

print("Best cv score:", max( metrics['cv_scores'].items(), key=operator.itemgetter(1) ))

print("Best Accuracy score:", max( metrics['acc_scores'].items(), key=operator.itemgetter(1) ))

print("Best F1 score:", max( metrics['f1_mean_scores'].items(), key=operator.itemgetter(1) ))



lists = [list(metrics['cv_scores'].values()),

         list(metrics['acc_scores'].values()),

         list(metrics['f1_mean_scores'].values())

        ]



a_columns = list(metrics['cv_scores'].keys())



df_metrics = pd.DataFrame(lists , columns = a_columns,

                    index = ['cv_scores', 'acc_scores', 'f1_scores'] )
lists = [list(metrics['cv_scores'].values()),

         list(metrics['acc_scores'].values()),

         list(metrics['f1_mean_scores'].values())

        ]



a_columns = list(metrics['cv_scores'].keys())



dfre1 = pd.DataFrame(lists , columns = a_columns,

                    index = ['cv_scores', 'acc_scores', 'f1_scores'] )



dfre1 = dfre1.T.sort_values(by="acc_scores", ascending=False)

dfre1
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# AdaBoost

adaB = AdaBoostClassifier()

adaB.fit(x_train,y_train)

y_pred = adaB.predict(x_test)



print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=['Death', 'Survives']))
from mlens.ensemble import SuperLearner



# Define Models List

models = list()

models.append(LogisticRegression(solver='liblinear'))

models.append(DecisionTreeClassifier())

models.append(SVC(kernel='linear'))

models.append(GaussianNB())

models.append(KNeighborsClassifier())

models.append(AdaBoostClassifier())    

models.append(BaggingClassifier(n_estimators=100))

models.append(RandomForestClassifier(n_estimators=100))

models.append(ExtraTreesClassifier(n_estimators=100))

models.append(XGBClassifier(scale_pos_weight=2))



# Create Super Model

ensemble = SuperLearner(scorer=accuracy_score, folds=10, shuffle=False, sample_size=len(x))

ensemble.add(models)

ensemble.add_meta(DecisionTreeClassifier()) # can change



# Fit 

ensemble.fit(x_train, y_train)

print(ensemble.data)



# Pred

y_pred = ensemble.predict(x_test)



# Evaluate

class_report(y_test, y_pred, name='SuperLeaner', labels=['Death', 'Survives'])



# One time out: acc: 0.89, f1: 0.59
import pickle

Pkl_Filename = "Pickle_Model.pkl"  



# Save the Modle to file in the current working directory

with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(ensemble, file)

    

# Load the Model back from file

with open(Pkl_Filename, 'rb') as file:  

    Pickled_ensemble = pickle.load(file)



Pickled_ensemble