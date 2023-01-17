! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm



import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff



import dabl



from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score



plt.style.use("fivethirtyeight")

orange_black = [

    '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'

]
# Let's read the data now

data = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

data.head()
data.describe()
data.isna().sum()
data['status'].value_counts()
# Fill 0 in place of NuLL values

data['salary'] = data['salary'].fillna(0)

data.head()
# Before we do viz, first drop "sl_no" column.

data = data.drop(['sl_no'], axis=1)
targets = data['gender'].value_counts().tolist()

values = list(dict(data['gender'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Gender Value Pie-chart',

)

fig.show()
targets = data['status'].value_counts().tolist()

values = list(dict(data['status'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Status Value Distribution',

    color_discrete_sequence=["cyan", "blue"]

    

)

fig.show()
targets = data['specialisation'].value_counts().tolist()

values = list(dict(data['specialisation'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Spec. Value Distribution',

    color_discrete_sequence=orange_black

    

)

fig.show()
fig = px.histogram(

    data, x="mba_p",

    marginal="violin",

    hover_data=data.columns,

    color_discrete_sequence=["maroon"],

    title=f"MBA Percent Distribution [\u03BC : ~{data['mba_p'].mean():.2f}% | \u03C3 : ~{data['mba_p'].std():.2f} %]",

)



fig.show()
fig = px.histogram(

    data[data['salary']!=0], x="salary",

    marginal="violin",

    hover_data=data.columns,

    color_discrete_sequence=["magenta"],

    title=f"MBA Percent Distribution [\u03BC : ~{data['mba_p'].mean():.2f}% | \u03C3 : ~{data['mba_p'].std():.2f} %]",

)



fig.show()
targets = data['workex'].value_counts().tolist()

values = list(dict(data['workex'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Work Exp. Distribution',

    color_discrete_sequence=["gray", "black"]

    

)

fig.show()
fig = px.histogram(

    data, x="etest_p",

    marginal="box",

    hover_data=data.columns,

    color_discrete_sequence=["red"],

    title=f"Performance in Employability Test [\u03BC : ~{data['etest_p'].mean():.2f}% | \u03C3 : ~{data['etest_p'].std():.2f} %]",

)



fig.show()
targets = data['degree_t'].value_counts().tolist()

values = list(dict(data['degree_t'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title="Candidate's Degree Type Chart",

)

fig.show()
fig = px.histogram(

    data, x="degree_p",

    marginal="box",

    hover_data=data.columns,

    color_discrete_sequence=["green"],

    title=f"Attained Degree Percentage [\u03BC : ~{data['degree_p'].mean():.2f}% | \u03C3 : ~{data['degree_p'].std():.2f} %]",

)



fig.show()
sci_avg_pcent = data[data['degree_t'] == 'Sci&Tech']['degree_p'].mean()

com_avg_pcent = data[data['degree_t'] == 'Comm&Mgmt']['degree_p'].mean()

print(f"Average Percentage for Science & Technology Students is: {sci_avg_pcent:.2f}% while the average percentage of Commerce & Management Students is: {com_avg_pcent:.2f}%")
targets = data['hsc_s'].value_counts().tolist()

values = list(dict(data['hsc_s'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title="Higher Secondry Spec. Type Chart",

    color_discrete_sequence=["red", "blue", "green"]

)

fig.show()
targets = data['hsc_b'].value_counts().tolist()

values = list(dict(data['hsc_b'].value_counts()).keys())



fig = px.pie(

    values=targets,

    names=values,

    title="Higher Secondry Board Type Chart",

    color_discrete_sequence=["orange", "gold"]

)

fig.show()
fig = px.histogram(

    data, x="ssc_p",

    marginal="box",

    hover_data=data.columns,

    color_discrete_sequence=["blue"],

    title=f"Higher Secondry Percentage [\u03BC : ~{data['ssc_p'].mean():.2f}% | \u03C3 : ~{data['ssc_p'].std():.2f} %]",

)



fig.show()
# A few utility functions to encode categorical data



def get_category_names(df, column_name):

    '''

    Column passed must be categorical

    '''

    unique_names_dict = dict(df[column_name].value_counts())

    unique_names = list(unique_names_dict.keys())

    

    _length = len(unique_names)

    return (_length, unique_names)



def replace_small_categorical_data(df, column_name, categorical_names):

    """

    Categorical Encodes a data

    """

    copy_frame = df.copy(deep=True)

    

    copy_frame[column_name].replace(categorical_names, [x for x in range(len(categorical_names))], inplace=True)

    

    return copy_frame
to_encode = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

encoded_data = data.copy(deep=True)



for col in to_encode:

    _, current_category_names = get_category_names(encoded_data, col)

    encoded_data = replace_small_categorical_data(encoded_data, col, current_category_names)
encoded_data.head()
data.head()
sns.pairplot(data)
plt.figure(figsize=(15, 12))

sns.heatmap(encoded_data.corr(), annot=True)
plt.rcParams['figure.figsize'] = (18, 6)

dabl.plot(encoded_data, target_col = 'status')
plt.rcParams['figure.figsize'] = (18, 6)

dabl.plot(encoded_data, target_col = 'salary')
# First, let's split the data

split_pcent = 0.10

split = int(split_pcent * len(encoded_data))

encoded_data = encoded_data.sample(frac=1).reset_index(drop=True)



encoded_data = encoded_data.drop(['salary'], axis=1)



test = encoded_data[:split]

train = encoded_data[split:]



trainY = train['status'].values

trainX = train.drop(['status'], axis=1)



testY = test['status'].values

testX = test.drop(['status'], axis=1)
# Mean Normalise the data

trainX = (trainX - trainX.mean()) / trainX.std()

testX = (testX - testX.mean()) / testX.std()
# We are only using 11 Classifiers, you can use more if you wish.

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"]



classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]
# Let's do the classification and store the name of the classifier and it's test score into a dictionary



clf_results = {}



for name, clf in tqdm(zip(names, classifiers)):

    # Fit on the traning data

    clf.fit(trainX, trainY)

    

    # Get the test time prediction

    preds = clf.predict(testX)

    

    # Calculate Test ROC_AUC

    score = roc_auc_score(testY, preds)

    

    # Store the results in a dictionary

    clf_results[name] = score
# Sort the Model Accuracies based on the test score

sort_clf = dict(sorted(clf_results.items(), key=lambda x: x[1], reverse=True))



# Get the names and the corresponding scores

clf_names = list(sort_clf.keys())[::-1]

clf_scores = list(sort_clf.values())[::-1]
# Plot the per-model performance

fig = px.bar(

    x=clf_scores,

    y=clf_names,

    color=clf_names,

    labels={'x':'Test ROC-AUC Score', 'y':'Models'},

    title=f"Model Performance [ Best Model: {clf_names[-1]} | Score: {clf_scores[-1]} ]"

)



fig.show()
clf_names