import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
job_data = pd.read_csv("/kaggle/input/jobposting_revised.csv")



# Ignore the warnings 

import warnings

warnings.filterwarnings('ignore')
job_data.head()
job_data.info()
job_data.drop_duplicates()
# Missing Values

job_data.isnull().sum().sort_values(ascending=False)
# I chose to drop company_revenue and company_scale because they are not related with the issues I want to discuss.

job_data = job_data.drop(['company_revenue','company_scale'], axis=1)
job_data = job_data.dropna(subset = ['location','skill','description','company_name','experience'])
job_data.isnull().sum().sort_values(ascending=False)
clean_data = job_data.drop(['title','company_type','company_name','skill','NC','CA','NY','VA','TX','MA','IL','WA','MD','DC','other_states','description','location'], axis=1)
clean_data.info()
clean_data.describe().T
print(clean_data['salary'].value_counts())

print(clean_data['experience'].value_counts())

print(clean_data['job_type'].value_counts())
# Use map to transfer String type to integer

salary_mapping = {"<80000": 70000,"80000-99999": 90000, "100000-119999": 110000, "120000-139999": 130000,"140000-159999":150000,">160000":170000}

clean_data['salary'] = clean_data['salary'].map(salary_mapping)

job_data['salary'] = job_data['salary'].map(salary_mapping)

experience_mapping = {"entry_level": 1,"mid_level": 2, "senior_level": 3}

clean_data['experience'] = clean_data['experience'].map(experience_mapping)

plot_data = clean_data.copy()
type_mapping = {"data_scientist": 1,"data_analyst": 2, "data_engineer": 3}

clean_data['job_type'] = clean_data['job_type'].map(type_mapping)

clean_data.head(3)
plt.subplots(figsize=(15,15))

ax = plt.axes()

ax.set_title("Job Post Analysis Correlation Heatmap")

corr = clean_data.corr()

sns.heatmap(corr, square = True, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot = True)
pair = clean_data[['experience','num_skills','salary','job_type','sql','r','machine_learning','python']]

sns.pairplot(pair, kind="scatter", hue="job_type", markers=["o", "s", "D"], palette="Set2")
import statsmodels.formula.api as smf

results = smf.ols('salary ~experience', data=clean_data).fit()

print(results.summary())
#graph distribution of qualitative data: Job Type

plt.subplots(figsize=(8,6))

axis1 = sns.barplot(x = 'experience', y = 'salary', hue = 'job_type', data=job_data)

axis1.set_title('Experience vs Job Type Salary Comparison')
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(14,12))



#how does experience factor with salary compare

sns.pointplot(x="experience", y="salary", hue="job_type",markers=["o", "s", "D"], palette="Set2",data=clean_data, ax = maxis1)



#how does num_skills factor with salary compare

sns.pointplot(x="num_skills", y="salary", hue="job_type",markers=["o", "s", "D"], palette="Set2",data=clean_data, ax = maxis2)
loc = pd.DataFrame(job_data['location'].value_counts())

cities_plot = loc.head(15).plot(kind = 'bar', width = .7, figsize = (10, 6), rot=45, title='States With The Most job Opptunities')
job_perstate = pd.DataFrame({'State':job_data['location'].value_counts().index, 'Counts':job_data['location'].value_counts().values})

job_perstate.head()
import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()

# plotly code for choropleth map

scale = [[0, 'rgb(229, 239, 245)'],[1, 'rgb(1, 97, 156)']]

data1 = [ dict(

        type = 'choropleth',

        colorscale = scale,

        autocolorscale = False,

        showscale = False,

        locations = job_perstate['State'],

        z = job_perstate['Counts'],

        locationmode = 'USA-states',

        marker = dict(

            line = dict (

                color = 'rgb(255, 255, 255)',

                width = 2

            ) ),

        ) ]

layout = dict(

        title = 'Data Related Jobs in United States by State',

        geo = dict(

            scope = 'usa',

            projection = dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)',

            countrycolor = 'rgb(255, 255, 255)')        

             )

 

figure = dict(data=data1, layout=layout)

iplot(figure)
job_count = job_data

job_count.replace(0,np.nan, inplace = True)

c_py=job_count['python'].value_counts().values

c_r=job_count['r'].value_counts().values

c_sql =job_count['sql'].value_counts().values

c_ml=job_count['machine_learning'].value_counts().values

c_h=job_count['hadoop'].value_counts().values

c_t =job_count['tableau'].value_counts().values

c_s=job_count['sas'].value_counts().values

c_sp=job_count['spark'].value_counts().values

c_j =job_count['java'].value_counts().values

c_bd =job_count['big_data'].value_counts().values

c_dm =job_count['data_mining'].value_counts().values

c_st =job_count['stat'].value_counts().values

c_tf =job_count['tensorflow'].value_counts().values

d = {'Skill': ['python', 'r','sql','machine_learning','hadoop','tableau','sas','spark','java','big_data','data_mining','stat','tensorflow'], 

     'Counts': [c_py.item(0), c_r.item(0),c_sql.item(0),c_ml.item(0),c_h.item(0),c_t.item(0),c_s.item(0),c_sp.item(0),c_j.item(0),c_bd.item(0),c_dm.item(0),c_st.item(0),c_tf.item(0)]}

skills = pd.DataFrame(data=d)





plt.rcdefaults()

fig, ax = plt.subplots()

# Example data

sk = 'python', 'r','sql','machine learning','hadoop','tableau','sas','spark','java','big_data','data_mining','stat','tensorflow'



ax.barh(skills['Skill'], skills['Counts'])

ax.set_yticks(skills['Skill'])

ax.set_yticklabels(sk)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Counts of Skills in Job Post')

ax.set_title('What data-related skills do job seeker need to pocess?')

plt.show()
job_data.dropna()

job_data['company_type'].describe()
com_type = pd.DataFrame({'Company_type':job_data['company_type'].value_counts().index, 'Counts':job_data['company_type'].value_counts().values})

com_type.head()
plt.rcdefaults()

fig, ax = plt.subplots()

ax.barh(com_type['Company_type'].head(), com_type['Counts'].head())

ax.set_yticks(com_type['Company_type'].head())

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Counts of company types appeared in Job Post')

ax.set_title('What type of company want to hire more data analysts?')

plt.show()
clean_data.info()
clean_data.info()
x = clean_data[['salary','experience','num_skills','python','sql','machine_learning','r','hadoop','tableau','sas','spark','java','stat','tensorflow','big_data','data_analysis','data_mining','natural_language_processing']]

y = clean_data['job_type']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
def build_eval_clf(clf,x_train,testX,y_test):

    clf.fit(x_train,y_train)

    #accuracy

    y_predict = clf.predict(testX)

    acc = sklearn.metrics.accuracy_score(y_test,y_predict)

    #f1_score

    f1 = sklearn.metrics.f1_score(y_test,y_predict,average='weighted')

    #precision

    p = sklearn.metrics.precision_score(y_test,y_predict,average='weighted')

    print('accuracy score is ' + str(acc))

    print('f1 score is ' + str(f1))

    print('precision score is ' + str(p))
import sklearn.metrics

from sklearn.linear_model import LogisticRegression

clf_lg = LogisticRegression()

build_eval_clf(clf_lg,x_train,x_train,y_train)
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier()

build_eval_clf(clf_knn,x_train,x_train,y_train)
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()

build_eval_clf(clf_dt,x_train,x_train,y_train)
from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()

build_eval_clf(clf_nb,x_train,x_train,y_train)
from sklearn.svm import LinearSVC

clf_svm = LinearSVC()

build_eval_clf(clf_svm,x_train,x_train,y_train)
from sklearn.ensemble import RandomForestClassifier

clf_ec = RandomForestClassifier()

build_eval_clf(clf_ec,x_train,x_train,y_train)
build_eval_clf(clf_lg,x_train,x_test,y_test)
build_eval_clf(clf_knn,x_train,x_test,y_test)
build_eval_clf(clf_dt,x_train,x_test,y_test)
build_eval_clf(clf_nb,x_train,x_test,y_test)
build_eval_clf(clf_svm,x_train,x_test,y_test)
build_eval_clf(clf_ec,x_train,x_test,y_test)
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()

param_grid = { 

    'n_estimators': [5, 100],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}

gsearch = GridSearchCV(rf,param_grid = param_grid, cv=5)

gsearch.fit(x_train, y_train)
gsearch.best_params_
rfc1=RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 100, max_depth=8, criterion='gini')
rfc1.fit(x_train, y_train)
pred=rfc1.predict(x_test)
build_eval_clf(rfc1,x_train,x_test,y_test)
plot_data = plot_data.drop(['salary','experience','num_skills','other_skills'], axis=1)

plot_data = plot_data.groupby(['job_type']).mean()

plot_data.head()
import plotly.graph_objs as go

plt_cols = ['python','sql', 'machine_learning', 'r', 'hadoop', 'tableau','sas', 'spark', 'java', 'big_data', 'data_mining',

       'data_analysis', 'stat', 'natural_language_processing']

plot_data.reset_index(inplace=True)

plt_data = [] 

for i in range(plot_data.shape[0]):

    trace = go.Scatterpolar(

        r = plot_data.loc[i,plt_cols],

        theta = plt_cols,

        name = plot_data.loc[i,'job_type'],

    )

    plt_data.append(trace)

    

layout = go.Layout(

  polar = dict(

    radialaxis = dict(

      visible = True,

      range = [0, 1],

    )

  ),

    height = 700,

    width = 700,

    title = "Needing Skills for Data Analyst/Engineer/Scientist",

    showlegend = True

)



fig = go.Figure(data=plt_data, layout=layout)

iplot(fig)