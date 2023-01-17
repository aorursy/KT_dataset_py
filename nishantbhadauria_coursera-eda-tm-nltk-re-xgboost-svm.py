import pandas as pd

cs=pd.read_csv(r'/kaggle/input/coursera-course-dataset/coursea_data.csv')
cs.describe(include= 'all')
!pip install langdetect

import langdetect

cs['Language'] = cs['course_title'].apply(lambda x: langdetect.detect(x))

cs['Language'].value_counts()
cs['enrolled'] = cs['course_students_enrolled'].map(lambda x: str(x)[:-1])

cs["enrolled"] = pd.to_numeric(cs["enrolled"])
cs["enrolled"]=cs["enrolled"]*1000
cs["enrolled"].describe()
import plotly.express as px

fig = px.pie(cs, values='enrolled', names='course_difficulty')

fig.update_traces(textposition='inside')

fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

fig.show()
import plotly.express as px

fig = px.scatter(cs, x="course_rating", y="enrolled", color="course_difficulty",

                 size='course_rating', hover_data=['enrolled'])

fig.show()

import plotly.express as px

fig = px.box(cs, x="course_difficulty", y="enrolled",color="Language")

fig.update_traces(quartilemethod="exclusive")

fig.show()

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

from wordcloud import WordCloud

cs['course_title_new'] = cs['course_title'].map(lambda x: x.split())

cs['course_title_new']=cs['course_title_new'].apply(lambda x: [item for item in x if item not in stop_words])

cs['course_title_new']=cs['course_title_new'].astype(str)
title_count = ','.join(list(cs['course_title_new'].values))
wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')

wordcloud.generate(title_count)

wordcloud.to_image()
cs["python"]= cs["course_title"].str.find("Python") 
cs.loc[cs['python'] == -1, 'python_yes'] = 0

cs.loc[cs['python'] > -1, 'python_yes'] = 1
unv_python=cs.groupby(['course_organization'],as_index=False).python_yes.sum()

unv_python
unv_python.sort_values(by='python_yes',ascending=False)
import re

patterns=['AI','Artificial Intelligence','Machine Learning','Data Science','Analytics','Neural Networks','Random Forest',

          'Deep Learning','Reinforcement Learning','Pattern Recognition','Feature Engineering','Kaggle','Data Visualization']

ultimate_pattern = '|'.join(patterns)

def Clean_names(x):

    if re.search(ultimate_pattern, x):

        return 1

    else: 

        return 0

cs['ML_yes'] = cs['course_title'].apply(Clean_names) 

cs['ML_yes'].sum()
unv_ml=cs.groupby(['course_organization'],as_index=False).ML_yes.sum()

unv_ml.sort_values(by='ML_yes',ascending=False)
ds_cs=cs[cs['ML_yes']==1]
import plotly.express as px

fig = px.bar(ds_cs, x="course_organization", y="enrolled",color="Language")

fig.show()

import plotly.express as px

fig = px.box(ds_cs, x="course_organization", y="course_rating",color="course_difficulty")

fig.update_traces(quartilemethod="exclusive")

fig.show()

unv_sp=cs.groupby(['course_organization','course_Certificate_type'],as_index=False).course_rating.mean()

unv_sp.sort_values(by='course_rating',ascending=False)
import re

patterns2=['Business','Management','Leadership','Finance','Accounts','Consulting','Administration']

ultimate_pattern2 = '|'.join(patterns2)

def Clean_names(x):

    if re.search(ultimate_pattern2, x):

        return 1

    else: 

        return 0

cs['B_yes'] = cs['course_title'].apply(Clean_names) 
B_cs=cs[cs['B_yes']==1]
import plotly.express as px

fig = px.bar(B_cs, x="course_organization", y="enrolled",color="Language")

fig.show()

!pip install distance

import distance

jd = lambda x, y: 1 - distance.jaccard(x, y)

sim_unv_course=ds_cs['course_organization'].apply(lambda x: ds_cs['course_title'].apply(lambda y: jd(x, y)))
sim_unv_course
import numpy as np

(sim_unv_course.values).trace()
cs['enrolled_cat'] = pd.cut(cs['enrolled'].astype(int), 5)
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

cs['enrolled_cat']=cs['enrolled_cat'].apply(LabelEncoder().fit_transform)

import plotly.express as px

fig = px.box(cs, x="enrolled_cat", y="course_rating",color="course_difficulty")

fig.update_traces(quartilemethod="exclusive")

fig.show()

cs_sub=cs[['course_Certificate_type','course_difficulty','enrolled_cat']]
cs_sub
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

cs_sub=cs_sub.apply(LabelEncoder().fit_transform)

cs_sub
import xgboost as xgb

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(cs_sub, cs['course_rating'], test_size=0.2)


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)

model = xg_reg.fit(X_train, Y_train)

import numpy as np

from sklearn.metrics import mean_squared_error



preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, preds))

print("RMSE: %f" % (rmse))
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
import matplotlib.pyplot as plt



xgb.plot_tree(xg_reg,num_trees=0)

plt.rcParams['figure.figsize'] = [10, 6]

plt.show()
dtrain = xgb.DMatrix(X_train,Y_train)

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}

xg_reg = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)
import matplotlib.pyplot as plt



xgb.plot_tree(xg_reg,num_trees=9)

plt.rcParams['figure.figsize'] = [50, 10]

plt.show()
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))

print("RMSE: %f" % (rmse))
error=y_pred-Y_test
error
min(error),max(error),plt.hist(error)