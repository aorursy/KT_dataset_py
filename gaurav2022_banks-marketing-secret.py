#importing required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
path = '../input/bank-marketing/bank-additional-full.csv'

df  = pd.read_csv(path, sep=';')

df.head()
df.info()
print(df.y.value_counts())



dfgrouped = df.groupby('y')
for type,data in dfgrouped:

  display(type)

  display(data)
def feature_perc(feature, groupby= 'yes'):



  count = dfgrouped.get_group(groupby)[feature].value_counts()

  total_count = df[feature].value_counts()[count.index]



  perc = (count/total_count)*100

  return perc 





def plot_barh(array,incrementer, bias,ax= None, text_color ='blue', palette_style = 'darkgrid',palette_color = 'RdBu'):



  sns.set_style(palette_style)

  sns.set_palette(palette_color)

    

  sns.barplot(x= array, y= array.index, ax=ax)

  #plt.barh(array.index, width = array.values, height = .5)

  plt.yticks(np.arange(len(array)))

  plt.xticks( range(0, round(max(array)) +bias, incrementer ))



  for index, value in enumerate(array.values):

    plt.text(value +.5, index, s= '{:.1f}%'.format(value), color = text_color)



  #plt.show()

  return plt
object_feature_list = list(df.dtypes[df.dtypes == 'object'].index)



for feature in object_feature_list[:-1]:

    

    feature_perct =  feature_perc(feature)

    plt.title('Success rate by {}'.format(feature))

    plot_barh(feature_perct.sort_values(ascending= False),5,10, text_color = 'blue')

    plt.show()
int_feature_list = list(df.dtypes[df.dtypes != 'object'].index)



for feature in int_feature_list:

    

    fig, ax = plt.subplots(1,2, figsize= (12,6))

    sns.boxplot(x='y', y=feature, data =df, showmeans=True, ax = ax[0])

    sns.distplot(df[feature], ax = ax[1], kde= False)

    ax[0].set_title('{} variation'.format(feature))

    plt.show()

#df.drop(df[df['duration']>4000].index,  inplace =True)
#df.drop( df[df.previous>100].index, inplace=True)
df['y'] = df.y.apply(lambda x:0 if x=='no' else 1)

display(df.head())
df2 = pd.get_dummies(df)

display(df2.head())
display(df2.corr()['y'].sort_values(ascending= False))
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
X_train, X_test, y_train, y_test = train_test_split(df2.drop('y',axis=1),

                                                    df2['y'],

                                                    test_size=.25, random_state = 42,

                                                    stratify= df2['y'])
stsc = StandardScaler()

s_X_train = stsc.fit_transform(X_train)

s_X_test = stsc.transform(X_test)
model = LogisticRegression()

#model = SVC()

#model = RandomForestClassifier()

model.fit(s_X_train,y_train)
model.score(s_X_train,y_train)
model.score(s_X_test,y_test)
confusion_matrix(y_train, model.predict(s_X_train))
print(classification_report(y_train, model.predict(s_X_train)))
confusion_matrix(y_test, model.predict(s_X_test))
print(classification_report(y_test, model.predict(s_X_test)))
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.pipeline import Pipeline
X_train, X_test, y_train, y_test  = train_test_split(

                                    df2.drop('y', axis=1), df2['y'], test_size=.25, random_state = 0, stratify= df2['y'])
#model = LogisticRegression()

#model = DecisionTreeClassifier()

model = RandomForestClassifier()

#model = SVC
steps = [('scaler', StandardScaler()),

         ('model', model)]

pipeline = Pipeline(steps)

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)

print(y_pred)
print('confusion matrix for training data:')

print(confusion_matrix(y_train, pipeline.predict(X_train)))

print('==============================')

print('confusion matrix for testing data:')

print(confusion_matrix(y_test, pipeline.predict(X_test)))
print('classification report for training data:')

print(classification_report(y_train, pipeline.predict(X_train)))

print('==============================')

print('classification report for testing data:')

print(classification_report(y_test, pipeline.predict(X_test)))
accuracy_score(y_test, pipeline.predict(X_test))