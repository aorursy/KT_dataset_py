%matplotlib inline

import numpy

import numpy.linalg

import pandas 

import sklearn

import sklearn.feature_selection

import sklearn.ensemble

import sklearn.metrics

import random

import seaborn

import scipy.spatial.distance



data = pandas.read_csv('/kaggle/input/healthcare-dataset-stroke-data/train_2v.csv',

                      index_col='id')

data
unique_values = {column:data[column].unique()

                 for column in ('gender','ever_married','work_type','Residence_type','smoking_status')}

unique_values
for (key,values) in unique_values.items():

    for (i,value) in enumerate(values):

        data.loc[data[key]==value,key] = i

data=data.fillna(0)

data
MI=pandas.DataFrame(numpy.zeros((data.shape[1],data.shape[1])),

                    index = data.columns,

                    columns = data.columns)

continuous = ('age','avg_glucose_level','bmi')

for (i,column) in enumerate(data.columns[:-1]):

    later = data.columns[i+1:]

    H = (sklearn.feature_selection.mutual_info_regression 

                                 if column in continuous 

                                 else sklearn.feature_selection.mutual_info_classif)(data[later],data[column])

    MI.loc[column,later] = H

    MI.loc[later,column] = H

seaborn.heatmap(MI)
MI['stroke'].plot.bar()
data.plot.scatter('age','avg_glucose_level')
data['stroke'].value_counts().plot.bar()
negative = data[data['stroke']==0].index.values.tolist()

positive = data[data['stroke']==1].index.values.tolist()

training_sample = random.sample(negative,(len(negative)*7)//10)+random.sample(positive,(len(positive)*7)//10)

test_sample = [n for n in data.index.values if n not in training_sample]

cols = [column for column in data.columns if column!='stroke']



model = sklearn.ensemble.RandomForestClassifier(n_estimators=100,

                                                n_jobs=-1,

                                                class_weight='balanced')

model.fit(data.loc[training_sample,cols].values,data.loc[training_sample,'stroke'].values)

predictions = model.predict(data.loc[test_sample,cols].values)

confusion = sklearn.metrics.confusion_matrix(data.loc[test_sample,'stroke'].values,predictions)

seaborn.heatmap(confusion)
def precision(cm):

    return cm[1,1]/cm[:,1].sum()



def recall(cm):

    return cm[1,1]/cm[1].sum()



def accuracy(cm):

    return (cm[0,0]+cm[1,1])/cm.sum()



def matthews(cm):

    return (cm[0,0]*cm[1,1]-cm[1,0]*cm[0,1])/numpy.sqrt(cm[0].sum()*cm[1].sum()*cm[:,0].sum()*cm[:,1].sum())



precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
pandas.Series(model.feature_importances_,

             index=cols).plot.bar()
stroke_patients = data.loc[data['stroke']==1]

stroke_patients.shape
continuous_for_stroke = stroke_patients.loc[:,continuous]

mean = continuous_for_stroke.mean(axis=0)

mean.plot.bar()
covar = continuous_for_stroke.cov()

seaborn.heatmap(covar)
invcov = numpy.linalg.inv(covar.values)

seaborn.heatmap(invcov)
non_stroke = data.loc[data['stroke']==0]

distances = pandas.Series(scipy.spatial.distance.cdist(non_stroke.loc[:,continuous].values,

                                                       mean.values.reshape((1,3)),

                                                       'mahalanobis',

                                                        invcov)[:,0],

                          index=non_stroke.index)

closest = distances.nsmallest(783)

closest
combined = stroke_patients.append(non_stroke.loc[closest.index])

MI=pandas.DataFrame(numpy.zeros((combined.shape[1],combined.shape[1])),

                    index = combined.columns,

                    columns = combined.columns)

continuous = ('age','avg_glucose_level','bmi')

for (i,column) in enumerate(combined.columns[:-1]):

    later = data.columns[i+1:]

    H = (sklearn.feature_selection.mutual_info_regression 

                                 if column in continuous 

                                 else sklearn.feature_selection.mutual_info_classif)(combined[later],combined[column])

    MI.loc[column,later] = H

    MI.loc[later,column] = H

seaborn.heatmap(MI)
MI['stroke'].plot.bar()