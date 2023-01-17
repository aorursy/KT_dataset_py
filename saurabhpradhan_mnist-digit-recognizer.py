import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble


dataFrame = pd.read_csv('../input/train.csv')
y = dataFrame['label'].values.tolist()
x = dataFrame.ix[:, dataFrame.columns != 'label'].values


classifier = ensemble.RandomForestClassifier(verbose=1)
parameters = {'n_estimators':[600,650,700,750]}
clf = GridSearchCV(classifier, parameters,n_jobs=7,scoring="f1_macro",cv=4,verbose=1)
clf.fit(x,y)


means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    print()
    
dataFrameTest = pd.read_csv('../input/test.csv')
yTest = dataFrameTest.values
yPrediction = clf.predict(yTest)
columns = ['ImageId','Label']

results = pd.DataFrame(columns=columns)
index = 0

for result in yPrediction:
    results.loc[index] = [index + 1,result]
    index = index + 1
    
    
results.head(6)