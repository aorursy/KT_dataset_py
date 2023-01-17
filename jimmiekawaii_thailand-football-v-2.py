import os

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

file_path = "../input/footballthailand/"

data = pd.read_csv(os.path.join(file_path,'Thailand_Football_100.csv'))

print(data)



data = data.apply(LabelEncoder().fit_transform)

print(data)



print(data.corr())



data.features = data[["competitor","stadium","weather"]]

data.targets = data.result 



feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.targets, test_size=.3)



model = DecisionTreeClassifier(criterion='entropy', presort=True)

model.fitted = model.fit(feature_train, target_train)

model.predictions = model.fitted.predict(feature_test)



print(confusion_matrix(target_test, model.predictions))

print(accuracy_score(target_test, model.predictions))



submission = pd.DataFrame({'competitor':feature_test['competitor'],'result':model.predictions})



submission["comp_enc"] = submission["competitor"].apply(lambda val: 

                                                       'Brunei' if val == 0 else (

                                                           'Combodia' if val == 1 else (

                                                           'Indonesia' if val == 2 else (

                                                           'Laos' if val == 3 else (

                                                           'Malaysia' if val == 4 else (

                                                           'Myanmar' if val == 5 else (

                                                           'Philippines' if val == 6 else (

                                                           'Singapore' if val == 7 else (

                                                           'Vietnam' if val == 8 else 0 )))))))))



submission["result_enc"] = submission["result"].apply(lambda val:

                                                     'draw' if val == 0 else (

                                                     'lose' if val == 1 else (

                                                     'win!' if val == 2 else 0 )))



predicted = submission[["competitor","comp_enc","result_enc"]]        

print(predicted)



filename = 'Thailand_Football_Predictions.csv'

predicted.to_csv(filename,index=False)

print('Saved file: ' + filename)