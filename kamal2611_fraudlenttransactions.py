# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import Imputer

from sklearn.metrics import accuracy_score
def prepareData( train_df, test_df):

    print(train_df.info())

    

    data = train_df.values

    test_data = test_df.values

    

    #if not train:

    #    data = np.concatenate( (data , np.zeros(data.shape[0])) , axis=1)

    X = data[: , 1:data.shape[1]-1]

    Y = data[: , data.shape[1]-1]

    print(X.shape, Y.shape)

    X_Test = test_data[: , 1:test_data.shape[1]-1]

    

    #imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    #imp.fit(X)

    #X = imp.transform(X)

    #X_Test = imp.transform(X_Test)

    

    labEncoder = LabelEncoder()

    labEncoder.fit(Y)

    label_y = labEncoder.transform(Y)

    #X[X == '?'] = 0

    final_X = None

    Test_X = None

    for cnt in range(0 , X.shape[1]):

        if train_df.columns[cnt+1].__contains__("cat"):

            labEncoder = LabelEncoder()

            labEncoder.fit(np.append(X[:,cnt] , X_Test[:,cnt]))

            curFeature = labEncoder.transform(X[:,cnt])

            curFeature = curFeature.reshape(X.shape[0], 1)

            

            curTestFeature = labEncoder.transform(X_Test[:,cnt])

            curTestFeature = curTestFeature.reshape(X_Test.shape[0], 1)

            

            #onehot_encoder = OneHotEncoder(sparse=False)

            #curFeature = onehot_encoder.fit_transform(curFeature)

        else:

            #imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

            #imp.fit_transform(X)

            curFeature = X[:,cnt]

            curFeature = curFeature.reshape(X.shape[0], 1)

            

            curTestFeature = X_Test[:,cnt]

            curTestFeature = curTestFeature.reshape(X_Test.shape[0], 1)

            



        if final_X is None:

            final_X = curFeature

        else:

            final_X = np.concatenate((final_X , curFeature) , axis=1)

        

        

        if Test_X is None:

            Test_X = curTestFeature

        else:

            Test_X = np.concatenate((Test_X , curTestFeature) , axis=1)

        

    #print(X=='tn')

    #final_X = final_X.astype('float32')

    # split data into train and test sets

    seed = 8

    test_size = .1

    X_train, X_test, Y_train, Y_test = train_test_split(final_X, label_y, test_size=test_size,

                                                        random_state=seed)



    



    return X_train, X_test, Y_train, Y_test , Test_X
train_dataset = pd.read_csv('../input/train.csv')

train_dataset.target.value_counts()

train_dataset = train_dataset.fillna('NaN')



test_dataset = pd.read_csv('../input/test.csv')

test_dataset['target'] = 0

test_dataset = test_dataset.fillna('NaN')



x_train, x_test, y_train, y_test , real_test = prepareData(train_dataset , test_dataset)



model = XGBClassifier(max_depth=10)

model.fit(x_train, y_train)

print(model)



# make predictions for test data

y_pred = model.predict(x_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
y_pred = model.predict(real_test)
pred_df = pd.DataFrame({'transaction_id':test_dataset['transaction_id'],'target':y_pred}, columns=["transaction_id" , "target"]).to_csv("yx9.csv" , index=False)