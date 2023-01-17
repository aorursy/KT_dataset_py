import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns #data visualization
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from matplotlib.pyplot import imshow
import copy

from subprocess import check_output
#bring in the train dataset
traindf = pd.read_csv('../input/train.csv')

#bring in the test dataset
testdf = pd.read_csv('../input/test.csv')

print(traindf.head(5));
# replace_names takes in a training set, then rolls up the titles in the Name column
# to a group title defined in the function
def replace_names(x):
    title = x['Name']
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        return 'Sir'
    elif title in ['the Countess', 'Mme', 'Lady']:
        return 'Lady'
    elif title in ['Mr']:
        return 'Mr'
    elif title in ['Mlle', 'Ms']:
        return 'Mrs'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Sir'
        else:
            return 'Lady'
    else:
        return 'Other'

def wrangle_data(dataset):
    # Drop unneeded columns
    dataset.drop('Ticket', axis=1, inplace=True);
    #dataset.drop('PassengerId', axis=1, inplace=True);

    # Set any fare less than 5 to NaN;
    dataset.loc[dataset['Fare'] < 5, 'Fare'] = np.NaN;
    # Apply the average for that particular passenger class to each NaN
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("mean"), inplace = True);

    # Replace missing values of age with mean age
    dataset[dataset.Age.isnull()];
    dataset["Age"].fillna(dataset["Age"].mean(), inplace = True);

    # Get title from "Name" feature using RegEx, replace existing name with the title
    dataset["Name"] = dataset["Name"].str.extract('.*?, (.*?)\.', expand = True);

    # Replace Cabin with first letter, set it to capital
    dataset["Cabin"] = dataset["Cabin"].astype(str).str[0].astype(str).str.upper();


    # apply replace_names functions to the Name column
    dataset['Name'] = dataset.apply(replace_names, axis=1);
    #print(dataset.Name.value_counts());

    # "Explode out" nominal features
    dataset = pd.get_dummies(dataset, columns=['Name', 'Sex', 'Embarked', 'Cabin']);

    # categorize the ordinal feature Pclass
    dataset.Pclass.astype("category", ordered=True);

    # mean normalize and feature scaling on age, fare, sibsp, parch
    scaler = preprocessing.StandardScaler();
    cols = ["Age", "Fare", "SibSp", "Parch"];
    dataset[cols] = scaler.fit_transform(dataset[cols]);
    return dataset;

#combine the train and test datasets so dummy feature are identical
num_train_objs = len(traindf);    
combineddf = pd.concat(objs=[traindf, testdf], axis=0);
combineddf = wrangle_data(combineddf);

#split the datasets back out
traindf = copy.copy(combineddf[:num_train_objs])
testdf = copy.copy(combineddf[num_train_objs:].loc[:, traindf.columns != 'Survived'])
#Split into X (input) and Y (output -- in this case, survival)
X = traindf.drop(['PassengerId', 'Survived'], axis = 1);

y = traindf.loc[:, traindf.columns == 'Survived'];
#print(y);
#Run XGBoost on the dataset
#train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25);

#my_imputer = Imputer()
#train_X = my_imputer.fit_transform(train_X)
#test_X = my_imputer.transform(test_X)

from xgboost import XGBClassifier;

my_model = XGBClassifier(n_estimators=1000);
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X, y.values.ravel(), verbose=False);
# make predictions
#predictions = my_model.predict(test_X)

#from sklearn.metrics import mean_absolute_error
#print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
#print (testdf);
X = testdf.drop(['PassengerId'], axis=1);
prediction = my_model.predict(X);


to_submit = pd.DataFrame({
    'PassengerId':testdf['PassengerId'],
    'Survived':prediction
}, dtype=int);
print(to_submit);
to_submit.to_csv('csv_to_submit.csv', index = False)
