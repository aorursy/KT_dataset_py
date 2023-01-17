import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier







def munge(data, train):

    data['HasName'] = data['Name'].fillna(0)

    data.loc[data['HasName'] != 0,"HasName"] = 1

    data['HasName'] = data['HasName'].astype(int)

    data['AnimalType'] = data['AnimalType'].map({'Cat':0,'Dog':1})



    if(train):

        data.drop(['AnimalID','OutcomeSubtype'],axis=1, inplace=True)

        data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2})

            

    gender = {'Neutered Male':1, 'Spayed Female':2, 'Intact Male':3, 'Intact Female':4, 'Unknown':5, np.nan:0}

    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)



    def agetodays(x):

        try:

            y = x.split()

        except:

            return None 

        if 'year' in y[1]:

            return float(y[0]) * 365

        elif 'month' in y[1]:

            return float(y[0]) * (365/12)

        elif 'week' in y[1]:

            return float(y[0]) * 7

        elif 'day' in y[1]:

            return float(y[0])

        

    data['AgeInDays'] = data['AgeuponOutcome'].map(agetodays)

    data.loc[(data['AgeInDays'].isnull()),'AgeInDays'] = data['AgeInDays'].median()



    data['Year'] = data['DateTime'].str[:4].astype(int)

    data['Month'] = data['DateTime'].str[5:7].astype(int)

    data['Day'] = data['DateTime'].str[8:10].astype(int)

    data['Hour'] = data['DateTime'].str[11:13].astype(int)

    data['Minute'] = data['DateTime'].str[14:16].astype(int)



    data['Name+Gender'] = data['HasName'] + data['SexuponOutcome']

    data['Type+Gender'] = data['AnimalType'] + data['SexuponOutcome']

    data['IsMix'] = data['Breed'].str.contains('mix',case=False).astype(int)

            

    return data.drop(['AgeuponOutcome','Name','Breed','Color','DateTime'],axis=1)









if __name__ == "__main__":

    in_file_train = '../input/train.csv'

    in_file_test = '../input/test.csv'



    print("Loading data...\n")

    pd_train = pd.read_csv(in_file_train)

    pd_test = pd.read_csv(in_file_test)





    pd_train = munge(pd_train,True)

    pd_test = munge(pd_test,False)



    pd_test.drop('ID',inplace=True,axis=1)



    train = pd_train.values

    test = pd_test.values







    print("Predicting... \n")

    forest = RandomForestClassifier(n_estimators = 400, max_features='auto')

    forest = forest.fit(train[0::,1::],train[0::,0])

    predictions = forest.predict_proba(test)



    output = pd.DataFrame(predictions,columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])

    output.columns.names = ['ID']

    output.index.names = ['ID']

    output.index += 1



    



    print(output)

 

    output.to_csv('predictions.csv')



    print("Done.\n")
import pandas as pd





print("Loading data...\n")

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")







def convert_AgeuponOutcome_to_weeks(df):

    result = {}

    for k in df['AgeuponOutcome'].unique():

        if type(k) != type(""):

            result[k] = -1

        else:

            v1, v2 = k.split()

            if v2 in ["year", "years"]:

                result[k] = int(v1) * 52

            elif v2 in ["month", "months"]:

                result[k] = int(v1) * 4.5

            elif v2 in ["week", "weeks"]:

                result[k] = int(v1)

            elif v2 in ["day", "days"]:

                result[k] = int(v1) / 7

                

    df['_AgeuponOutcome'] = df['AgeuponOutcome'].map(result).astype(float)

    df = df.drop('AgeuponOutcome', axis = 1)

                

    return df



train_df = convert_AgeuponOutcome_to_weeks(train_df)

test_df = convert_AgeuponOutcome_to_weeks(test_df)





names = pd.concat([test_df['Name'], train_df['Name']])

values = dict(names.value_counts())



train_df['_NameFreq'] = train_df['Name'].map(values)

test_df['_NameFreq'] = test_df['Name'].map(values)



train_df['_NameFreq'] = train_df['_NameFreq'].fillna(99999)

test_df['_NameFreq'] = test_df['_NameFreq'].fillna(99999)







def convert_to_numeric(df):

    for col in ['Name', 'AnimalType', 'SexuponOutcome',

                'Breed', 'Color', 'OutcomeType']:

        if col in df.columns:

            _col = "_%s" % (col)

            values = df[col].unique()

            _values = dict(zip(values, range(len(values))))

            df[_col] = df[col].map(_values).astype(int)

            df = df.drop(col, axis = 1)

    return df



train_df = convert_to_numeric(train_df)

test_df = convert_to_numeric(test_df)







def fix_date_time(df):

    def extract_field(_df, start, stop):

        return _df['DateTime'].map(lambda dt: int(dt[start:stop]))

    df['Year'] = extract_field(df,0,4)

    df['Month'] = extract_field(df,5,7)

    df['Day'] = extract_field(df,8,10)

    df['Hour'] = extract_field(df,11,13)

    df['Minute'] = extract_field(df,14,16)

    

    return df.drop(['DateTime'], axis = 1)



train_df = fix_date_time(train_df)

test_df = fix_date_time(test_df)







train_df = train_df.reindex(columns = ['AnimalID', '_Name', '_NameFreq','_AnimalType', '_SexuponOutcome','_AgeuponOutcome', '_Breed', '_Color','Year', 'Month', 'Day', 'Hour', 'Minute',

                                       '_OutcomeType'])





cut = int(len(train_df) * 0.8)

_validation_df = train_df[cut:]

_train_df = train_df[:cut]







import sklearn

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

print("Prediction...\n")

A1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2),

                        n_estimators = 100,

                        learning_rate = 0.1)



classifiers = [c.fit(_train_df.values[:,1:-1],

                     _train_df.values[:,-1].astype(int)) \

               for c in [A1]]

results = [c.predict_proba(_validation_df.values[:,1:-1]) \

           for c in classifiers]





from sklearn.metrics import log_loss







ab = classifiers[0].fit(train_df.values[:,1:-1],

                        train_df.values[:,-1].astype(int))







ab_result = ab.predict_proba(test_df.values[:,1:])

ab_sub_df = pd.DataFrame(ab_result, columns=['Adoption', 'Died', 'Euthanasia','Return_to_owner', 'Transfer'])

ab_sub_df.columns.names = ['ID']

ab_sub_df.index.names = ['ID']

ab_sub_df.index += 1







print(ab_sub_df)







ab_sub_df.to_csv("submission.csv")



print("Done.")
import numpy as np 

import pandas as pd 





print("Loading data...")

train_file = pd.read_csv("../input/train.csv")

test_file = pd.read_csv("../input/test.csv")





def fx(x):

    if type(x)=='str':

        if len(x) > 0:

            return 1

        else:

            return 0

        return np.nan

train_file.Name=train_file.Name.apply(lambda x: fx(x) )

test_file.Name=test_file.Name.apply(lambda x: fx(x) )







def gx(x):

    if x=='cat':

        return 1

    else:

        return 0

    return np.nan

train_file.AnimalType=train_file.AnimalType.apply(lambda x: gx(x) )

test_file.AnimalType=test_file.AnimalType.apply(lambda x: gx(x) )









#DateTime will be converted to categorical Year, Month and Day of the Week

from datetime import datetime

def convert_date(dt):

    d = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

    return d.year, d.month, d.isoweekday()



train_file["Year"], train_file["Month"], train_file["WeekDay"] = zip(*train_file["DateTime"].map(convert_date))

test_file["Year"], test_file["Month"], test_file["WeekDay"] = zip(*test_file["DateTime"].map(convert_date))

train_file.drop(["DateTime"], axis=1, inplace=True)

test_file.drop(["DateTime"], axis=1, inplace=True)



#Separating IDs

train_id = train_file[["AnimalID"]]

test_id = test_file[["ID"]]

train_file.drop(["AnimalID"], axis=1, inplace=True)

test_file.drop(["ID"], axis=1, inplace=True)



#Separating target variable

train_outcome = train_file["OutcomeType"]

train_file.drop(["OutcomeType"], axis=1, inplace=True)





#pd.options.mode.chained_assignment = None  # default='warn

#Encode the categorical data, with the complete set (train and test)

train_file["train"] = 1

test_file["train"] = 0

conjunto = pd.concat([train_file, test_file])

conjunto_encoded = pd.get_dummies(conjunto, columns=conjunto.columns)

train = conjunto_encoded[conjunto_encoded["train_1"] == 1]

test = conjunto_encoded[conjunto_encoded["train_0"] == 1]

train.drop(["train_0","train_1"], axis=1, inplace=True)

test.drop(["train_0","train_1"], axis=1, inplace=True)

# SPLIT TRAIN DATASET 



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, train_outcome, test_size=0.1)



#TRAIN A CLASSIFIER 

print("Prediction...")



from sklearn.linear_model import LogisticRegression

model = LogisticRegression()





N_ESTIMATORS=100





model.fit(X_train, y_train)

# GENERATE PREDICTIONS

y_pred_val = model.predict(X_val)

y_pred_val_prob = model.predict_proba(X_val)

mydataframe = pd.DataFrame(y_pred_val_prob)

mydataframe.columns =  ["Adoption","Died","Euthanasia","Return_to_owner","Transfer"]

mydataframe.head()

#EVALUATE THE MODEL/CLASSIFIER

from sklearn.metrics import classification_report, accuracy_score, log_loss





#GENERATE SUBMISSION FILE







model = LogisticRegression()



model.fit(train, train_outcome)

y_pred = model.predict_proba(test)





results = pd.read_csv("../input/sample_submission.csv")

results = pd.DataFrame(ab_result, columns=['Adoption', 'Died', 'Euthanasia','Return_to_owner', 'Transfer'])

results.columns.names = ['ID']

results.index.names = ['ID']

results.index += 1

results['Adoption'], results['Died'], results['Euthanasia'], results['Return_to_owner'], results['Transfer'] = y_pred[:,0], y_pred[:,1], y_pred[:,2], y_pred[:,3], y_pred[:,4]

print(results)



results.to_csv("outputs.csv")

print("Done.")