import pandas as pd
TRAINING_DATASET_PATH = "/kaggle/input/animal-shelter-fate/train.csv"
TEST_DATASET_PATH = "/kaggle/input/animal-shelter-fate/test.csv"
SAMPLE_DATASET_PATH = "/kaggle/input/animal-shelter-fate/sample.csv"
def load_dataset(path):
    return pd.read_csv(path)
df = load_dataset(TRAINING_DATASET_PATH)
df.head(5)
df["Breed"].unique()
df.dtypes
df['Date of Birth'].isnull().values.any()
df['Age upon Outcome'].isna().sum()
df.shape
df['Outcome Type'].unique()
import math
def convert_age_outcome_to_months(time):
    out_time = time
    
    if '-' in out_time:
        out_time = out_time.replace('-','')

    if 'days' in out_time or 'day' in out_time:
        out_time = out_time.replace('days','').replace('day','')
        
    if 'weeks' in out_time or 'week' in out_time:
        out_time = out_time.replace('weeks','').replace('week','')
        
    elif 'months' in out_time or 'month' in out_time:
        out_time = int(out_time.replace('months','').replace('month','')) * 30

    elif 'years' in out_time or 'year' in out_time:
         out_time = int(out_time.replace('years','').replace('year','')) * 365
       
    return round(int(out_time)/30,0)

def convert_breed(text):
    if 'Mix' in text or 'mix' in text:
        return "1!" + text.replace("Mix",'').replace('mix', '')
    else:
        return "0!" + text
def transform_data(df, is_training = True):
    ldf = df.copy(deep=True)
    ldf['Date of Birth'] = pd.to_datetime(ldf['Date of Birth'], format='%m/%d/%Y')
    
    ldf['date_time'] = pd.to_datetime(ldf['Date of Birth'], format='%m/%d/%Y').dt.year


    ldf['months_old'] = round((pd.to_datetime('2020-09-25') - ldf['Date of Birth']).dt.days / 30 , 0)
    ldf['months_old'] = ldf['months_old'].astype(int)

    if is_training:
        ldf = ldf[ldf['Age upon Outcome'].notna()]
    else:
        ldf = ldf.replace(np.nan, '25', regex=True)

    ldf['months_upon_outcome'] = ldf['Age upon Outcome'].apply(convert_age_outcome_to_months)
    ldf['months_upon_outcome'] = ldf['months_upon_outcome'].astype(int)

    ldf['operation'], ldf['sex'] = ldf['Sex upon Outcome'].str.split(' ', 1).str

    ldf['sex'] = ldf['sex'].fillna('Unknown')

    ldf['color1'], ldf['color2'] = ldf['Color'].str.split('/', 1).str
    ldf['color2'] = ldf['color2'].fillna('None')
    
    ldf["color1"] = ldf["color1"].str.strip()
    ldf["color2"] = ldf["color2"].str.strip()

    ldf['mixedbreeds'] = ldf.apply(lambda row: convert_breed(row['Breed']), axis=1)
    ldf['mixed'], ldf['breeds'] = ldf['mixedbreeds'].str.split('!').str
    
    return ldf
def drop_unnecessary_columns(df):
    ldf = df.copy(deep=True)
    ldf = ldf.drop(['Name','Animal ID', 'Sex upon Outcome', 'DateTime', 'Date of Birth', 'Age upon Outcome','Color','mixedbreeds','Breed', 'ID'], axis=1)    

    return ldf
def factorize_data(df, is_training = True):
    ldf = df.copy(deep=True)
    ldf['Animal Type'] = pd.factorize(ldf['Animal Type'])[0]
    ldf['color1'] = pd.factorize(ldf['color1'])[0]
    ldf['color2'] = pd.factorize(ldf['color2'])[0]
    ldf['operation'] = pd.factorize(ldf['operation'])[0]
    ldf['sex'] = pd.factorize(ldf['sex'])[0]
    ldf['breeds'] = pd.factorize(ldf['breeds'])[0]
    
    if is_training:
        ldf['Outcome Type'] = pd.factorize(ldf['Outcome Type'], sort=True)[0]
    
    return ldf
df = load_dataset(TRAINING_DATASET_PATH)
df = transform_data(df)
df = drop_unnecessary_columns(df)
df = factorize_data(df)
df.head()
from sklearn.model_selection import train_test_split
x_train = df.loc[:, df.columns != 'Outcome Type']
y_train = df['Outcome Type']
train, test = train_test_split(df, test_size=0.2)
x_test = test.loc[:, df.columns != 'Outcome Type']
y_test = test['Outcome Type']
x_train.shape
y_train.shape
x_train.head()
y_train.head()
type(y_train)
y_array = y_train.values
y_array[0:10]
y_classes = []

for i in y_array:
    single = [0] * 9
    single[i] = 1
    y_classes.append(single)
y_classes[0:10]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
from sklearn import linear_model
# random forest
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
classifier = RandomForestClassifier(n_estimators = 1000, max_depth=50, random_state = 234)
classifier.fit(x_train, y_train)
# importance of features
from matplotlib import pyplot

importance = classifier.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
# predicting test set results
value = 0
y_pred = classifier.predict_proba(x_test)
print(y_pred[0:5])
print(y_test.head())
y_pred[0:10]
y_test[0:10]
len(x_test)
test = load_dataset(TEST_DATASET_PATH)
test_ids = test["ID"]
test = transform_data(test, False)
test = drop_unnecessary_columns(test)
test = factorize_data(test, False)
test.head()
y_test_predicted_2 = classifier.predict_proba(test)
y_test_predicted_2[0]
len(y_test_predicted_2)
len(test.index)
#save to file:
f = open("result6_proba.csv", "w")
f.write("ID,Adoption,Died,Disposal,Euthanasia,Missing,Relocate,Return to Owner,Rto-Adopt,Transfer\n")
for i in range(len(y_test_predicted_2)):
    f.write(str(test_ids[i]) + "," +  ",".join(str(v) for v in y_test_predicted_2[i]))
    if i + 1 < len(y_test_predicted_2):
        f.write("\n")

f.close()