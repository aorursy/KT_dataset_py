import pandas as pd
input_io_dir="../input/"
original_train_data=pd.read_csv(input_io_dir+"train.csv")
original_test_data=pd.read_csv(input_io_dir+"test.csv")
print('PrepareDataSets:original_train_data',original_train_data.shape)
print('PrepareDataSets:original_test_data',original_test_data.shape)
passengerId = original_test_data['PassengerId']
print('PrepareDataSets:passengerId (%d)'%len(passengerId))
survived=original_train_data['Survived']
print('PrepareDataSets:Survived (%d)'%len(survived))
# Once stored, let's drop the column
original_train_data=original_train_data.drop('Survived',axis=1)
# This will let us build a combined dataset
original_all_data=original_train_data.append(original_test_data)
print('PrepareDataSets:original_alldata',original_all_data.shape)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Support functions
################################################################################################
# Replace texts based on a dictionary
def multipleReplace(text, wordDic):
    for key in wordDic:
        if text.lower()==key.lower():
            text=wordDic[key]
            break
    return text

# Normalise title names by grouping them
def normaliseTitle(title):
    wordDic = {
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mrs':'Mrs',
    'Master':'Master',
    'Mme': 'Mrs',
    'Lady': 'Nobility',
    'Countess': 'Nobility',
    'Capt': 'Army',
    'Col': 'Army',
    'Dona': 'Other',
    'Don': 'Other',
    'Dr': 'Other',
    'Major': 'Army',
    'Rev': 'Other',
    'Sir': 'Other',
    'Jonkheer': 'Other',
    }    
    title=multipleReplace(title,wordDic)
    return title

# Extract Title feature from name
def extractTitleFromName(name):
    pos_point=name.find('.')
    if pos_point == -1: return ""
    wordList=name[0:pos_point].split(" ")
    if len(wordList)<=0: return ""
    title=wordList[len(wordList)-1]
    normalisedTitle=normaliseTitle(title)
    return normalisedTitle

# Extract TicketType feature from name
def getTicketType(name, normalise=True):
    item=name.split(' ')
    itemLength=len(item)
    if itemLength>1:
        ticketType=""
        for i in range(0,itemLength-1):
            ticketType+=item[i].upper()
    else:
        ticketType="NORMAL"
    if normalise==True:
        ticketType= ticketType.translate(str.maketrans('','','./'))
    return ticketType

# Custom pipeline filter to add new features
################################################################################################
class CustomFeatureExtender(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['Title']=X['Name'].apply(lambda x: extractTitleFromName(x)).astype('category')
        X['NoCabin']=X['Cabin'].isnull().apply(lambda x: 1 if x is True else 0).astype('category')
        X['TicketType']=X['Ticket'].apply(lambda x: getTicketType(x)).astype('category')
        X['IsAlone']=(X["SibSp"]+X["Parch"]).apply(lambda x: 0 if x>0 else 1).astype('category')
        X['FamilySize']=X["SibSp"]+X["Parch"]+1
        return X
################################################################################################
preprocessor=Pipeline(steps=[
        ('extender', CustomFeatureExtender()),
    ])
print("PrepareDataSets:Features extended")
enriched_train_data=preprocessor.fit_transform(original_train_data)
enriched_test_data=preprocessor.fit_transform(original_test_data)
print('PrepareDataSets:enriched_train_data',enriched_train_data.shape)
print('PrepareDataSets:enriched_test_data',enriched_test_data.shape)
enriched_train_data.head()
exclude_features=['Name','SibSp','Parch','Ticket','Cabin']
filtered_train_data=enriched_train_data.drop(exclude_features,axis=1)
filtered_test_data=enriched_test_data.drop(exclude_features[1:],axis=1)
print('PrepareDataSets:filtered_train_data',filtered_train_data.shape)
print('PrepareDataSets:filtered_test_data',filtered_test_data.shape)
filtered_train_data.head()
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Helper functions
#############################################################################

# Column selection
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

# Assign names to columns
class ColumnLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X=pd.DataFrame(X,columns=self.column_names)
        return X

# Transform features - continuous to discrete
class CustomRangeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        Xdf=pd.DataFrame(X)
        Xdf.loc[Xdf['Age'] <= 16, 'Age'] = 0
        Xdf.loc[(Xdf['Age'] > 16) & (Xdf['Age'] <= 32), 'Age'] = 1
        Xdf.loc[(Xdf['Age'] > 32) & (Xdf['Age'] <= 48), 'Age'] = 2
        Xdf.loc[(Xdf['Age'] > 48) & (Xdf['Age'] <= 64), 'Age'] = 3
        Xdf.loc[ Xdf['Age'] > 64, 'Age'] = 4
        Xdf.loc[Xdf['Fare'] <= 7.91, 'Fare'] = 0
        Xdf.loc[(Xdf['Fare'] > 7.91) & (Xdf['Fare'] <= 14.454), 'Fare'] = 1
        Xdf.loc[(Xdf['Fare'] > 14.454) & (Xdf['Fare'] <= 31), 'Fare']   = 2
        Xdf.loc[ Xdf['Fare'] > 31, 'Fare'] = 3
        return Xdf

# Numeric pipeline into action!
#############################################################################
numeric_features=['Age','Fare','FamilySize']
# Let's first ensure there are no missing values
numeric_pipeline_step1 = Pipeline(steps=[
    ('selector', DataFrameSelector(numeric_features)),
    ('imputer', SimpleImputer(strategy='median')),
    ('labeler',ColumnLabeler(numeric_features))
    ])
num_encoded_train_data=pd.DataFrame(numeric_pipeline_step1.fit_transform(filtered_train_data))
print(num_encoded_train_data.head())
print('-----------------------------------------')
print('Notice below there are no missing values ')
print('-----------------------------------------')
num_encoded_train_data.info()
# Now, let's transform continuous into discrete
numeric_pipeline_step2 = Pipeline(steps=[
    ('selector', DataFrameSelector(numeric_features)),
    ('imputer', SimpleImputer(strategy='median')),
    ('labeler',ColumnLabeler(numeric_features)),
    ('range_transformer',CustomRangeTransformer()),
    ])
num_encoded_train_data=pd.DataFrame(numeric_pipeline_step2.fit_transform(filtered_train_data))
num_encoded_train_data.head()
# Let's apply the complete pipeline, including scaling
numeric_pipeline = Pipeline(steps=[
    ('selector', DataFrameSelector(numeric_features)),
    ('imputer', SimpleImputer(strategy='median')),
    ('labeler',ColumnLabeler(numeric_features)),
    ('range_transformer',CustomRangeTransformer()),
    ('scaler', StandardScaler()),
    ('labeler2',ColumnLabeler(numeric_features)),
    ])
num_encoded_train_data=pd.DataFrame(numeric_pipeline.fit_transform(filtered_train_data))
print('PrepareDataSets:num_encoded_train_data',num_encoded_train_data.shape)
num_encoded_test_data=pd.DataFrame(numeric_pipeline.fit_transform(filtered_test_data))
print('PrepareDataSets:num_encoded_test_data',num_encoded_test_data.shape)
num_encoded_train_data.head()
# Helper functions
#############################################################################

# Fill missing data - just set to most frequent value as there are only 2 missing values
class CustomFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['Embarked'].fillna('S')
        return X

# One hot encoding using pandas's getdummies
class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dummy_na):
        self.dummy_na=dummy_na
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame()
        # We iterate so that we set column name as prefix for newly created features
        for col in sorted(X.columns):
            dummies=pd.get_dummies(X[col],prefix=col, drop_first=True,dummy_na=self.dummy_na)
            df[dummies.columns]=dummies
        X=df
        X=X.astype('category')
        return X

# Categorical pipeline in action!
#############################################################################
categorical_features = ['Embarked', 'Sex','Pclass','Title','NoCabin','IsAlone']
categorical_pipeline = Pipeline(steps=[
        ('selector',DataFrameSelector(categorical_features)),
        ('filler', CustomFiller()),
        ('dummy', DummyTransformer(dummy_na=False)),
    ])
cat_encoded_train_data=pd.DataFrame(categorical_pipeline.fit_transform(filtered_train_data))
print('PrepareDataSets:cat_encoded_train_data',cat_encoded_train_data.shape)
cat_encoded_test_data=pd.DataFrame(categorical_pipeline.fit_transform(filtered_test_data))
print('PrepareDataSets:cat_encoded_test_data',cat_encoded_test_data.shape)
cat_encoded_train_data.head()
encoded_train_data=pd.concat([num_encoded_train_data,cat_encoded_train_data],axis=1)
print('PrepareDataSets:encoded_train_data',encoded_train_data.shape)
encoded_test_data=pd.concat([num_encoded_test_data,cat_encoded_test_data],axis=1)
print('PrepareDataSets:encoded_test_data',encoded_test_data.shape)
encoded_train_data.head()
encoded_test_data.head()
# Helper to normalise columns between training and test set
def NormaliseColumns(dataframeA,dataframeB):
    for testCol in dataframeB.columns:
        if testCol not in dataframeA.columns:
            dataframeA[testCol]=0
    for trainCol in dataframeA.columns:
        if trainCol not in dataframeB.columns:
            dataframeB[trainCol]=0
    return dataframeA,dataframeB

encoded_train_data,encoded_test_data=NormaliseColumns(encoded_train_data,encoded_test_data)
print('PrepareDataSets:Adjusted encoded_train_data',encoded_train_data.shape)
print('PrepareDataSets:Adjusted encoded_test_data',encoded_test_data.shape)
encoded_train_data.head()
encoded_test_data.head()
encoded_train_data=encoded_train_data.reindex(sorted(encoded_train_data.columns), axis=1)
encoded_test_data=encoded_test_data.reindex(sorted(encoded_test_data.columns), axis=1)
encoded_train_data.head()
encoded_test_data.head()
train_features=encoded_train_data
test_features=encoded_test_data
train_labels=survived
passengerId.to_csv("passengerId.csv",index=False,header=False)
train_features.to_csv("train_features.csv",index=False,header=True)
test_features.to_csv("test_features.csv",index=False,header=True)
train_labels.to_csv("train_labels.csv",index=False,header=False)