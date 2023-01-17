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
df = pd.read_csv('../input/train.csv')

#Dropping the Cabin column due to most of the elements are missing
df1 = df.drop('Cabin',axis=1)
print(df1.info())
print( df1['Age'].mean() , df1['Age'].median()) #Mean and median of age are almost the same

print(df[['PassengerId','Embarked']].groupby(by='Embarked').count()) # S=Southampton is the most frequent Port

#Filling the missing values in the Age column with the mean and in the Embarked column with S
df2 = df1.fillna(value={'Age':df1['Age'].mean(),'Embarked':'S'})

print(df2.head())

df3 = pd.get_dummies(df2, columns=['Sex','Embarked'],drop_first=True)



CATEGORICAL = ['Survived']
NUMERICAL = ['PassengerId','Age','Fare','Pclass','SibSp','Parch','Sex_male','Embarked_Q','Embarked_S']
TEXT = ['Name','Ticket']

categorize = lambda x: x.astype('category')

df3[CATEGORICAL] = df3[CATEGORICAL].apply(categorize, axis=0)
print(df3.info())
print(df3[CATEGORICAL].head())

#print(df3[NUMERICAL+CATEGORICAL].head())
X = df3.drop('Survived',axis=1)
y = df3[['Survived']]
def combine_text_columns(data_frame, to_drop=NUMERICAL + CATEGORICAL):
    """ Takes the dataset as read in, drops the non-feature, non-text columns and
        then combines all of the text columns into a single vector that has all of
        the text for a row.
        
        :param data_frame: The data as read in with read_csv (no preprocessing necessary)
        :param to_drop (optional): Removes the numeric and label columns by default.
    """
    # drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    
    # replace nans with blanks
    #text_data.fillna("", inplace=True)
    
    # joins all of the text items in a row (axis=1)
    # with a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)
print(combine_text_columns(X).head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

#TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

get_text_data = FunctionTransformer(combine_text_columns,validate=False)
get_numeric_data = FunctionTransformer( lambda x: x[NUMERICAL],validate=False)


print(get_text_data.fit_transform(df3.head()))
print(get_numeric_data.fit_transform(df3.head()))

text_pl = Pipeline([('selector', get_text_data),('vectorizer',CountVectorizer())])
numeric_pl = Pipeline([('selector',get_numeric_data)])


union = FeatureUnion(transformer_list=[('numeric', numeric_pl),('text',text_pl)])

model_pl = Pipeline([('union',union),('clf',DecisionTreeClassifier())])

model_pl.fit(X_train,y_train)

print(model_pl.score(X_test,y_test))


holdout = pd.read_csv('../input/test.csv')

holdout1 = holdout.drop('Cabin',axis=1)

holdout2 = holdout1.fillna(value={'Age':df1['Age'].mean(),'Fare': df1['Fare'].mean()})

holdout3 = pd.get_dummies(holdout2, columns=['Sex','Embarked'],drop_first=True)
print(holdout3.info())

predictions = model_pl.predict(holdout3)
print(predictions)

pred_df = pd.DataFrame({'PassengerId':holdout['PassengerId'].values,'Survived':predictions})
print(pred_df.head())

pred_df.to_csv("predictions.csv",index=False)