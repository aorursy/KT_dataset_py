import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

def load_titanic_data():
    csv_path = os.path.join("/kaggle/input/titanic", "train.csv")
    return pd.read_csv(csv_path)

def load_titanic_test_data():
    csv_path = os.path.join("/kaggle/input/titanic", "test.csv")
    return pd.read_csv(csv_path)

def clean_data(titanic):

    ordinal_encoder = OrdinalEncoder()
    # Change the sex from a string to a number
    sex_cat = titanic[["Sex"]]
    sex_enc = ordinal_encoder.fit_transform(sex_cat)
    titanic['Sexenc'] = sex_enc
    titanic.drop('Sex', axis='columns', inplace=True)
    
    # Change embarked to a number
    titanic["Embarked"] = titanic["Embarked"].fillna('S')
    embarked_cat = titanic[['Embarked']]
    embarked_enc = ordinal_encoder.fit_transform(embarked_cat)
    titanic['Embarkedenc'] = embarked_enc
    titanic.drop('Embarked', axis='columns', inplace=True)
    
    # Change the Fare, fill missing values with median
    titanic['Fare'].replace('', np.nan, inplace=True)
    mean = titanic['Fare'].mean() 
    titanic['Fare'].fillna(mean, inplace=True)
    titanic['Fare'] = pd.to_numeric(titanic['Fare'])
    
    titanic = titanic.fillna("")
    # The first letter of the Cabin indicates the deck, change to a number
    titanic['Deck'] = titanic['Cabin'].str[:1]
    deck_cat = titanic[['Deck']]
    deck_enc = ordinal_encoder.fit_transform(deck_cat)
    titanic['Deckenc'] = deck_enc
    titanic.drop('Deck', axis='columns', inplace=True)
    titanic.drop('Cabin', axis='columns', inplace=True)
    
    # Change the age, fill missing values with median
    titanic['Age'].replace('', np.nan, inplace=True)
    mean = titanic['Age'].mean() 
    titanic['Age'].fillna(mean, inplace=True)
    titanic['Age'] = pd.to_numeric(titanic['Age'])
    
    titanic.loc[titanic['PassengerId'] == 631, 'Age'] = 48
    
    # Drop columns that don't have strong correlation with survived
    titanic.drop('SibSp', axis='columns', inplace=True)
    titanic.drop('Parch', axis='columns', inplace=True)
    titanic.drop('Name', axis='columns', inplace=True)
    titanic.drop('Ticket', axis='columns', inplace=True)
    
    titanic.set_index("PassengerId", inplace = True)
    
    return titanic

def removeOutliers(titanic):
    print(titanic.count())
    removeOutliersFromColumn(titanic, "Fare")
    removeOutliersFromColumn(titanic, "Age")
    return titanic
    
def removeOutliersFromColumn(titanic, column):
    q_low = titanic[column].quantile(0.01)
    q_hi  = titanic[column].quantile(0.99)
    #print(f"{q_low} {q_hi}")
    df_filtered = titanic[(titanic[column] < q_hi) & (titanic[column] > q_low)]
    print(df_filtered.count())
    
def showCorrelation(titanic):
    corr_matrix = titanic.corr()
    print(corr_matrix['Survived'])
titanic=load_titanic_data()
titanic=clean_data(titanic)
titanic=removeOutliers(titanic)
showCorrelation(titanic)
print(titanic.columns.tolist())
y = titanic['Survived'].copy()
titanic.drop('Survived', axis='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(titanic, y, random_state=42)
X_train_scaled = scale(titanic)
X_test_scaled = scale(X_test)

clf_svm = SVC(kernel = 'rbf', random_state=42)
clf_svm.fit(X_train_scaled, y)
#plot_confusion_matrix(clf_svm, X_test_scaled, y_test, values_format='d', display_labels=['Survived', 'Died'])
titanic_test = load_titanic_test_data()
result_df = pd.DataFrame(titanic_test['PassengerId'].values)
titanic_test = clean_data(titanic_test)
X_test_scaled = scale(titanic_test)

prediction = pd.DataFrame(clf_svm.predict(X_test_scaled))

result_df['Survived']=prediction
result_df=result_df.rename(columns = {0:'PassengerId'})
result_df.set_index('PassengerId', inplace=True)
csv_path = os.path.join("/kaggle/working/", "titanic_prediction.csv")
result_df.to_csv(csv_path)
