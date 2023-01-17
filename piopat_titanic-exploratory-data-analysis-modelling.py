import pandas as pd
import numpy as np
from numpy import char
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
def pyplot_options(width, length):
    plt.figure(figsize = (width, length))
sns.set(style="ticks", color_codes=True)
warnings.filterwarnings('ignore')
df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
df_sub = pd.read_csv("../data/gender_submission.csv")
df_train.head(1)
df_test.head(1)
df_sub.head(1)
df_train.dtypes
print("Shape of train dataset: ", df_train.shape)
print("Shape of test dataset: ", df_test.shape)
print("Shape of sub dataset: ", df_sub.shape)
pair_plot_features = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Age', 'Fare']
g = sns.pairplot(data=df_train.loc[:, pair_plot_features], hue='Survived', palette="Set1", diag_kind = 'hist')
pyplot_options(8, 5)
sns.countplot(x="Survived", data = df_train, palette="Set1")
plt.title("Number of survivors of the Titanic disaster")
df_train_new = df_train.copy()
#df_train_new['Sex'] = df_train_new.Sex.map({'male':1, 'female':0})
df_train_new['Sex'] = df_train_new['Sex'].astype('category')
pyplot_options(12, 8)
sns.countplot(data=df_train_new, x='Sex', palette="Set1")
plt.title("Number of men and women on board the Titanic", fontsize=15)
pyplot_options(12, 8)
sns.countplot(data=df_train_new, x='Sex', hue = "Survived", palette="Set1")
plt.title("Number of Male and Female people survived from disaster", fontsize=15)
df_train.Sex.isnull().value_counts()
age_cut=pd.cut(df_train_new.Age, bins = [0, 6, 14, 20, 30, 45, 60, 100])
d = {"CutAge":age_cut, "Sex": df_train_new.Sex, "Survived": df_train_new.Survived}
age_df = pd.DataFrame(data=d)
print("Age dataset shape:", age_df.shape)
pyplot_options(12, 8)
sns.countplot(data=age_df, x = "CutAge", hue = "Survived", palette="Set1")
plt.title("Number of survivors in contrast to Age variable", fontsize = 15)
def countplot(x, hue, **kwargs):
    sns.countplot(x=x, hue=hue, **kwargs, palette="Set1")
    
grid = sns.FacetGrid(age_df, col="Sex", size=4, aspect=1.2)
fig = grid.map(countplot, "CutAge", "Survived")
fig.add_legend()
age_null_values = 100*(df_train_new[df_train_new.Age.isnull() == True].shape[0]/df_train_new.shape[0])
print("Almost {}% of people have Age NaN values".format(np.round(age_null_values, 2)))
pyplot_options(10, 6)
sns.countplot(x = df_train.loc[:, 'Pclass'], hue = df_train.loc[:, 'Survived'], palette="Set1")
plt.title("Countplot of Passanger Class in contrast to Survived")
pclass_null_values = 100*(df_train_new[df_train_new.Pclass.isnull() == True].shape[0]/df_train_new.shape[0])
print("Almost {}% of people have Pclass NaN values".format(np.round(pclass_null_values, 2)))
pyplot_options(12, 8)
sns.countplot(data = df_train_new, x='SibSp', hue ='Survived', palette="Set1")
plt.title("Number of sibilings aboard to Titanic in contrast to Survived", fontsize=15)
def sibsp_function(feature):
    if feature == 0:
        return '0'
    elif feature == 1 or feature == 2: 
        return '1-2' 
    else: 
        return '>2'

df_train_new['SibSp_0'] = df_train_new.apply(lambda df: sibsp_function(df['SibSp']), axis=1)
pyplot_options(12, 8)
sns.countplot(data = df_train_new, x='SibSp_0', hue ='Survived', palette="Set1", order=['0', '1-2', '>2'])
plt.title("Number of sibilings aboard to Titanic in contrast to Survived", fontsize=15)
sib_null_values = 100*(df_train_new[df_train_new.SibSp.isnull() == True].shape[0]/df_train_new.shape[0])
print("Almost {}% of people have SibSp NaN values".format(np.round(sib_null_values, 2)))
pyplot_options(12, 8)
sns.countplot(data=df_train_new, x = 'Parch', hue = 'Survived', palette="Set1")
plt.title("Number of Parents or Children aboard to Titanic in contrast to Survived", fontsize=15)
def parch_function(feature):
    if feature == 0:
        return '0'
    else:
        return '>= 1'

df_train_new['Parch_0'] = df_train_new.apply(lambda df: parch_function(df['Parch']), axis=1)
pyplot_options(12, 8)
sns.countplot(data=df_train_new, x = 'Parch_0', hue = 'Survived', palette="Set1")
plt.title("Number of Parents or Children aboard to Titanic in contrast to Survived", fontsize=15)
ticket_counts = df_train_new.Ticket.value_counts()
ticket_counts = pd.DataFrame({'Ticket':ticket_counts.index, 'PassengerTicket':ticket_counts}).reset_index().drop(['index'],axis=1)
df_train_new = pd.merge(df_train_new, ticket_counts, left_on='Ticket', right_on = 'Ticket', how='left')
df_train_new.head()
pyplot_options(12, 6)
sns.countplot(data=df_train_new, x="PassengerTicket", hue="Survived", palette="Set1")
plt.title("Number of people on one ticket in contrast to survivors", fontsize=15)
pyplot_options(12, 8)
sns.countplot(data=df_train_new[df_train_new.PassengerTicket == 1], x="Survived", palette="Set1")
plt.title("Number of people on one ticket (only one person) in contrast to survivors", fontsize=15)
pyplot_options(12, 8)
sns.countplot(data=df_train_new[df_train_new.PassengerTicket >= 2 ], x="Survived", palette="Set1")
plt.title("Number of people on one ticket (greater than 2) in contrast to survivors", fontsize=15)
pyplot_options(12, 8)
sns.boxplot(data=df_train_new, x = "Survived", y="Fare", orient="v", palette="Set1")
plt.title("Boxplot of Fare in contrast to Survived", fontsize=15)
df_train_new['FareBins'] = pd.cut(df_train_new.Fare, bins=[0, 10, 25, 50, 75, 1000])
pyplot_options(12, 8)
sns.countplot(data=df_train_new, x = "FareBins", hue="Survived", palette="Set1")
plt.title("Countplot interval of Fare in contrast to Survived", fontsize=15)
cabin_counts = df_train_new.Cabin.value_counts()
cabin_counts = pd.DataFrame({'Cabin':cabin_counts.index, 'PassengerCabin':cabin_counts}).reset_index().drop(['index'],axis=1)
df_train_new2 = pd.merge(df_train_new.copy(), cabin_counts, left_on='Cabin', right_on = 'Cabin', how='left')
df_train_new2.PassengerCabin[df_train_new2.PassengerCabin.isna() == True] = 0
df_train_new2.head()
pyplot_options(12, 8)
sns.countplot(data=df_train_new2, x="PassengerCabin", hue="Survived", palette="Set1")
plt.title("Number of Passengers who have cabin in contrast to Survived", fontsize=15)
pyplot_options(12, 8)
sns.countplot(data=df_train_new, x = "Embarked", hue="Survived", palette="Set1")
plt.title("Place of Embarked in contrast to Survived", fontsize=15)
df_train_new.Name.value_counts()
df_train_new.columns
df_train_new.set_index('PassengerId', inplace=True)
df_name = df_train_new.Name.str.split(",", expand=True).rename(columns = lambda x: f"Part_Name_{x+1}")
df_name_2 = df_name.Part_Name_2.str.split(".", expand=True).rename(columns = lambda x: f"Part_Name_{x+1}")
df_name.rename(columns={"Part_Name_1":"Surname", "Part_Name_2":"Title_Name"}, inplace=True)
df_name_2.rename(columns = {"Part_Name_1":"Title", "Part_Name_2":"Surname", "Part_Name_3":"Rest"}, inplace=True)
df_name.head(1)
df_name_2.head(1)
df_name_2.Title.unique()
df_name_2['Title'] = df_name_2.Title.str.strip()
df_name_2['Surname'] = df_name_2.Surname.str.strip()
df_name_2['Rest'] = df_name_2.Rest.str.strip()
pyplot_options(18, 8)
sns.countplot(data=df_name_2, x = "Title", palette="Set1")
normal_titles = ('Mr', 'Miss', 'Mrs', 'Master', 'Ms')
unusual_titles = ('Dr', 'Rev', 'Mlle', 'Col', 'Capt', 'the Countess', 'Mme', 'Lady')
df_plot = pd.merge(df_name_2, pd.DataFrame(df_train_new), how='inner', left_on='PassengerId', right_on='PassengerId') 
df_plot['NormalTitle'] = df_plot.Title.isin(normal_titles).astype(int)
df_plot.head(1)
pyplot_options(16, 6)
sns.countplot(x='Title', hue = 'Survived', palette="Set1", data = df_plot)
df_plot['Youth'] = df_plot.Title.isin(['Miss', 'Master']).astype(int)
pyplot_options(12, 8)
sns.countplot(x='Youth', hue ='Survived', palette="Set1", data=df_plot)
pyplot_options(12, 8)
sns.countplot(x='NormalTitle', hue ='Survived', palette='Set1', data=df_plot)
df_plot.Age.isnull().value_counts()
age_youth = np.round(df_plot[df_plot.Youth == 1]['Age'].mean())
age_adult = np.round(df_plot[df_plot.Youth == 0]['Age'].mean())
age_mr = np.round(df_plot[df_plot.Title == 'Mr']['Age'].mean())
age_mrs = np.round(df_plot[df_plot.Title == 'Mrs']['Age'].mean())

print("Youth Age:", age_youth)
print("Adult Age:", age_adult)
print("Mr Age:", age_mr)
print("Mrs Age:", age_mrs)
def set_index_data(data):
    data.set_index('PassengerId', inplace=True)
    return print("Complete")
def convert_features(data):
    df = data.copy()
    df['Pclass'] = df['Pclass'].astype('category') 
    df['SibSp'] = df['SibSp'].astype('category')
    df['Parch'] = df['Parch'].astype('category')
    df['Sex'] = df['Sex'].map({'male':1, 'female':0}).astype('category')
    df['Embarked'] = df['Embarked'].astype('category')
    df['Fare'] = df['Fare'].round(2)
    return df
def feature_engineering_agebins(data, bins=[0, 6, 20, 30, 45, 60, 100], inplace=False):
    data['AgeBins']=pd.cut(data.Age, bins=bins, right=False, labels=bins[:-1])
    print("Complete")
    return data
def imputation_age(df):
    age_youth = np.round(df[df.Youth == 1]['Age'].mean())
    age_mr = np.round(df[df.Title == 'Mr']['Age'].mean())
    age_mrs = np.round(df[df.Title == 'Mrs']['Age'].mean())
    age_adult = np.round(df[df.Youth == 0]['Age'].mean())
    
    df.loc[(df.Youth == 1) & (df.Age.isnull()), 'Age']  = age_youth
    df.loc[(df.Title == 'Mr') & (df.Age.isnull()), 'Age']  = age_mr
    df.loc[(df.Title == 'Mrs') & (df.Age.isnull()), 'Age']  = age_mrs
    df.loc[(df.Youth == 0) & (df.Age.isnull()), 'Age']  = age_adult
    return df
def feature_passenger_ticket(data):
    ticket_counts = data.Ticket.value_counts()
    ticket_counts = pd.DataFrame({'Ticket':ticket_counts.index, 
                                  'PassengerTicket':ticket_counts}).reset_index().drop(['index'],axis=1) 
    data = pd.merge(data, ticket_counts, left_on='Ticket', right_on = 'Ticket', how='left')
    data['PassengerTicketGT2'] = 0
    data['PassengerTicketGT2'][data.PassengerTicket >= 2] = 1
    print("Complete")
    return data
def feature_passenger_cabin(data):
    cabin_counts = data.Cabin.value_counts()
    cabin_counts = pd.DataFrame({'Cabin':cabin_counts.index, 
                                 'PassengerCabin':cabin_counts})\
    .reset_index()\
    .drop(['index'],axis=1)
    data = pd.merge(data, cabin_counts, left_on='Cabin', right_on = 'Cabin', how='left')
    data.PassengerCabin[data.PassengerCabin.isna() == True] = 0
    data['PassengerCabinGT2'] = 0
    data['PassengerCabinGT2'][data.PassengerCabin >= 2] = 1
    print("Complete")
    return data
def feature_engineering_newcabin(data):
    data['NewCabin']=char.ljust(np.array(data.Cabin.replace(np.nan, 'X')).astype(str), width=0)
    print("Complete")
    return data
def feature_engineering_farebins(data, bins=[0, 10, 25, 50, 75, 1000]):
    data['FareBins']=pd.cut(data.Fare, bins=bins, right=False, labels=bins[:-1])
    print('Complete')
    return data
def feature_engineering_sibsp(data):
    def sibsp_function(feature):
        if feature == 0:
            return '0'
        elif feature == 1 or feature == 2: 
            return '1-2' 
        else: 
            return '>2'
        
    data['SibSp_0'] = data.apply(lambda df: sibsp_function(df['SibSp']), axis=1)
    return data
def feature_engineering_parch(data):
    def parch_function(feature):
        if feature == 0:
            return 0
        else:
            return 1

    data['NewParch'] = data.apply(lambda df: parch_function(df['Parch']), axis=1)
    return data
def feature_engineering_name(data):
    
    # Copy input dataset
    df = data.copy()
    
    # Split Name feature into several columns
    df_name = df.Name.str.split(",", expand=True).rename(columns = lambda x: f"Part_Name_{x+1}")
    df_name_2 = df_name.Part_Name_2.str.split(".", expand=True).rename(columns = lambda x: f"Part_Name_{x+1}")
    
    # Rename new features and remove whitespaces
    df_name_2.rename(columns = {"Part_Name_1":"Title"}, inplace=True)
    df_name_2['Title'] = df_name_2.Title.str.strip()
    
    # Create vector with normal titles
    normal_titles = ('Mr', 'Miss', 'Mrs', 'Master', 'Ms')
    
    # Create new variables 
    df_name_2['NormalTitle'] = df_name_2.Title.isin(normal_titles).astype(int)
    df_name_2['Youth'] = df_name_2['Title'].isin(['Miss', 'Master']).astype(int)
    df_name_2 = df_name_2.loc[:, ('Title', 'NormalTitle', 'Youth')]
    
    df = pd.merge(df, df_name_2, how="inner", on="PassengerId")
    return df
df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
df_sub = pd.read_csv("../data/gender_submission.csv")
from sklearn.model_selection import train_test_split
X = df_train.copy().drop(['Survived'], axis=1)
y = df_train['Survived']
np.random.seed(23)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('Shape of training features: ', X_train.shape)
print('Shape of testing features: ', X_test.shape)
print('Shape of training target: ', y_train.shape)
print('Shape of testing target: ', y_test.shape)
set_index_data(X_train)
set_index_data(X_test)
set_index_data(df_test)
set_index_data(df_train)
X_train=convert_features(X_train)
X_test=convert_features(X_test)
df_test = convert_features(df_test)
df_train = convert_features(df_train)
X_train=feature_engineering_name(X_train)
X_test=feature_engineering_name(X_test)
df_test=feature_engineering_name(df_test)
df_train=feature_engineering_name(df_train)
X_train = imputation_age(X_train)
X_test = imputation_age(X_test)
df_train = imputation_age(df_train)
df_test = imputation_age(df_test)
X_train=feature_engineering_agebins(X_train, inplace=False)
X_test=feature_engineering_agebins(X_test, inplace=False)
df_test=feature_engineering_agebins(df_test, inplace=False)
df_train=feature_engineering_agebins(df_train, inplace=False)
X_train=feature_passenger_ticket(X_train)
X_test=feature_passenger_ticket(X_test)
df_test=feature_passenger_ticket(df_test)
df_train=feature_passenger_ticket(df_train)
X_train=feature_passenger_cabin(X_train)
X_test=feature_passenger_cabin(X_test)
df_test=feature_passenger_cabin(df_test)
df_train=feature_passenger_cabin(df_train)
X_train=feature_engineering_newcabin(X_train)
X_test=feature_engineering_newcabin(X_test)
df_test=feature_engineering_newcabin(df_test)
df_train=feature_engineering_newcabin(df_train)
X_train=feature_engineering_farebins(X_train)
X_test=feature_engineering_farebins(X_test)
df_test=feature_engineering_farebins(df_test)
df_train=feature_engineering_farebins(df_train)
X_train=feature_engineering_sibsp(X_train)
X_test=feature_engineering_sibsp(X_test)
df_test=feature_engineering_sibsp(df_test)
df_train=feature_engineering_sibsp(df_train)
X_train=feature_engineering_parch(X_train)
X_test=feature_engineering_parch(X_test)
df_test=feature_engineering_parch(df_test)
df_train=feature_engineering_parch(df_train)
X_train.head()
X_train.columns
def convert_features_all(data):
    columns_category = ('Pclass', 'SibSp', 'Parch', 'Embarked', 'Title', 'AgeBins', 'PassengerTicket', 'PassengerCabin', 'NewCabin', 'FareBins', 'SibSp_0')
    columns_float = ('Sex', 'Age', 'Fare', 'NormalTitle', 'Youth', 'PassengerTicketGT2', 'PassengerCabinGT2', 'NewParch')
    for i in columns_category:
        data[i] = data[i].astype('category')
        
    for i in columns_float:
        data[i] = data[i].astype('float64')
        
    return data
X_train = convert_features_all(X_train)
X_test = convert_features_all(X_test)
df_train = convert_features_all(df_train)
df_test = convert_features_all(df_test)
X_train = X_train.drop(['Name', 'Ticket',  'Cabin'], axis=1)
X_test = X_test.drop(['Name', 'Ticket',  'Cabin'], axis=1)
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_train = df_train.drop(['Name', 'Ticket',  'Cabin'], axis=1)
df_test.dtypes
from sklearn.impute import SimpleImputer 
def na_values(data, feature):
    return data[data[feature].isna() == True].shape[0]

def number_of_class(data, feature):
    return len(data[feature].unique())
def isin_feature(df_train, df_test, feature):
    unique_values = df_train[feature].unique()
    data_test = df_test.copy()
    data_test.loc[data_test[feature].isin(unique_values), feature] = data_test[feature]
    data_test.loc[~data_test[feature].isin(unique_values), feature] = np.nan
    return data_test
features = ['FareBins', 'AgeBins', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'NewCabin', 'PassengerCabin', 'PassengerCabinGT2', 'PassengerTicket', 'PassengerTicketGT2', 'SibSp_0']
for i in features:
    print("NA values in {} feature: {}".format(i, na_values(df_test, i)))
    print("Number of class in {} feature: {}".format(i, number_of_class(df_test, i)))
    print()
X_test = isin_feature(X_train, X_test, 'Parch')
df_test = isin_feature(df_train, df_test, 'Parch')
categorical_variables = ('Pclass', 'SibSp', 'Parch', 'Embarked', 'AgeBins', 'PassengerTicket', 'PassengerCabin', 'NewCabin', 'FareBins', 'SibSp_0')
categorical_imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
categorical_imputer.fit(X_train.loc[:, categorical_variables])
X_train.loc[:, categorical_variables] = categorical_imputer.transform(X_train.loc[:, categorical_variables])
X_test.loc[:, categorical_variables] = categorical_imputer.transform(X_test.loc[:, categorical_variables])
df_test.loc[:, categorical_variables] = categorical_imputer.transform(df_test.loc[:, categorical_variables])
df_train.loc[:, categorical_variables] = categorical_imputer.transform(df_train.loc[:, categorical_variables])
X_train.dtypes
X_train = convert_features_all(X_train)
X_test = convert_features_all(X_test)
df_train = convert_features_all(df_train)
df_test = convert_features_all(df_test)
from sklearn.preprocessing import OneHotEncoder

encoder_features = ['FareBins', 'AgeBins', 'Pclass', 'SibSp', 'Parch', 'Embarked', 'Title', 'NewCabin', 'PassengerCabin', 'PassengerTicket', 'SibSp_0']
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(df_train.loc[:, encoder_features])
encoding_features_train=encoder.transform(X_train.loc[:, encoder_features]).toarray()
encoding_features_test=encoder.transform(X_test.loc[:, encoder_features]).toarray()
encoding_features_df_test=encoder.transform(df_test.loc[:, encoder_features]).toarray()
encoding_features_df_train=encoder.transform(df_train.loc[:, encoder_features]).toarray()
X_train.drop(columns=encoder_features, inplace=True)
X_test.drop(columns=encoder_features, inplace=True)
df_test.drop(columns=encoder_features, inplace=True)
df_train.drop(columns=encoder_features, inplace=True)
df_encoding_train = pd.DataFrame(encoding_features_train, index=X_train.index)
df_encoding_train.columns = encoder.get_feature_names(encoder_features)

df_encoding_test = pd.DataFrame(encoding_features_test, index=X_test.index)
df_encoding_test.columns = encoder.get_feature_names(encoder_features)

df_encoding_df_test = pd.DataFrame(encoding_features_df_test, index=df_test.index)
df_encoding_df_test.columns = encoder.get_feature_names(encoder_features)

df_encoding_df_train = pd.DataFrame(encoding_features_df_train, index=df_train.index)
df_encoding_df_train.columns = encoder.get_feature_names(encoder_features)
X_train=pd.concat([X_train, df_encoding_train], join='outer', axis=1, ignore_index=False)
X_test=pd.concat([X_test, df_encoding_test], join='outer', axis=1, ignore_index=False)
df_test=pd.concat([df_test, df_encoding_df_test], join='outer', axis=1, ignore_index=False)
df_train=pd.concat([df_train, df_encoding_df_train], join='outer', axis=1, ignore_index=False)
print("Shape of train data ", X_train.shape)
print("Shape of test data ", X_test.shape)
print("Shape of test data ", df_test.shape)
print("Shape of training data ", df_train.shape)
df_train.columns
df_train.shape
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
model = SVC(kernel="linear")
rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(5), scoring='accuracy')

rfecv.fit(X_train, y_train)
print("Optimal number of features: %d" % rfecv.n_features_)
rfecv_results = pd.DataFrame({'Columns':X_train.columns, 'Ranking':rfecv.ranking_})
rfecv_results[rfecv_results.Ranking == 1]
feature_selection_array = np.array(rfecv_results[rfecv_results.Ranking == 1].Columns)
pyplot_options(12, 8)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X_train, y_train)
importances = pd.DataFrame({'Features':X_train.columns, 'Importance':clf.feature_importances_})
importances.sort_values(by='Importance', ascending=False).head(10)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
fit = pca.fit(X_train)
print("Explained Variance: %s" % fit.explained_variance_ratio_)
#print(fit.components_)
pd.DataFrame({'Components': np.arange(1, 3), 'Explanined Varianve': fit.explained_variance_ratio_})
X_pca_train = pca.transform(X_train)
X_pca_test = pca.transform(X_test)
X_train_new = X_train[feature_selection_array]
X_test_new = X_test[feature_selection_array]
X_train.columns
my_features = np.array(['Sex', "AgeBins_0", "AgeBins_6", "AgeBins_20", "AgeBins_30", "AgeBins_45", "AgeBins_60", "Pclass_3", "Parch_0", "PassengerCabinGT2", 
                        "PassengerTicketGT2", "SibSp_0", 'SibSp_1'])
X_train_my = X_train[my_features]
X_test_my = X_test[my_features]
my_features2 = np.array(['Sex', 'Parch_4', 'Parch_5', 'SibSp_0_>2', 'Title_Master', 'NewCabin_E', 'Fare', 'Youth', "AgeBins_0", "AgeBins_6", "AgeBins_20", "AgeBins_30", "AgeBins_45", "AgeBins_60"])
X_train_my2 = X_train[my_features]
X_test_my2 = X_test[my_features]
X_train_new.columns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def logistic_regression_grid_search(X_train, X_test, y_train, y_test):

    grid_param = [
        {'C' : range(0, 11, 1),
        'fit_intercept' : [True, False],
        'penalty' : ['l2'],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear'],
        'class_weight' : ['balanced']},
        {'C' : range(0, 11, 1),
        'fit_intercept' : [True, False],
        'penalty' : ['l1'],
        'solver' : ['liblinear'],
        'class_weight' : ['balanced']},
        {'C' : range(0, 11, 1),
        'fit_intercept' : [True, False],
        'penalty' : ['elasticnet'],
        'solver' : ['saga'],
        'class_weight' : ['balanced'],
        'l1_ratio' : [0.1, 0.3, 0.5, 0.7]}]

    scores = ['accuracy', 'precision', 'recall']
    best_params = []

    for score in scores:
        model_logistic = LogisticRegression()
        model_logistic_gs = GridSearchCV(estimator=model_logistic, param_grid=grid_param, cv=5, scoring=score)
        model_logistic_gs.fit(X_train, y_train)
        print("Score: %s" % score)
        print("Best parameters:", model_logistic_gs.best_params_)
        print()
        print("Training score: %s" % model_logistic_gs.score(X_train, y_train))
        print("Testing score: %s" % model_logistic_gs.score(X_test, y_test))
        best_params = np.append(best_params, model_logistic_gs.best_params_)
    return best_params
    
best_params_lr_new = logistic_regression_grid_search(X_train_new, X_test_new, y_train, y_test)
best_params_lr_my = logistic_regression_grid_search(X_train_my, X_test_my, y_train, y_test)
best_params_lr_my2 = logistic_regression_grid_search(X_train_my2, X_test_my2, y_train, y_test)
best_params_lr_pca = logistic_regression_grid_search(X_pca_train, X_pca_test, y_train, y_test)
# Save Results
import pickle

filename1 = "logistic_reg_new_20200612.sav"
filename2 = "logistic_reg_my_20200612.sav"

pickle.dump(best_params_lr_new, open(filename1, 'wb'))
pickle.dump(best_params_lr_my, open(filename2, 'wb'))
from sklearn.ensemble import RandomForestClassifier

def random_forest_grid_search(X_train, X_test, y_train, y_test):
    grid_param = [
        {'n_estimators' : range(10, 110, 10),
        'criterion' : ['gini', 'entropy'],
        'min_samples_split' : [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'min_samples_split' : [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'bootstrap' : [True, False],
        'random_state' : [123],
        'class_weight' : ['balanced', 'balanced_subsample']}]

    scores = ['accuracy']#, 'precision', 'recall']
    best_params = []

    for score in scores:
        random_forest_model = RandomForestClassifier()
        random_forest_model_gs = GridSearchCV(estimator=random_forest_model, param_grid=grid_param, cv=5, scoring=score)
        random_forest_model_gs.fit(X_train, y_train)
        print("Score: %s" % score)
        print("Best parameters:", random_forest_model_gs.best_params_)
        print()
        print("Training score: %s" % random_forest_model_gs.score(X_train, y_train))
        print("Testing score: %s" % random_forest_model_gs.score(X_test, y_test))
        best_params = np.append(best_params, random_forest_model_gs.best_params_)
    return best_params


best_params_rf_new = random_forest_grid_search(X_train_new, X_test_new, y_train, y_test)
best_params_rf_my = random_forest_grid_search(X_train_my, X_test_my, y_train, y_test)
filename1 = "random_forest_new_20200612.sav"
filename2 = "random_forest_my_20200612.sav"

pickle.dump(best_params_rf_new, open(filename1, 'wb'))
pickle.dump(best_params_rf_my, open(filename2, 'wb'))
from sklearn.ensemble import AdaBoostClassifier
import time as t

def adaboost_grid_search(X_train, X_test, y_train, y_test):
    grid_param = [
        {'n_estimators' : range(10, 110, 10),
        'learning_rate' : [ 0.01, 0.05, 0.1],
        'algorithm' : ['SAMME', 'SAMME.R']}]

    scores = ['accuracy', 'precision', 'recall']
    best_params = []
    start = t.time()
    
    for score in scores:
        adaboost_model = AdaBoostClassifier()
        adaboost_model_gs = GridSearchCV(estimator=adaboost_model, param_grid=grid_param, cv=5, scoring=score)
        adaboost_model_gs.fit(X_train, y_train)
        print("Score: %s" % score)
        print("Best parameters:", adaboost_model_gs.best_params_)
        print()
        print("Training score: %s" % adaboost_model_gs.score(X_train, y_train))
        print("Testing score: %s" % adaboost_model_gs.score(X_test, y_test))
        print()
        best_params = np.append(best_params, adaboost_model_gs.best_params_)
    end = t.time() - start
    print("Evaluation time: %s" % end)
    return best_params

best_params_ab_new = adaboost_grid_search(X_train_new, X_test_new, y_train, y_test)
best_params_ab_my = adaboost_grid_search(X_train_my, X_test_my, y_train, y_test)
filename1 = "adaboost_new_20200612.sav"
filename2 = "adaboost_my_20200612.sav"

pickle.dump(best_params_ab_new, open(filename1, 'wb'))
pickle.dump(best_params_ab_my, open(filename2, 'wb'))
from sklearn.ensemble import GradientBoostingClassifier
import time as t

def gradient_boosting_grid_search(X_train, X_test, y_train, y_test):
    grid_param = [
        {'n_estimators' : range(10, 110, 10),
        'learning_rate' : [ 0.01, 0.05, 0.1]}]

    scores = ['accuracy', 'precision', 'recall']
    best_params = []
    start = t.time()
    
    for score in scores:
        gradient_boosting_model = GradientBoostingClassifier()
        gradient_boosting_model_gs = GridSearchCV(estimator=gradient_boosting_model, param_grid=grid_param, cv=5, scoring=score)
        gradient_boosting_model_gs.fit(X_train, y_train)
        print("Score: %s" % score)
        print("Best parameters:", gradient_boosting_model_gs.best_params_)
        print()
        print("Training score: %s" % gradient_boosting_model_gs.score(X_train, y_train))
        print("Testing score: %s" % gradient_boosting_model_gs.score(X_test, y_test))
        print()
        best_params = np.append(best_params, gradient_boosting_model_gs.best_params_)
    end = t.time() - start
    print("Evaluation time: %s" % end)
    return best_params
best_params_gb_new = gradient_boosting_grid_search(X_train_new, X_test_new, y_train, y_test)
best_params_gb_my = gradient_boosting_grid_search(X_train_my, X_test_my, y_train, y_test)
filename1 = "gradboost_new_20200612.sav"
filename2 = "gradboost_my_20200612.sav"

pickle.dump(best_params_gb_new, open(filename1, 'wb'))
pickle.dump(best_params_gb_my, open(filename2, 'wb'))
from xgboost import XGBClassifier

def xgboosting_grid_search(X_train, X_test, y_train, y_test):
    grid_param = [
        {'n_estimators' : range(10, 110, 10),
        'learning_rate' : [ 0.01, 0.05, 0.1],
        'booster': ['gbtree', 'gblinear', 'dart'],
        'max_depth':[4,6,8,10,12],
        'min_child_weight':[4,5,6],
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5]}]

    scores = ['accuracy']#, 'precision', 'recall']
    best_params = []
    start = t.time()
    
    for score in scores:
        xgb_model = XGBClassifier()
        xgb_model_gs = GridSearchCV(estimator=xgb_model, param_grid=grid_param, cv=5, scoring=score)
        xgb_model_gs.fit(X_train, y_train)
        print("Score: %s" % score)
        print("Best parameters:", xgb_model_gs.best_params_)
        print()
        print("Training score: %s" % xgb_model_gs.score(X_train, y_train))
        print("Testing score: %s" % xgb_model_gs.score(X_test, y_test))
        print()
        best_params = np.append(best_params, xgb_model_gs.best_params_)
    end = t.time() - start
    print("Evaluation time: %s" % end)
    return best_params
best_params_xgb_new = xgboosting_grid_search(X_train_new, X_test_new, y_train, y_test)
best_params_xgb_my = xgboosting_grid_search(X_train_my, X_test_my, y_train, y_test)
filename1 = "xgb_new_20200612.sav"
filename2 = "xgb_my_20200612.sav"

pickle.dump(best_params_xgb_new, open(filename1, 'wb'))
pickle.dump(best_params_xgb_my, open(filename2, 'wb'))
X_new = X[feature_selection_array]
df_test_new = df_test[feature_selection_array]
X_my2 = X[my_features2]
lr_params = {'C': 5, 'class_weight': 'balanced', 'fit_intercept': True, 'penalty': 'l2', 'solver': 'newton-cg'}
lr_params2 = {'C': 1, 'class_weight': 'balanced', 'fit_intercept': False, 'l1_ratio': 0.1, 'penalty': 'elasticnet', 'solver': 'saga'}
rf_params = {'bootstrap': False, 'class_weight': 'balanced', 'criterion': 'gini', 'min_samples_split': 0.05, 'n_estimators': 20, 'random_state': 123}
ab_params = {'algorithm': 'SAMME.R', 'learning_rate': 0.1, 'n_estimators': 100}
gb_params = {'learning_rate': 0.01, 'n_estimators': 50}
xgb_params = {'booster': 'gbtree', 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 50}
model_lr = LogisticRegression(**lr_params)
model_lr2 = LogisticRegression(**lr_params2)
model_rf = RandomForestClassifier(**rf_params)
model_ab = AdaBoostClassifier(**ab_params)
model_gb = GradientBoostingClassifier(**gb_params)
model_xgb = XGBClassifier(**xgb_params)
model_lr.fit(X_new, y)
model_lr.score(X_new, y)
model_lr.fit(X_my2, y)
model_lr.score(X_my2, y)
model_rf.fit(X_new, y)
model_rf.score(X_new, y)
model_ab.fit(X_new, y)
model_ab.score(X_new, y)
model_gb.fit(X_new, y)
model_gb.score(X_new, y)
model_xgb.fit(X_new, y)
model_xgb.score(X_new, y)
df_sub_lr = model_lr.predict(df_test_new)
df_sub_rf = model_rf.predict(df_test_new)
df_sub_lr_model=pd.DataFrame({'PassengerId':df_test.PassengerId, 'Survived':df_sub_lr})
df_sub_rf_model=pd.DataFrame({'PassengerId':df_test.PassengerId, 'Survived':df_sub_rf})
df_sub_lr_model.to_csv("../data/df_sub_lr_20200612.csv", columns = ("PassengerId", "Survived"), index=False)
df_sub_rf_model.to_csv("../data/df_sub_rf_20200612.csv", columns = ("PassengerId", "Survived"), index=False)