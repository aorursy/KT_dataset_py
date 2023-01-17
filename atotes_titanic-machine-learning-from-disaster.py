 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns





from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold





from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier





import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Combine test and train dataset

def CombineTestTrainData():

    join_df = train_data

    join_df.drop(['Survived'], 1, inplace=True)

    join_df = join_df.append(test_data)

    join_df.reset_index(inplace=True)    

    return join_df



#Convert CategoricalFeature to numeric 

def MapCategoricalFeature(data):

    data['Sex'] = data['Sex'].map({'male': 1, 'female': 2})

    data['Embarked'] = data['Embarked'].map({'C': 1, 'Q': 2, 'S':3})

    return data



#Remove alphabet from column

def ConverToDigit(ticket):

    return ''.join(filter(lambda i: i.isdigit(), ticket))



#Plot Chart

def plot_univariant_analysis_for_categorical_col(df,col): 

    sns.set(style="darkgrid")

    total = float(len(df))

    plt.figure(figsize=(15,3))

    ax = sns.countplot(x=col,data=df,palette=sns.color_palette("Set2"))

    plt.xlabel(col)

    plt.ylabel('Frequency of occurence')        

    plt.xticks(rotation=45)

    for p in ax.patches:

        height = np.nan_to_num(p.get_height(), 0)

        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(100 * height/total)+' %',ha="center")
#Read train data

train_data = pd.read_csv(os.path.join(dirname, 'train.csv'), sep=",")

test_data = pd.read_csv(os.path.join(dirname, 'test.csv'), sep=",")
#Preview train data

train_data.head()
#Preview Test data

test_data.head()
#Check null value in train data

round(100*(train_data.isnull().sum()/len(train_data.index)),2)
#Remove rows having Embarked value null

train_data= train_data[pd.notnull(train_data['Embarked'])]
train_data.reset_index(inplace=True)

dependent = pd.DataFrame(index=train_data.index)

dependent['Survived'] = train_data['Survived']
#Combine train and test data for treating missing values

join_df = CombineTestTrainData()
#Preview combine dataframe

join_df.head()
round(100*(join_df.isnull().sum()/len(join_df.index)),2)
#Delete index column

join_df.drop(['index'],inplace=True, axis=1)
#Cabin has almost more than 50% of data missing so better delete cabin column

join_df.drop(['Cabin'],inplace=True, axis=1)
#Fill missing value in fare column with median values.

join_df['Fare'].fillna(join_df['Fare'].median(), inplace = True)
#use below column as independent feature to predict age value

df = join_df[['Pclass','Sex','Age','Fare','Embarked']]
#Convert categorical value to numeric

df = MapCategoricalFeature(df)
#Preview dataframe 

df.head()
test_df = df[pd.isnull(df['Age'])]

train_df = df[pd.notnull(df['Age'])]
y_train = train_df.pop('Age')

X_train = train_df
lm = LinearRegression()

lm.fit(X_train, y_train)
y_test = test_df.pop('Age')

X_test = test_df
y_pred = lm.predict(X_test)

y_pred
X_test['Age'] = y_pred
#merge predicted age value in join dataframe

join_df.fillna(X_test, inplace=True)
#Check null vales in dataframe

round(100*(join_df.isnull().sum()/len(join_df.index)),2)
#Remove text from ticket column and keep digits

join_df['Ticket'] = join_df['Ticket'].apply(lambda x: ConverToDigit(x))
#Split data back to train and test

train_data = join_df.iloc[:889]

test_data = join_df.iloc[889:]
train_data.info()
test_data.info()
100*(dependent.isnull().sum()/len(dependent.index))
train_data['Survived'] = dependent['Survived']
#Check the prcentage of survived people

plot_univariant_analysis_for_categorical_col(train_data,'Survived')
plot_univariant_analysis_for_categorical_col(train_data,'Sex')
plot_univariant_analysis_for_categorical_col(train_data,'Pclass')
plot_univariant_analysis_for_categorical_col(train_data,'Embarked')
plt.figure(figsize=(10,5))

sns.distplot(train_data['Age'])

plt.title("Distribution Plot")

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(train_data['Fare'])

plt.title("Distribution Plot")

plt.show()
sns.countplot(hue='Survived', x='Pclass', orient='h', data=train_data)
sns.countplot(hue='Survived', x='Sex', orient='h', data=train_data)
sns.countplot(hue='Survived', x='Embarked', orient='h', data=train_data)
join_df = CombineTestTrainData()

join_df.head()
join_df = MapCategoricalFeature(join_df)
join_df.head()
#Derive new feature as Family size

join_df["FamilySize"] = join_df["SibSp"] + join_df["Parch"] + 1

join_df.drop(['index','SibSp','Parch'], inplace = True, axis=1)
#Derive new feature Title

Title = [i.split(",")[1].split(".")[0].strip() for i in join_df["Name"]]

join_df["Title"] = pd.Series(Title)

join_df["Title"] = join_df["Title"].replace(['Lady', 'the Countess',

                                             'Capt', 'Col','Don', 'Dr', 

                                             'Major', 'Rev', 'Sir', 'Jonkheer',

                                             'Dona'], 'Rare')



join_df["Title"] = join_df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 ,

                                         "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, 

                                         "Rare":3})



join_df["Title"] = join_df["Title"].astype(int)
#Derive new feature Name

join_df.drop(['Name'], inplace = True, axis=1)
bins = [0, 10, 20, 30, 40,50,60,70,80,90]

labels = [1, 2, 3, 4,5,6,7,8,9]

join_df['age_bin'] = pd.cut(join_df.Age, bins=bins, labels=labels)
join_df.drop(['Age','level_0'],inplace = True,axis = 1)
train_data = join_df.iloc[:889]

test_data = join_df.iloc[889:]

train_data['Survived'] = dependent['Survived']
train_data.info()
test_data.head()
train_data.drop("Ticket",axis=1, inplace=True)

test_data.drop("Ticket",axis=1, inplace=True)
x_train = train_data.drop(columns=['Survived'])

y_train = train_data.Survived
#Build LogisticRegression 

log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)

y_pred = log_reg.predict(test_data)

log_reg.score(x_train, y_train)
#Build SVC 

# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of hyperparameters

# Set the parameters by cross-validation

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]}]





# specify model

model = SVC(kernel="rbf")



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = model, 

                        param_grid = hyper_params, 

                        scoring= 'accuracy', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv.fit(x_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



# # plotting

plt.figure(figsize=(20,8))



# subplot 1/3

plt.subplot(131)

gamma_01 = cv_results[cv_results['param_gamma']==0.01]



plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])

plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.01")

#plt.ylim([0.80, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')



# subplot 2/3

plt.subplot(132)

gamma_001 = cv_results[cv_results['param_gamma']==0.001]



plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])

plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.001")

#plt.ylim([0.80, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')





# subplot 3/3

plt.subplot(133)

gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]



plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])

plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title("Gamma=0.0001")

#plt.ylim([0.80, 1])

plt.legend(['test accuracy', 'train accuracy'], loc='upper left')

plt.xscale('log')
# printing the optimal accuracy score and hyperparameters

best_score = model_cv.best_score_

best_hyperparams = model_cv.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
# random forest - the class weight is used to handle class imbalance - it adjusts the cost function

forest = RandomForestClassifier(class_weight={0:0.1, 1: 0.9}, n_jobs = -1)



# hyperparameter space

params = {"criterion": ['gini', 'entropy'], "max_features": ['auto', 0.4]}



# create 5 folds

folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)



# create gridsearch object

model = GridSearchCV(estimator=forest, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)



# fit model

model.fit(x_train, y_train)



# printing the optimal accuracy score and hyperparameters

best_score = model.best_score_

best_hyperparams = model.best_params_



print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))
predictions = model.predict(test_data)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")