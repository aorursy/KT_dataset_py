

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



from plotly.offline import iplot, init_notebook_mode



import cufflinks as cf

import plotly.graph_objs as go



init_notebook_mode(connected=True)

cf.go_offline(connected=True)



# Set global theme

cf.set_config_file(world_readable=True, theme='ggplot')
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()
train_df.info()
train_df.describe()
preprocess_df = train_df.copy()



preprocess_df.Pclass =  preprocess_df.Pclass.astype(str)

test_df.Pclass =  test_df.Pclass.astype(str)
preprocess_df.drop(["Name","PassengerId"],inplace=True, axis=1)
passenger_id = test_df.PassengerId

test_df.drop(["Name","PassengerId"],inplace=True, axis=1)
preprocess_df.SibSp =  preprocess_df.SibSp.astype(str)

preprocess_df.Parch =  preprocess_df.Parch.astype(str)





test_df.SibSp =  test_df.SibSp.astype(str)

test_df.Parch =  test_df.Parch.astype(str)
preprocess_df.Survived = preprocess_df.Survived.astype(str).str.replace('0',"No")

preprocess_df.Survived = preprocess_df.Survived.astype(str).str.replace('1',"Yes")





def customized_heatmap(corr_df):

#     corr_mat = corr_df.iloc[1:,:-1].copy()

    corr_mat = corr_df.copy()

    

    #Create masks

    mask = np.triu(np.ones_like(corr_mat), k=1)

    

    # Plot

    plt.figure(figsize=(20,10))

    plt.title("Heatmap Corrleation")

    ax = sns.heatmap(corr_mat, vmin=-1, vmax=1, cbar=False,

                     cmap='coolwarm', mask=mask, annot=True)

    

    # format the text in the plot to make it easier to read

    for text in ax.texts:

        t = float(text.get_text())

        if -0.15 < t < 0.15:

            text.set_text('')        

        else:

            text.set_text(round(t, 2))

        text.set_fontsize('x-large')

    plt.xticks( size='x-large')

    plt.yticks(rotation=0, size='x-large')

    plt.show()

    
sns.heatmap(preprocess_df.corr(), annot=True)

preprocess_df.loc[:,preprocess_df.isnull().sum()>0].info()
preprocess_df.groupby('Embarked').mean()
sns.heatmap(preprocess_df[['Pclass','Age','Cabin']].isnull(), cmap="viridis")
preprocess_df.Age.describe()
preprocess_df.Age.min()
preprocess_df.Age.hist()
preprocess_df.groupby('Pclass').mean().round(0)
import math
def age_null_values(columns):

    age =columns[0]

    Pclass = columns[1]

    if math.isnan(age):

        if Pclass ==1:

            return 38

        elif Pclass == 2:

            return 30

        else:

            return 25

    else:

        return age

preprocess_df['Age']=preprocess_df[['Age',"Pclass"]].apply(age_null_values,axis=1)

test_df['Age']=test_df[['Age',"Pclass"]].apply(age_null_values,axis=1)


preprocess_df.drop("Cabin",axis=1, inplace=True)

preprocess_df.dropna(axis=0,inplace=True)



#Drop Cabin Column for test data

test_df.drop("Cabin",axis=1, inplace=True)



#TO create 418 rows replace na by either mode and mean in Embarked and Fare column respectively

test_df['Embarked'].fillna(test_df['Embarked'].mode())

fare_mean = test_df['Fare'].mean()

test_df['Fare'].fillna(fare_mean,inplace=True)

# lets drop ticket columns

preprocess_df.Ticket.describe()


preprocess_df.drop("Ticket",inplace=True,axis=1)

test_df.drop("Ticket",inplace=True,axis=1)
eda_df = preprocess_df.copy()
eda_df.Survived.value_counts()
fig= go.Figure()

fig.add_trace(go.Pie(labels=eda_df.Survived.value_counts().index, values=eda_df.Survived.value_counts().values))

fig.update_layout(title="Survived OR Not", legend_title="Survival Legend", template="plotly_dark")
sib_sp_to_str = {

    '0':'Zero',

    '1':"One",

    '2':"Two",

    '3':"Three",

    '4':"Four",

    '5':"Five",

    '6':"Six",

    '7':"Seven",

    "8":"Eight"

    }
eda_df.SibSp = eda_df.SibSp.replace(sib_sp_to_str)

test_df.SibSp = test_df.SibSp.replace(sib_sp_to_str)
grp_by_sibsp = eda_df.groupby('SibSp').mean().sort_values("Age", ascending=False)

fig= go.Figure()

fig.add_trace(go.Bar(x=grp_by_sibsp.index, y=grp_by_sibsp.Age, name='Age'))

fig.add_trace(go.Bar(x=grp_by_sibsp.index, y=grp_by_sibsp.Fare, name="Fare"))

# fig.update_layout(title="Survived OR Not", legend_title="Survival Legend", template="plotly_dark")
int_columns=  eda_df.loc[:,eda_df.dtypes!="object"].columns



fig,axes = plt.subplots(len(int_columns), figsize=(10,10))

for i,col in enumerate(int_columns):

    axes[i].hist(eda_df[eda_df.Survived == "No"][col].values, alpha=0.5, color="maroon", bins=15 )

    axes[i].hist(eda_df[eda_df.Survived == "Yes"][col].values, alpha=0.5, bins=15)

    axes[i].set_title(col)

    axes[i].set_yticks(())#cause we are not actually looking for numbers

axes[0].set_xlabel("Feature Columns")

axes[0].set_ylabel("Frequency")

axes[0].legend(["No", "Yes"], loc="best", title="Survived?")

fig.tight_layout()
pclass_to_str ={'1':'First','2':"Second",'3':"Third"}



eda_df.Pclass= eda_df.Pclass.replace(pclass_to_str)

test_df.Pclass= test_df.Pclass.replace(pclass_to_str)
class_1_survived_yes = eda_df[(eda_df.Pclass == "First") & (eda_df.Survived == "Yes")].shape[0]

class_1_survived_no = eda_df[(eda_df.Pclass == "First") & (eda_df.Survived == "No")].shape[0]





class_2_survived_yes = eda_df[(eda_df.Pclass == "Second") & (eda_df.Survived == "Yes")].shape[0]

class_2_survived_no = eda_df[(eda_df.Pclass == "Second") & (eda_df.Survived == "No")].shape[0]



class_3_survived_yes = eda_df[(eda_df.Pclass == "Third") & (eda_df.Survived == "Yes")].shape[0]

class_3_survived_no= eda_df[(eda_df.Pclass == "Third") & (eda_df.Survived == "No")].shape[0]
survived_no =pd.Series([class_1_survived_no,class_2_survived_no,class_2_survived_no])

survived_yes =pd.Series([class_1_survived_yes,class_2_survived_yes,class_2_survived_yes])
pclass_survived_avg = pd.pivot_table(eda_df, values=['Age'],index='Pclass',columns='Survived',)

pclass_survived_avg.columns =["Survived = No","Survived = Yes"]
pclass_survived_avg['Survived = No'] = pclass_survived_avg['Survived = No']/survived_no.values



pclass_survived_avg['Survived = Yes'] = pclass_survived_avg['Survived = Yes']/survived_yes.values

pclass_survived_avg_percentage =pclass_survived_avg*100
fig= go.Figure()

fig.add_trace(go.Bar(x=pclass_survived_avg_percentage.T.index, y=pclass_survived_avg_percentage.T['First'], name='First'))

fig.add_trace(go.Bar(x=pclass_survived_avg_percentage.T.index, y=pclass_survived_avg_percentage.T['Second'], name="Second"))



fig.add_trace(go.Bar(x=pclass_survived_avg_percentage.T.index, y=pclass_survived_avg_percentage.T['Third'], name="Third"))



fig.update_layout(title="Average Survival Percentage By Passenger Class", template="plotly_dark", xaxis_title="Survived ?", yaxis_title="Percentage", legend_title="Pclass")
fig= go.Figure()

fig.add_trace(go.Pie(labels=eda_df.Sex.value_counts().index, values=eda_df.Sex.value_counts().values))

fig.update_layout(title="Population", legend_title="Sex", template="plotly_dark")
plt.figure(figsize=(10,7))

sns.countplot(eda_df['Sex'], hue=eda_df['Survived'], palette="viridis")

plt.xlabel("Gender")

plt.ylabel("Frequency")

plt.title("Survival By Gender");
df = eda_df.copy()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()
scaler.fit(df[["Age","Fare"]])

df[["Age","Fare"]] = scaler.transform(df[["Age","Fare"]])



target = df['Survived']



df_dummy = df.loc[:,df.columns!="Survived"]



# For test 



test_df[["Age","Fare"]] = scaler.transform(test_df[["Age","Fare"]])
df_dummy = pd.get_dummies(df_dummy, drop_first=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( df_dummy.loc[:,df_dummy.columns!="Survived"], target, test_size=0.1, random_state=101)
#Import model and train model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier





from sklearn.model_selection import KFold, cross_val_score
num_folds = 10

scoring = "accuracy"

models=[]

rand_seed= 101



models.append(("Knn",KNeighborsClassifier(n_neighbors=5,p=2,leaf_size=10,) ))

models.append(("Log",LogisticRegression(random_state=rand_seed) ))

models.append(("Svm",SVC(random_state=rand_seed )))

models.append(("Rf", RandomForestClassifier(n_estimators=50,max_depth=5,random_state=rand_seed )))







results=[]

names=[]

metrics=[]
for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=rand_seed, shuffle=True)

    cv_score = cross_val_score(model,X_train,y_train, cv=kfold, scoring=scoring)

    

    names.append(name)

    results.append(cv_score)

    metrics.append(cv_score.mean())

    

    print("{name}: {score}".format(name=name,score= cv_score.mean()*100))
from sklearn.model_selection import RandomizedSearchCV



from sklearn.model_selection import GridSearchCV
# Number of trees in random forest

n_estimators = [i for i in range(200,2001, 200)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [i for i in range(10, 111,10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)

# rf = RandomForestClassifier()





# # Random search of parameters, using 3 fold cross validation, 

# # search across 80 different combinations, and use all available cores

# rf_random_cv = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 80, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# # Fit the random search model

# rf_random_cv.fit(X_train,y_train)
# print(rf_random_cv.best_params_)
param_grid_rf = {

    'bootstrap':[True],

    'max_depth': [5,10,15,20],

    'max_features': ['sqrt'],

    'min_samples_leaf': [ 1,3,5,7],

    'min_samples_split': [8,10,12,14],

    'n_estimators': [200,600,1200,1600,1800]

}

# # Instantiate the grid search model

# grid_search_cv_rf = GridSearchCV(RandomForestClassifier(), param_grid = param_grid_rf, 

#                           cv = 10, n_jobs = -1, verbose = 1)



# grid_search_cv_rf.fit(X_train, y_train)

# print(grid_search_cv_rf.best_params_)

# print(grid_search_cv_rf.best_score_)

final_rf= RandomForestClassifier(bootstrap=True,                                

                                   max_depth=10,

                                 max_features='sqrt',

                                   min_samples_leaf=1,

                                   min_samples_split=10,

                                   n_estimators=1600)
final_rf.fit(X_train,y_train)
y_pred = final_rf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred), "\n")



print(classification_report(y_test, y_pred), "\n")
test_df.Parch.unique()
eda_df.Parch.unique()
test_df_dummy = pd.get_dummies(test_df, drop_first=True)

test_df_dummy.drop('Parch_9', inplace=True, axis=1)




final_test_data_pred = final_rf.predict(test_df_dummy)
submission_csv = pd.DataFrame(data={'PassengerId':passenger_id,'Survived': final_test_data_pred})
submission_csv.to_csv("Titianic_Submission.csv", index=False)