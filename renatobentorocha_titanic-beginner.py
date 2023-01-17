import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
train_data_path = '../input/train.csv'
test_data_path = '../input/test.csv'

train = pd.read_csv(train_data_path)
test = pd.read_csv(test_data_path)
train.isnull().any()
def fill_age(data_frame, class_group):
    '''
        Update the age column with then mean age by column group.
    '''
    df_grouped_pclass = data_frame.groupby(by=class_group);
    mean_age_by_pclass = df_grouped_pclass.Age.mean()
    
    df_out_1 = data_frame.copy();

    for index in mean_age_by_pclass.index:    
        df_out_1 = data_frame[data_frame.Age.isnull() & (data_frame.Pclass==index)];
        df_out_1 = df_out_1.Age.fillna(math.floor(mean_age_by_pclass.iloc[index-1]));
        data_frame.update(df_out_1);   

fill_age(train, 'Pclass');
print('Row number of null: {}'.format(train.Age.isnull().sum()));
train[train['Fare']==0].count().any()
def update_fare_column(data_frame):
    df_grouped_pclass = data_frame.groupby(by='Pclass');
    mean_fare_by_pclass = df_grouped_pclass.Fare.mean();
    df_out_1 = data_frame.copy();

    for index in mean_fare_by_pclass.index:    
        df_out_1 = data_frame[((data_frame.Fare==0) | data_frame.Fare.isnull()) & (data_frame.Pclass==index)];
        df_out_1.Fare.replace(to_replace=[0, np.nan], value=math.floor(mean_fare_by_pclass.iloc[int(index)-1]), inplace=True);    
        data_frame.update(df_out_1);
update_fare_column(train)
train[train['Fare']==0].count().any()
import operator
import random


def GetIntervals(data_frame):
    """
     - Return a dictionary of dictionary (intervals).
     
     - The key to the most external dictionary is the passenger's class of boarding;
       its value is a dictionary where the key is the place of embarkation and the 
       value is the percentage of shipments made in this place.
     
     Ex: {1: {'Q': 0.0022497187851518562, 'C': 0.09561304836895389, 'S': 0.14285714285714285} }
    """
    group_class_embark = data_frame.groupby(by=['Pclass', 'Embarked']);
    keys_class_embark = group_class_embark.groups.keys();
    total_embarked = data_frame.Embarked.count();
    
    intervals = {}
    clean_sub_intervals = {}    

    for k in keys_class_embark:
        if np.nan not in k:        
            if k[0] not in intervals:        
                intervals[k[0]] = {}

            intervals[k[0]][k[1]] = \
                group_class_embark.get_group(k).Embarked.count() / total_embarked;

    for k, v in intervals.items():
        intervals[k] = dict(sorted(v.items(), key=operator.itemgetter(1)));
    
    return intervals


def GetEmbarked(intervals, Pclass):
    """
     - Intervals is a dictionary made by: GetIntervals()
     - Pclass is a shipping class
     
     - Given the dictionary of shipment probabilities per class and class, 
       a value between 0 and 1 is randomly generated to select the place of shipment of the class.
     
     - Return: shipping place
    """    
    n = random.random();
    sub_intervals = intervals[Pclass];
    Embarked = list(sub_intervals.keys())[-1];
        
    for k, v in sub_intervals.items():        
        if n <= v:            
            Embarked = k
            break
        
    return Embarked

def UpdateNullEmbarked(data_frame):
    """
    - Returns dataframe where the records that have the null shipment column are 
      updated according to probability of shipment per class;
    """
    null_embarked = data_frame[data_frame.Embarked.isnull()];

    for index, row in null_embarked.iterrows():        
        null_embarked.at[index, 'Embarked'] = GetEmbarked(GetIntervals(data_frame), row.Pclass);

    return null_embarked
train.update(UpdateNullEmbarked(train))
print('Null embarked: ', train.Embarked.isnull().sum())
train.info()
def update_column_sex(data_frame):
    data_frame.update(train.Sex.apply(lambda x: 1 if x == 'male' else 2));    
def update_survived_column_type(data_frame):
    data_frame.Survived = data_frame.Survived.astype(int);
def update_age_column_type(data_frame):
    data_frame.Age = data_frame.Age.astype(int);
def update_columns_type(data_frame):    
    data_frame.Pclass = data_frame.Pclass.astype(int);
    data_frame.Sex = data_frame.Sex.astype(int);    
    data_frame.update(data_frame.Embarked.apply(lambda x: ord(x)));
    data_frame.Embarked = data_frame.Embarked.astype(int);
    
update_column_sex(train);
update_survived_column_type(train);
update_age_column_type(train)
update_columns_type(train);
train.info()
train.head()
def getDictSuvived(dataframe):
    """
    - Returns a dictionary where each key is the index of each row of the dataframe
      passed by parameter and the value is the value of each column        
    """
    dict_dataframe = {}    
        
    for x, y in dataframe.items():
        dict_dataframe[x] = y;
        
    return dict_dataframe
%matplotlib inline
import matplotlib.pyplot as plt

# Survived by class
dict_survived = getDictSuvived(train[train.Survived == 1].groupby(by='Pclass').count()['PassengerId']);
dict_not_survived = getDictSuvived(train[train.Survived == 0].groupby(by='Pclass').count()['PassengerId']);

plt.bar(list(dict_survived.keys()), list(dict_survived.values()), .8, alpha=0.5, color='g', label='Survived');
plt.bar(list(dict_not_survived.keys()), list(dict_not_survived.values()), .5, alpha=0.5, color='b', label='Not survived');
plt.xticks(np.arange(3) + 1, ('1º Class', '2º Class', '3º Class'));
plt.legend();
SibSp_dict_survived = getDictSuvived(train[train.Survived == 1].groupby(by='Pclass').sum()['SibSp']);
SibSp_dict_not_survived = getDictSuvived(train[train.Survived == 0].groupby(by='Pclass').sum()['SibSp']);

plt.bar(list(SibSp_dict_survived.keys()), list(SibSp_dict_survived.values()), .8, alpha=0.5, color='g', label='Survived');
plt.bar(list(SibSp_dict_not_survived.keys()), list(SibSp_dict_not_survived.values()), .5, alpha=0.5, color='b', label='Not survived');
plt.xticks(np.arange(3) + 1, ('1º C. Siblings', '2º C. Siblings', '3º C. Siblings'));
plt.legend();
parch_dict_survived = getDictSuvived(train[train.Survived == 1].groupby(by='Pclass').sum()['Parch']);
parch_dict_not_survived = getDictSuvived(train[train.Survived == 0].groupby(by='Pclass').sum()['Parch']);

plt.bar(list(parch_dict_survived.keys()), list(parch_dict_survived.values()), .8, alpha=0.5, color='g', label='Survived');
plt.bar(list(parch_dict_not_survived.keys()), list(parch_dict_not_survived.values()), .5, alpha=0.5, color='b', label='Not survived');
plt.xticks(np.arange(3) + 1, ('1º C. Parents', '2º C. Parents', '3º C. Parents'));
plt.legend();
survided_male = train[(train.Survived == 1) & (train.Sex == 1)]
survided_female = train[(train.Survived == 1) & (train.Sex == 2)]

male_dict_survived = getDictSuvived(survided_male.groupby(by='Pclass').count()['Sex']);
female_dict_survived = getDictSuvived(survided_female.groupby(by='Pclass').count()['Sex']);

plt.bar(list(male_dict_survived.keys()), list(male_dict_survived.values()), .8, alpha=0.5, color='g', label='Male survived');
plt.bar(list(female_dict_survived.keys()), list(female_dict_survived.values()), .5, alpha=0.5, color='b', label='Female survived');
plt.xticks(np.arange(3) + 1, ('1º Class', '2º Class', '3º Class'));
plt.legend();
train.head()
random_seed = 10
target = 'Survived'
predictors = ['Pclass', 'Sex', 'Fare', 'Embarked']

Y = train[target]
X = train[predictors]
X_test = test[predictors]
Id_test = test['PassengerId']

X_train, X_val, y_train, y_val = train_test_split(X, Y, random_state=random_seed)
def get_cross_val_score(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, random_state=2)    
    return cross_val_score(model ,train_X, train_y, cv=5).mean()   

candidates_max_leaf_nodes = list(range(2,600, 10))
max_leaf_nodes_value_predicted = []

for max_leaf_nodes in candidates_max_leaf_nodes:
    my_mae = get_cross_val_score(max_leaf_nodes, X_train, X_val, y_train, y_val)
    max_leaf_nodes_value_predicted.append(my_mae)    
    
max_leaf_nodes_index = max_leaf_nodes_value_predicted.index(max(max_leaf_nodes_value_predicted));
max_leaf_nodes = candidates_max_leaf_nodes[max_leaf_nodes_index];

print('Best max leaf nodes: {}'.format(max_leaf_nodes));
print('Score: {}'.format(max(max_leaf_nodes_value_predicted)));
print('');
def training_accuracy(max_leaf_nodes, train_X, val_X, train_y, val_y):    
    model = RandomForestClassifier(max_leaf_nodes=max_leaf_nodes, random_state=2)
    model.fit(train_X, train_y)    
    return [model, model.score(train_X, train_y)]

model, accuracy = training_accuracy(max_leaf_nodes, X_train, X_val, y_train, y_val)
print("Training accuracy {:.5f}".format(accuracy));

def test_accuracy(model, val_X, val_y):
    predictions = model.predict(val_X)    
    return [predictions, precision_score(val_y, predictions)]   

predictions, accuracy = test_accuracy(model, X_val, y_val);
print("Test accuracy {:.5f}".format(accuracy));
update_fare_column(X_test);
X_test.update(UpdateNullEmbarked(X_test))
update_column_sex(X_test);
update_columns_type(X_test);
predictions = model.predict(X_test);
dt = pd.DataFrame({"PassengerId": Id_test, "Survived": predictions})
dt.to_csv("submit.csv", index = False)