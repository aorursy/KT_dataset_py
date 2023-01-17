import pandas as pd # pandas reads and manipulates the data  
import numpy as np  # numpy is basically a package for calculation 
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(2)
train.isnull().sum()
del train['PassengerId']
del train['Embarked'] 
# Embarked means where did people get to the chip from (and that has no indication on the data in a logical manner so we delete it )
# map Sex
train.Sex=train.Sex.map({'male':1,'female':2})
# use the name to get (mr.,ms. ... etc)
train.head(2)
import re 

def search_pattern(index):
    return re.search(',.*[/.\b]',train.Name[index])[0][2:-1]

train['Social_name']=[search_pattern(counter) for counter in range(train.shape[0]) ]

train.Social_name.unique() # see the unique values from the Social_name column 
# cleaning the things that the regex couldn't get
train.Social_name.replace({"""Mrs. Martin (Elizabeth L""":'Mrs',
                       'Mme':'Miss',
                       'Ms':'Miss',
                       'the Countess':'Countess',
                        'Mr. Carl':'Mr',
                        'Mlle':'Miss'},inplace=True)
train.Social_name.unique()
train.head(2)
train.Social_name.unique()
# encoding the values into numbers
for index in range(len(train.Social_name.unique())):
    
    a=train.Social_name.unique()[index] # the string (e.g. (Mr.))
    train.Social_name.replace({a:index},inplace=True)

# delete the name because we don't need it anymore 
del train['Name']

train.head(2)
train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median() 
# this is why we use the 3 of them to get the best median ever to fill the nans with 
# fill na age values given (Pclass , Sex , Social_name)
grouped=train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()

pclass=grouped.index.labels[0] 

sex=grouped.index.labels[1] 

social_name=grouped.index.labels[2]



for counter in range(len(grouped.index.labels[1])):
    # HERE
    train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
              (train.Sex==train.Sex.unique()[sex[counter]]) &
              (train.Social_name==train.Social_name.unique()[social_name[counter]])),
              'Age']=\
    train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
              (train.Sex==train.Sex.unique()[sex[counter]]) &
              (train.Social_name==train.Social_name.unique()[social_name[counter]])),
              'Age'].fillna(value=grouped.values[counter])
    # THERE

# from HERE to THERE is the same as putting inplace=True on the fillna but it seems that inplace doesn't work for no specific reason . . . 
 # 
train.head(2)
# name the cabin with it's first letter 

for x in range(len(train)):
    if pd.isnull(train.loc[x,'Cabin']):
        continue  # pass the nan values 
    else : 
        train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]

# filling the nan cabin with a defaulted value 
train.Cabin.fillna('N',inplace=True)

# add dummies to the data and concating them to the origional dataset 
train = pd.concat([train, pd.get_dummies(train.Cabin)], axis=1)

# delete the nan values and the origional Cabin column 
del train['N']
del train['Cabin']
train.head(2)
len(train.Ticket.unique())
 # i can't see any useful information from keeping the tickets so i will just delete it 
del train['Ticket']
train.head(2)
train.isnull().sum()
def main(train):
    
    import numpy as np 

    # delete the passenger id since it gives no indication on the data whatsoever 

    del train['PassengerId']
    # map Sex
    train.Sex=train.Sex.map({'male':1,'female':2})
    # use the name to get (mr.,ms. ... etc)
    import re 

    def search_pattern(index):
        return re.search(',.*[/.\b]',train.Name[index])[0][2:-1]

    train['Social_name']=[search_pattern(counter) for counter in range(train.shape[0]) ]

    # cleaning the things that the regex couldn't get 
    train.Social_name.replace({"""Mrs. Martin (Elizabeth L""":'Mrs',
                           'Mme':'Miss',
                           'Ms':'Miss',
                           'the Countess':'Countess',
                            'Mr. Carl':'Mr',
                            'Mlle':'Miss'},inplace=True)

    # mapping the values 
    for x in range(len(train.Social_name.unique())):
        a=train.Social_name.unique()[x]
        b=x

        train.Social_name.replace({a:b},inplace=True)


    # delete the name because we don't need it anymore 
    del train['Name']

    # fill na age values given (Pclass , Sex , Social_name)
    grouped=train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()

    pclass=grouped.index.labels[0] ; sex=grouped.index.labels[1] ; social_name=grouped.index.labels[2]


    for counter in range(len(grouped.index.labels[1])):
        # HERE
        train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
                  (train.Sex==train.Sex.unique()[sex[counter]]) &
                  (train.Social_name==train.Social_name.unique()[social_name[counter]])),
                  'Age']=\
        train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
                  (train.Sex==train.Sex.unique()[sex[counter]]) &
                  (train.Social_name==train.Social_name.unique()[social_name[counter]])),
                  'Age'].fillna(value=grouped.values[counter])
        # THERE

    # from HERE to THERE is the same as putting inplace=True on the fillna but it seems that inplace doesn't work for no specific reason . . . 



    # map Embarked 
    train.Embarked=train.Embarked.map({'S':1,'C':2,'Q':3})

    # fill Embarked nans 
    train.Embarked.groupby(train.Embarked).count() # the max is 1 so we fill the nans with it 
    train.Embarked.fillna(1,inplace=True)

    # name the cabin with it's first letter 

    for x in range(len(train)):
        if pd.isnull(train.loc[x,'Cabin']):
            continue 
        else : 
            train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]

    # filling the nan cabin with a defaulted value 
    train.Cabin.fillna('N',inplace=True)

    # add dummies to the data and concating them to the origional dataset 
    train = pd.concat([train, pd.get_dummies(train.Cabin)], axis=1)

    # delete the nan values and the origional Cabin column 
    del train['N']
    del train['Cabin']
    

    


    # rounding the ages 

    train.Age=train.Age.values.round().astype(int)

     # i can't see any useful information from keeping the tickets so i will just delete it 

    del train['Ticket']

    # rounding the fares to give less unique numbers 

    train.Fare=train.Fare.round().astype(int)

    # Embarked means where did people get to the chip from (and that has no indication on the data in a logical manner so we delete it )
    del train['Embarked']
    return train
df_train = pd.read_csv('../input/train.csv')
 # read the tainng data 
df_train=main(df_train) # pass the data to the manipulating function
# there is train['T'] but there is only one value in the whole data so we will delete it 
#model_training['T'].sum() # unhash the line to see the number 
del df_train['T']
model_training=df_train.loc[:,df_train.columns!='Survived']
model_testing=df_train.loc[:,'Survived']
df_test= pd.read_csv('../input/test.csv')
# df_test=main(df_test) # this code doesn't work because there is an nan value in the Fare (just one nan)
df_test.Fare.fillna(df_test.Fare.median(skipna=True),inplace=True) # filling the nan value with the median fare 
df_test=main(df_test) # now it works 
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

the_model=logistic.fit(model_training,model_testing)
results=the_model.predict(df_test) # the output is returned as a list 
final=pd.read_csv('../input/test.csv',usecols=['PassengerId'])
final['Survived']=results

final=final.set_index('PassengerId')

final.to_csv('final.csv') # 0.77511 on kaggle for that 