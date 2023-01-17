###import numpy, pandas, matplotlib, seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn import preprocessing
###Adaline
class Ada_(object):
    
    def __init__(self, eta = 0.1, nterm = 50, rands = 1):
        self.eta = eta
        self.nterm = nterm
        self.rands = rands
        
    def fit(self, x, y):
        rgen = np.random.RandomState(self.rands)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost_ = []
        
        for i in range(self.nterm):
            output = self.n_input(x)
            error = y - output
            self.w_[1:] += self.eta * x.T.dot(error)
            self.w_[0] += self.eta * error.sum()
            cost = (error ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
            
    def n_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predict(self, x):
        return np.where(self.n_input(x) >= 0.0, 1, -1)
###get titanic & test csv file as DataFrame
tit_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

#preview data file
tit_df.head()
###Preview for missing datas
tit_df.info()
print("----------------------")
test_df.info()
###drop unnecessary infomation by columns
tit_df = tit_df.drop(['PassengerId', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)
test_df = test_df.drop(['Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)
###Start Processing data
##Name

#Find out master or miss or mlle, then drop Name
def get_mmm(passenger):
    name = passenger
    if(   ('Master' in str(name)) \
       or ('Miss'   in str(name)) \
       or ('Mlle'   in str(name))):
        return 1
    else:
        return 0

tit_df['MMM'] = tit_df[['Name']].apply(get_mmm, axis = 1)
test_df['MMM'] = test_df[['Name']].apply(get_mmm, axis = 1)

tit_df  = tit_df.drop(['Name'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
##Embarked

#Fill in the null with the most commen letter 'S'
#tit_df["Embarked"] = tit_df["Embarked"].fillna("S")

#Create dummy variables for Embarked
#emb_dummy_tit = pd.get_dummies(tit_df["Embarked"])
#emb_dummy_test = pd.get_dummies(test_df["Embarked"])

#tit_df = tit_df.join(emb_dummy_tit)
#test_df = test_df.join(emb_dummy_test)

#tit_df = tit_df.drop(["Embarked"], axis = 1)
#test_df = test_df.drop(["Embarked"], axis = 1)
##Pclass

sns.factorplot('Pclass', 'Survived', order = [1,2,3], data = tit_df, size = 4)

#Create dummy variables for Pclass
pc_dummy_tit = pd.get_dummies(tit_df['Pclass'])
pc_dummy_test = pd.get_dummies(test_df['Pclass'])

pc_dummy_tit.columns = ['Class_1', 'Class_2', 'Class_3']
pc_dummy_test.columns = ['Class_1', 'Class_2', 'Class_3']

tit_df = tit_df.join(pc_dummy_tit)
test_df = test_df.join(pc_dummy_test)

#tit_df.drop(['Pclass'], axis = 1, inplace = True)
#test_df.drop(['Pclass'], axis = 1, inplace = True)
##Sex

#Create dummy variables for Sex, then drop Sex
sex_dummy_tit = pd.get_dummies(tit_df['Sex'])
sex_dummy_test = pd.get_dummies(test_df['Sex'])

tit_df = tit_df.join(sex_dummy_tit)
test_df = test_df.join(sex_dummy_test)

tit_df = tit_df.drop(['Sex'], axis = 1)
test_df = test_df.drop(['Sex'], axis = 1)
##Parch

#Seperate passengers who has parent or children with passengers who doesn't
def Pch_sep(passenger):
    parch = passenger
    if(parch > 0):
        return 1
    else:
        return 0

tit_df['Parch'] = tit_df['Parch'].apply(Pch_sep)
test_df['Parch'] = test_df['Parch'].apply(Pch_sep)
##Now I have the idea 
#Maybe the servival rate will be higher for children than for parents
#Because parents will try their best to let their children live
#Also high class passengers and female passengers should have more chance to survive

#So I make following assumption:
#Passengers' name with Master, Miss, Mlle should more likely to be children or teenagers
#If their Parch value is 1, means they should be child of someone
#So they may have a batter survival rate
print("Amound of people under each condition:")
table0e = pd.pivot_table(tit_df, values = 'Survived', \
                    index = ['Parch', 'female', 'MMM'], \
                    columns=['Pclass'], \
                    aggfunc='count')
print( table0e.iloc[::-1],'\n' )

print("Percent of survival under each condition:")
table0e = pd.pivot_table(tit_df, values = 'Survived', \
                    index = ['Parch', 'female', 'MMM'], \
                    columns=['Pclass'], \
                    aggfunc=np.mean)
print( table0e.iloc[::-1],'\n' )
##Age
#77 missing value in tit_df, 86 missing value in test_df
#Need to fill in the values
#Passengers with Master, Miss, and Melle are mostly children and teenagers
#Which means to fill in people who are MMM need a smaller random number
#Others will need larger random numbers
#Find the age distribution of MMM and non-MMM
facet = sns.FacetGrid(tit_df, hue = "MMM",aspect = 5)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, tit_df['Age'].max()))
facet.add_legend()

facet = sns.FacetGrid(test_df, hue = "MMM",aspect = 5)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, tit_df['Age'].max()))
facet.add_legend()

#Show average survived passengers by age
fig, axis1 = plt.subplots(1, 1, figsize = (30,5))
avg_age = tit_df[["Age", "Survived"]].groupby(['Age'], as_index = False).mean()
sns.barplot(x='Age', y='Survived', data = avg_age)
#For MMM members set a random age range about 0-30
#For non-MMM memebers set a random age range about 15-50

#Generate ramdom numbers
#Find out the amount of people who have or don't have MMM in their names and have a missing age
mmm = 0
nmmm = 0
for m in range(891):
    if(tit_df.loc[m, 'MMM'] == 1 \
       and np.isnan(tit_df.loc[m, 'Age'])):
        mmm += 1
    if(tit_df.loc[m, 'MMM'] == 0 \
       and np.isnan(tit_df.loc[m, 'Age'])):
        nmmm += 1
    else:
        0

#Generate radom numbers based on the amout above
rand_MMM_tit = np.random.randint(0, 32, size = mmm)
rand_NMMM_tit = np.random.randint(15, 50, size = nmmm)

mmm = 0
nmmm = 0
for m in range(418):
    if(test_df.loc[m, 'MMM'] == 1 \
       and np.isnan(test_df.loc[m, 'Age'])):
        mmm += 1
    if(test_df.loc[m, 'MMM'] == 0 \
       and np.isnan(test_df.loc[m, 'Age'])):
        nmmm += 1
    else:
        0

rand_MMM_test = np.random.randint(0, 35, size = mmm)
rand_NMMM_test = np.random.randint(16, 50, size = nmmm)

tit_df['Age'].dropna().astype(int)
test_df['Age'].dropna().astype(int)

#Replace all the Null
for p in range(891):
    is_m = 0
    not_m = 0
    
    if(tit_df.loc[p, 'MMM'] == 1 \
       and np.isnan(tit_df.loc[p, 'Age'])):
        tit_df.loc[p, 'Age'] = rand_MMM_tit[is_m]
        is_m += 1
        
    if(tit_df.loc[p, 'MMM'] == 0 \
       and np.isnan(tit_df.loc[p, 'Age'])):
        tit_df.loc[p, 'Age'] = rand_NMMM_tit[not_m]
        not_m += 1
    
    else:
        0
        
for p in range(418):
    is_m = 0
    not_m = 0
    
    if(test_df.loc[p, 'MMM'] == 1 \
       and np.isnan(test_df.loc[p, 'Age'])):
        test_df.loc[p, 'Age'] = rand_MMM_test[is_m]
        is_m += 1
        
    if(test_df.loc[p, 'MMM'] == 0 \
       and np.isnan(test_df.loc[p, 'Age'])):
        test_df.loc[p, 'Age'] = rand_NMMM_test[not_m]
        not_m += 1
    
    else:
        0
##SibSp

#Seperate passengers who has parent or children with passengers who doesn't
def Pch_sep(passenger):
    sibsp = passenger
    if(sibsp > 0):
        return 1
    else:
        return 0

tit_df['SibSp'] = tit_df['SibSp'].apply(Pch_sep)
test_df['SibSp'] = test_df['SibSp'].apply(Pch_sep)

print("Amound of people under each condition:")
table0e = pd.pivot_table(tit_df, values = 'Survived', \
                    index = ['SibSp', 'female'], \
                    columns=['Pclass'], \
                    aggfunc='count')
print( table0e.iloc[::-1],'\n' )

print("Percent of survival under each condition:")
table0e = pd.pivot_table(tit_df, values = 'Survived', \
                    index = ['SibSp', 'female'], \
                    columns=['Pclass'], \
                    aggfunc=np.mean)
print( table0e.iloc[::-1],'\n' )
#It seems SibSp is not really the feature affecting survival rate

#Since most training data is 1 and 0
#In order to use Adline, data should not have very large differenc
#Which means age values are too extreme need to be rearange to between 0-1
def norm_age(x):
    age = x
    return age/80

tit_df['Age'] = tit_df['Age'].apply(norm_age)
test_df['Age'] = test_df['Age'].apply(norm_age)

#Finally drop column that are just used for preview datas: Pclass
#Also drop column that are not very much affecting their survival rates: SibSp
tit_df = tit_df.drop(['Pclass', 'SibSp'], axis = 1)
test_df = test_df.drop(['Pclass', 'SibSp'], axis = 1)

#Review data file
print(tit_df.head())
print(test_df.head())
#define true class label & training data
y = tit_df.iloc[0:891, 0].values
y = np.where(y == 1, 1, -1)

x = tit_df.iloc[0:891, [1,8]].values

#read test data
x_t = test_df.iloc[0:418, [1,8]].values
#train with adaline
ada_tit = Ada_(nterm = 200, eta = 0.0001).fit(x, y)

#plot the training cost
plt.plot(range(1, len(ada_tit.cost_) + 1), ada_tit.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()
#make prediction & change data for submition
y_predict = ada_tit.predict(x_t)

y_predict = pd.DataFrame(y_predict).replace(-1, '0')
y_predict = np.array(y_predict).flatten()
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('titanic.csv', index=False)