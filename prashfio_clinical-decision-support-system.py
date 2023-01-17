#imported important libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

import warnings

warnings.filterwarnings('ignore')
#reading the dataset

df = pd.read_csv('../input/Training.csv')
df.head()
#They are 4920 rows, 133 columns

df.shape
#seeing any null values are there with descending format

df.isnull().sum().sort_values(ascending=False)
df.columns
#looking how much percent each diseases having

df['prognosis'].value_counts(normalize = True)
#as we can see each no. diseases having the same percentage through bar chart

df['prognosis'].value_counts(normalize = True).plot.bar()

plt.subplots_adjust(left = 0.9, right = 2 , top = 2, bottom = 1)
#checking if there are any other data types

df.dtypes.unique()
#Analyzing each symptoms/variable

for x in range(df.shape[1]):

    plt.subplot(7,22,x+1)

    plt.subplots_adjust(left = 0.5, right = 16 , top = 10, bottom = 0.5)

    sns.countplot(df[df.columns[x]]).set_title(df.columns[x],fontsize=23)
#checking the relationship between the variables by applying the correlation 

corr = df.corr()

mask = np.array(corr)

mask[np.tril_indices_from(mask)] = False

plt.subplots_adjust(left = 0.5, right = 16 , top = 20, bottom = 0.5)

sns.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
#took two high correlation variables and analysing if it is satisfying null hypothesis or alternate hypothesis

pd.crosstab(df['cold_hands_and_feets'],df['weight_gain'])
#imported the chi square contingency

from scipy.stats import chi2_contingency
#as p value is  0.0  which is less than 0.05 then they are actually different from each other which satisfy the alternate hypothesis 

chi2_contingency(pd.crosstab(df['cold_hands_and_feets'],df['weight_gain']))
#seperated the independent and dependent values to repective variables 

x = df.drop(['prognosis'],axis =1)

y = df['prognosis']
from sklearn.model_selection import train_test_split
#divided into testing and training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#imported naive_baye algorithm

from sklearn.naive_bayes import MultinomialNB
#fitted the model

mnb = MultinomialNB()

mnb = mnb.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
y_pred = mnb.predict(x_test)
accuracy_score(y_pred,y_test)
#by cross validating we got mean also 100%

from sklearn.model_selection import cross_val_score

scores = cross_val_score(mnb, x_test, y_test, cv=3)

print (scores)

print (scores.mean())
real_diseases = y_test.values
#for the cross checking purpose i want to see if predicted values and actual values are same else it gives me worng prediction 

for i in range(0, len(real_diseases)):

    if y_pred[i] == real_diseases[i]:

        print ('Pred: {0} Actual:{1}'.format(y_pred[i], real_diseases[i]))

    else:

        print('worng prediction')

        print ('Pred: {0} Actual:{1}'.format(y_pred[i], real_diseases[i]))
#imported Kfold

from sklearn.model_selection import KFold
## Function to run multiple algorithms with different K values of KFold.

def evaluate(train_data,kmax,algo):

    test_scores = {}

    train_scores = {}

    for i in range(2,kmax,2):

        kf = KFold(n_splits = i)

        sum_train = 0

        sum_test = 0

        data = df

        for train,test in kf.split(data):

            train_data = data.iloc[train,:]

            test_data = data.iloc[test,:]

            x_train = train_data.drop(["prognosis"],axis=1)

            y_train = train_data['prognosis']

            x_test = test_data.drop(["prognosis"],axis=1)

            y_test = test_data["prognosis"]

            algo_model = algo.fit(x_train,y_train)

            sum_train += algo_model.score(x_train,y_train)

            y_pred = algo_model.predict(x_test)

            sum_test += accuracy_score(y_test,y_pred)

        average_test = sum_test/i

        average_train = sum_train/i

        test_scores[i] = average_test

        train_scores[i] = average_train

        print("kvalue: ",i)

    return(train_scores,test_scores)   
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier()

nb = MultinomialNB()

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy',)

from sklearn.ensemble import RandomForestClassifier

ran = RandomForestClassifier(n_estimators = 10)
algo_dict = {'l_o_g':log,'d_t':dt,'r_a_n':ran,'N_B' : nb}

algo_train_scores={}

algo_test_scores={}
#decision tree was found to be best fit with training score of 0.1 and testing score of 0.87 with k value of 2 in the k fold cross validation. All the other algorithm seems to be overfit.

max_kfold = 11

for algo_name in algo_dict.keys():

    print(algo_name)

    tr_score,tst_score = evaluate(df,max_kfold,algo_dict[algo_name])

    algo_train_scores[algo_name] = tr_score

    algo_test_scores[algo_name] = tst_score

print(algo_train_scores)

print(algo_test_scores)
df_test = pd.DataFrame(algo_test_scores)

df_train = pd.DataFrame(algo_train_scores)
df_test.plot(grid = 1)

plt.show()

plt.grid()
#building the model at k value 2 

test_scores={}

train_scores={}

for i in range(2,4,2):

    kf = KFold(n_splits = i)

    sum_train = 0

    sum_test = 0

    data = df

    for train,test in kf.split(data):

        train_data = data.iloc[train,:]

        test_data = data.iloc[test,:]

        x_train = train_data.drop(["prognosis"],axis=1)

        y_train = train_data['prognosis']

        x_test = test_data.drop(["prognosis"],axis=1)

        y_test = test_data["prognosis"]

        algo_model = dt.fit(x_train,y_train)

        sum_train += dt.score(x_train,y_train)

        y_pred = dt.predict(x_test)

        sum_test += accuracy_score(y_test,y_pred)

    average_test = sum_test/i

    average_train = sum_train/i

    test_scores[i] = average_test

    train_scores[i] = average_train

    print("kvalue: ",i)  
print(train_scores)

print(test_scores)
#saved the model 

from sklearn.externals import joblib

joblib.dump(dt,'my_model_for_healthcare')
a = list(range(2,134))

i_name  = (input('Enter your name :'))

i_age = (int(input('Enter your age:')))

for i in range(len(x.columns)):

    print(str(i+1+1) + ":", x.columns[i])

choices = input('Enter the Serial no.s which is your Symptoms are exist:  ')

b = [int(x) for x in choices.split()]

count = 0

while count < len(b):

    item_to_replace =  b[count]

    replacement_value = 1

    indices_to_replace = [i for i,x in enumerate(a) if x==item_to_replace]

    count += 1

    for i in indices_to_replace:

        a[i] = replacement_value

a = [0 if x !=1 else x for x in a]

y_diagnosis = dt.predict([a])

y_pred_2 = dt.predict_proba([a])

print(('Name of the infection = %s , confidence score of : = %s') %(y_diagnosis[0],y_pred_2.max()* 100),'%' )

print(('Name = %s , Age : = %s') %(i_name,i_age))