import numpy as np

import pandas as pd
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data[:50]
train_data.describe(include='all')
women = train_data[train_data['Sex'] == 'female']['Survived']

rate_women = sum(women)/len(women)

print('% of women who survived:', rate_women)
men = train_data[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print('% of men who survived:', rate_men)
# added code

train_data['Survived'].mean()
# alternative way of computing the above

train_data[['Sex', 'Survived']].groupby(['Sex']).mean()
#added code

train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean()
#added code

train_data[['Cabin', 'Survived']].groupby(['Cabin']).mean()
#added code

train_data[['SibSp', 'Survived']].groupby(['SibSp']).mean()
#added code

train_data[['Parch', 'Survived']].groupby(['Parch']).mean()
#added code

train_data[['Age', 'Survived']].groupby(['Age']).mean()
train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()
# generate correlation data (larger values signify a clear positive/negative correlation between row/column labels)

train_data.corr()
women_count = 0

women_survived_count = 0

for idx, row in train_data.iterrows():

    if row['Sex'] == 'female':

        women_count += 1

        if row['Survived'] == 1:

            women_survived_count += 1

women_survived_count / women_count
#added code

age_range_num = 0

age_range_survive = 0



for idx, row in train_data.iterrows():

    if row['Age'] > 50:

        if row['Age'] < 100:

            age_range_num += 1

            if row['Survived'] == 1:

                age_range_survive += 1

age_range_survive/age_range_num
#added code

fare_range_num = 0

fare_range_survive = 0



for idx, row in train_data.iterrows():

    if row['Fare'] > 100:

        if row['Fare'] < 1000:

            fare_range_num += 1

            if row['Survived'] == 1:

                fare_range_survive += 1

fare_range_survive/fare_range_num
predictions = []



for idx, row in test_data.iterrows():

    # make your changes in this cell!

   

    weight_sum_max = 8

    sex_weight = 0.0

    pclass_weight = 0.0

    age_weight = 0.0

    sibsp_weight = 0.0

    parch_weight = 0.0

    fare_weight = 0.0

    embark_weight = 0.0

    weight_sum_act = 0.0

    

    if row['Sex'] == 'female':

        sex_weight = 0.74

    else:

        sex_weight = 0.19



    if row['Pclass'] == 1:

        pclass_weight = 0.63

    elif row['Pclass'] == 2:

        pclass_weight = 0.47

    else:

        pclass_weight = 0.24

        

    if row['Age'] > 0:

        if row['Age'] < 10:

            age_weight = 0.61

    elif row['Age'] > 10:

        if row['Age'] < 30:

            age_weight = 0.38

    elif row['Age'] > 30:

        if row['Age'] < 50:

            age_weight = 0.42

    else:

        age_weight = 0.34

        

    if row['SibSp'] == 0:

        sibsp_weight = 0.34

    elif row['SibSp'] == 1:

        sibsp_weight = 0.53

    elif row['SibSp'] == 2:

        sibsp_weight = 0.46

    elif row['SibSp'] == 3:

        sibsp_weight = 0.25

    elif row['SibSp'] == 4:

        sibsp_weight = 0.16

    else:

        sibsp_weight = 0.0

        

    if row['Parch'] == 0:

        parch_weight = 0.34

    elif row['Parch'] == 1:

        parch_weight = 0.55

    elif row['Parch'] == 2:

        parch_weight = 0.5

    elif row['Parch'] == 3:

        parch_weight = 0.6

    elif row['Parch'] == 5:

        parch_weight = 0.2

    else:

        parch_weight = 0.0

        

    if row['Fare'] > 0:

        if row['Fare'] < 20:

            fare_weight = 0.28

    if row['Fare'] > 20:

        if row['Fare'] < 50:

            fare_weight = 0.48

    if row['Fare'] > 50:

        if row['Fare'] < 600:

            fare_weight = 0.61

        

    if row['Embarked'] == 'C':

        embark_weight = 0.55

    elif row['Embarked'] == 'Q':

        embark_weight = 0.39

    else:

        embark_weight = 0.33

        

    weight_sum_act = sex_weight + pclass_weight + age_weight + sibsp_weight + parch_weight + fare_weight + embark_weight

    if row['Sex'] == 'female':

        if (weight_sum_act/weight_sum_max) >= 0.31:

            predictions.append(1)

        else:

            predictions.append(0)

    else:        

        if (weight_sum_act/weight_sum_max) >= 0.36:

            predictions.append(1)

        else:

            predictions.append(0)

assert len(predictions) == len(test_data), 'Number of predictions must match number of test data rows!'
test_data['Survived'] = predictions
test_data[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
#added code

#accuracy test

predictions_test = []



for idx, row in train_data.iterrows():

    

    weight_sum_max = 8

    sex_weight = 0.0

    pclass_weight = 0.0

    age_weight = 0.0

    sibsp_weight = 0.0

    parch_weight = 0.0

    fare_weight = 0.0

    embark_weight = 0.0

    weight_sum_act = 0.0

    

    if row['Sex'] == 'female':

        sex_weight = 0.74

    else:

        sex_weight = 0.19



    if row['Pclass'] == 1:

        pclass_weight = 0.63

    elif row['Pclass'] == 2:

        pclass_weight = 0.47

    else:

        pclass_weight = 0.24

        

    if row['Age'] > 0:

        if row['Age'] < 10:

            age_weight = 0.61

    elif row['Age'] > 10:

        if row['Age'] < 30:

            age_weight = 0.38

    elif row['Age'] > 30:

        if row['Age'] < 50:

            age_weight = 0.42

    else:

        age_weight = 0.34

        

    if row['SibSp'] == 0:

        sibsp_weight = 0.34

    elif row['SibSp'] == 1:

        sibsp_weight = 0.53

    elif row['SibSp'] == 2:

        sibsp_weight = 0.46

    elif row['SibSp'] == 3:

        sibsp_weight = 0.25

    elif row['SibSp'] == 4:

        sibsp_weight = 0.16

    else:

        sibsp_weight = 0.0

        

    if row['Parch'] == 0:

        parch_weight = 0.34

    elif row['Parch'] == 1:

        parch_weight = 0.55

    elif row['Parch'] == 2:

        parch_weight = 0.5

    elif row['Parch'] == 3:

        parch_weight = 0.6

    elif row['Parch'] == 5:

        parch_weight = 0.2

    else:

        parch_weight = 0.0

        

    if row['Fare'] > 0:

        if row['Fare'] < 20:

            fare_weight = 0.28

    if row['Fare'] > 20:

        if row['Fare'] < 50:

            fare_weight = 0.48

    if row['Fare'] > 50:

        if row['Fare'] < 600:

            fare_weight = 0.61

        

    if row['Embarked'] == 'C':

        embark_weight = 0.55

    elif row['Embarked'] == 'Q':

        embark_weight = 0.39

    else:

        embark_weight = 0.33

        

    weight_sum_act = sex_weight + pclass_weight + age_weight + sibsp_weight + parch_weight + fare_weight + embark_weight

    if row['Sex'] == 'female':

        if (weight_sum_act/weight_sum_max) >= 0.31:

            predictions_test.append(1)

        else:

            predictions_test.append(0)

    else:        

        if (weight_sum_act/weight_sum_max) >= 0.36:

            predictions_test.append(1)

        else:

            predictions_test.append(0)



count = 0

index = 0



for idx, row in train_data.iterrows():

    if index < len(predictions_test)-1:

        if row['Survived'] == predictions_test[index]:

            count += 1

        index += 1

count/len(train_data['Survived'])
#added

test_data['Survived'].mean()

#added

train_data