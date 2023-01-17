!pip install Faker
import pandas as pd 

# read in the Titanic training data csv file

train_data = pd.read_csv('../input/titanic/train.csv')

# take a look

train_data.head(10)
from faker import Faker

fake = Faker()



def Sex(row):

    if row['Sex'] == 'female':

        new_name = fake.name_female()

    else:

        new_name = fake.name_male()

    return new_name



train_data['Name'] = train_data.apply(Sex, axis=1)



# take a quick look

train_data.head(10)
train_data.to_csv('pseudonymized_train.csv', index=False)