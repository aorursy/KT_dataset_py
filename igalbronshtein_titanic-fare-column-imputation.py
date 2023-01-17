import pandas as pd
titanic_test = pd.read_csv("../input/titanic/test.csv") 

# Because it's a file for test. Train file was used for calculation.

titanic_train = pd.read_csv("../input/titanic/train.csv")

# Passenger with missing Fare

the_passenger = titanic_test[titanic_test.Fare.isna()].style.highlight_null(null_color='gainsboro'); the_passenger
first = round(titanic_train.Fare.mean(),4); first 
first_class_mean = round(titanic_train[titanic_train.Pclass == 1].Fare.mean(),4)

second_class_mean = round(titanic_train[titanic_train.Pclass == 2].Fare.mean(),4)

third_class_mean = round(titanic_train[titanic_train.Pclass == 3].Fare.mean(),4)
first_class_mean, second_class_mean, third_class_mean
# It can give a more accurate result. If use mean, only for those with the same class as the passenger.

second = third_class_mean; second
titanic_train.Age.describe()
the_passenger
filter_class = titanic_train.Pclass == 3;

filter_sex = titanic_train.Sex == "male"; 

# Age of the passeger: 60.5. 75% percentile = 38

filter_age = titanic_train.Age > titanic_train.Age.quantile(0.75);

filter_SibSp = titanic_train.SibSp == 0;

filter_Parch = titanic_train.Parch == 0;

filter_Embarked = titanic_train.Embarked == 'S';
filtered_titanic = titanic_train[filter_class & filter_sex & filter_age & filter_SibSp & filter_Parch & filter_Embarked]; filtered_titanic
round(filtered_titanic.Fare.mean(),4)
# There two lines that can be removed. Only one passenger has info about cabin.

# Only one passenger with Fare zero and uncommon ticket "LINE".

filtered_titanic = filtered_titanic[(filtered_titanic.Ticket != "LINE") & (filtered_titanic.Cabin.isna())]

filtered_titanic
final_mean_result = round(filtered_titanic.Fare.mean(),4); third = final_mean_result

print('\u001b[1m'+ str(final_mean_result) +'\x1b[0m') 
# Results of all levels

first, second, third