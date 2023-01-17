%matplotlib inline



#At first we need some libraries

#Сначала - загрузка необходимых библиотек

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
#Loading of the data

#Загрузка данных

titanic_train = pd.read_csv(r'../input/train.csv')

titanic_test = pd.read_csv(r'../input/test.csv')
#The Test dataset gets Survived feature to make its feature set same as feature set of the Train dataset.

#Survived добавляется к тестовым данным, чтобы сделать одинаковый набор свойств у тестовых и тренировочных данных.

titanic_test['Survived'] = -1



#Creating one combined dataset

#Объединение всех записей

titanic_full = pd.concat([titanic_train, titanic_test])
#Data sample

#Первые нескольких записей.

titanic_full.head()
#Description of the numeric features.

#We can see missing values in Age and Fare. Age feature has about 20% of missing data.



#Описательная статистика для числовых свойств.

#Как мы видим, у свойств Age и Fare есть отсутствующие значения. Причем у Age отсутствует примерно 20% значений.



titanic_full.describe()
#Description of the string features.

#There are two missing values in Embarked and many values are missing in Cabin.



#Описание строковых свойств.

#У Embarked есть два пропущенных значения, а у Cabin отсутствует очень много значений.



titanic_full.describe(include=['O'])
#And we have passengers with the same name: Kelly, Mr. James and Connolly, Miss. Kate.

#И у нас есть пассажиры с одинаковыми именами: Kelly, Mr. James и Connolly, Miss. Kate.

pd.concat(n for _, n in titanic_full.groupby('Name') if len(n) > 1)
#Let's look at rows with missing Embarked

#Давайте посмотрим на записи с пустым Embarked

titanic_full[pd.isnull(titanic_full['Embarked'])]
#There are two women were traveling together. So we can assume they were boarded at the same city. 

#Because most of the passenger were boarded at Southampton, missing values for Embarked can be filled with 'S'



#Это две дамы, которые путешествовали вместе. Можно предположить, что они сели в одном городе. 

#Поскольку большинство пассажиров сели в Саутгемптоне, заменим эти отсутствующие значения на 'S'.



titanic_full['Embarked'].fillna('S', inplace=True)
#Missing Fare

#Отсутствующие значения Fare

titanic_full[pd.isnull(titanic_full['Fare'])]
#Missing Fare can be replaced with mean value for city of boarding and class. 

#But there were passengers who did not pay for the voyage: the Titanic's orchestra and some others. 

#Musicians were part of the crew but they lived in cabins for regular passengers and in this dataset they were 

#treated as passengers.



#Отсутствующие данные оплаты можно заменить на среднее значение для порта посадки и класса.

#Однако, некоторые пассажиры не платили за это путешествие: оркестр Титаника и некоторые другие.

#Музыканты фактически были членами экипажа, но жили в пассажирских каютах и в этом наборе данных они 

#рассматриваются как пассажиры.



fare_mean = titanic_full[titanic_full['Fare'] != 0].groupby(['Embarked', 'Pclass'])['Fare'].mean()



def get_avg_fare(df_fare):

    

    if pd.isnull(df_fare['Fare']):

        return fare_mean[df_fare['Embarked'], df_fare['Pclass']]

    else:

        return df_fare['Fare']



titanic_full['Fare'] = titanic_full.apply(get_avg_fare, axis=1)
#All names of the passengers are unique and there is a little chance that names can affect. 

#But names in the dataset contain title and this information may be useful.



#Все имена пассажиров уникальные, поэтому мало шансов, что имя может помочь в определении выживших пассажиров

#Но имена в нашем наборе данных содержат титулы, а это может быть полезной информацией.



titanic_full['Title'] = titanic_full['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(titanic_full['Title'], titanic_full['Sex'])
#Let's group some titles

#Давайте сгруппируем некорые титулы

titanic_full['Title'] = titanic_full['Title'].replace(['Mlle', 'Ms'], 'Miss')

titanic_full['Title'] = titanic_full['Title'].replace(['Mme'], 'Mrs')

titanic_full['Title'] = titanic_full['Title'].replace(['Capt', 'Col', 'Major'], 'Military')

titanic_full['Title'] = titanic_full['Title'].replace(['Don', 'Jonkheer', 'Sir'], 'NobleMale')

titanic_full['Title'] = titanic_full['Title'].replace(['Dona', 'Countess', 'Lady'], 'NobleFemale')

titanic_full['Title'] = titanic_full['Title'].replace(['Dr', 'Rev'], 'Other')



pd.crosstab(titanic_full['Title'], titanic_full['Sex'])
#Missing Age can be replaced with mean value for Sex, Title and Pclass.

#Отсутствующие данные про возраст заменяются на средний возраст для комбинации Пол/Класс/Титул

age_mean = titanic_full.groupby(['Sex', 'Pclass', 'Title'])['Age'].mean()



def get_avg_age(df_age):

    

    if pd.isnull(df_age['Age']):

        return age_mean[df_age['Sex'], df_age['Pclass'], df_age['Title']]

    else:

        return df_age['Age']



titanic_full['Age'] = titanic_full.apply(get_avg_age, axis=1)
#Tickets are almost unique, but their prefixes are inconsistent. 

#Let's keep only ticket numbers.



#Номера билеты практически уникальны, но серии билетов выглядят несогласованными

#Оставим только числовую часть от номера билета.



titanic_full['TNumber'] = titanic_full['Ticket'].str.extract('( [0-9]+$)', expand=True)

titanic_full['TNumber'].fillna(titanic_full['Ticket'], inplace=True)
#And lets count each ticket number in the dataset.

#А теперь подсчитаем количество вхождений каждого номера билеты в набор данных.

titanic_full['TNumberCount'] = titanic_full.groupby('TNumber')['TNumber'].transform('count')
#With these features, we can calculate number of family members onboard for every passenger.

#Используя эти свойства, мы можем подсчитать количество членов семьи, которые находились на борту, для каждого пассажира.



titanic_full['FamilySize'] = titanic_full['Parch'] + titanic_full['SibSp'] + 1
#From FamilySize feature we can derive Boolean feature to indicate whether the passenger has family onboard.

#Используя свойство FamilySize мы можем вывести индикатор была ли у пассажира семья на борту.



titanic_full['HasFamily'] = 0 

titanic_full.loc[titanic_full['FamilySize'] > 1, 'HasFamily'] = 1
#Now we can check if the passengers were sharing their tickets with other people.

#Теперь мы можем проверить путешествовали ли пассажиры с не членами семи по одному билету.

titanic_full['HasCompany'] = 0 

titanic_full.loc[(titanic_full['TNumberCount'] > 1) & \

                 (titanic_full['HasFamily'] == 0), 'HasCompany'] = 1
#Let's look at unique values of first letters of cabin numbers

#It is the exact list of Titanic's decks. Except of the T value.



#Давайте посмотрим на список первых букв номеров кают.

#Этот список в точности повторяет список палуб Титаника, кроме значения T.



pd.Series.unique(titanic_full['Cabin'].str[0])
#As we can see, T is the full number of the cabin. Titanic had several cabins named with one letter in its upper Boat Desk 

#near the rooms of the officers. So it is an acceptable value.



#Как мы можем увидеть, T - это полный номер каюты. И действительно, на Титанике было несколько кают с номерами, 

#состоящими только из одной буквы. Это каюты располагались на самой верхней палубе возле кают офицеров экипажа.

#Это приемлемое значение.



titanic_full[titanic_full['Cabin'].str[0] == 'T']
#And we can define Deck feature. Some cabins have 'F ' prefix followed by E or G. We will assume these cabins had access 

#from Deck F and we will use their numbers without the 'F ' prefix.



#Теперь мы можем создать свойство Deck. Некоторые номера кают начинаются с приставки 'F ', за которой следуют буквы E или G.

#Предположим, что таким образом отмечались каюты с доступам через палубу F и не будем учитывать эту приставку.



titanic_full['Cabin'].fillna('#', inplace=True)

titanic_full['Cabin'] = titanic_full['Cabin'].str.replace('F ', '')



titanic_full['Deck'] = titanic_full['Cabin'].str[0]
#Also, several passengers had several cabins. Let's count cabins for passengers.

#Также, некоторые пассажиры занимали несколько кают. Подсчитаем количество кают для пассажиров.



titanic_full['CabinCount'] = titanic_full['Cabin'].apply(lambda x: str.count(x, ' ') + 1)



titanic_full['HasManyCabins'] = 0 

titanic_full.loc[titanic_full['CabinCount'] > 1, 'HasManyCabins'] = 1
#Titanic had hole in its right side between the bow and its second funnel and then cracked right before 

#the second funnel (if we count from bow). 

#Let's define placement of the cabins: before second funnel (B) or after (S); left shipboard (L) or right (R).

#All even cabins were on the left shipboard, all odd cabins were on the right. Except Desk E: all its cabins were on the right.

#The T cabin was also on the right shipboard.



#Титаник получил пробоину в правом борту между носом и второй трубой, а затем разломился перед второй трубой.

#Давайте определим расположение кают: перед второй трубой (B) или после (S); левый борт (L) или правый борт (R).

#Все четные каюты находились по левому борту, нечетные - по правому. Все каюты на палубе E находились справа. 

#Каюта T также находилась справа.



titanic_full['CabinNumber'] = titanic_full['Cabin'].str.extract('([0-9]+$)', expand=True)

titanic_full['CabinNumber'].fillna('0', inplace=True)

titanic_full['CabinNumber'] = titanic_full['CabinNumber'].astype(int)



#Side of the ship

#Правый или левый борт

titanic_full['Side'] = '#'

titanic_full.loc[(titanic_full['CabinNumber'] != 0) & \

                 (titanic_full['CabinNumber'] % 2 == 0) & \

                 (titanic_full['Deck'] != 'E'), 'Side'] = 'L'

                 

titanic_full.loc[(titanic_full['CabinNumber'] != 0) & \

                 (titanic_full['CabinNumber'] % 2 != 0) & \

                 (titanic_full['Deck'] != 'E'), 'Side'] = 'R'



titanic_full.loc[titanic_full['Deck'] == 'E', 'Side'] = 'R'

titanic_full.loc[titanic_full['Deck'] == 'T', 'Side'] = 'R'
#Bow or stern

#Нос или корма

titanic_full['Part'] = '#'



#All cabins on Decks F and G were on the stern.

#На палубах F и G каюты пассажиров находились только в корме

titanic_full.loc[(titanic_full['Deck'] == 'F') | (titanic_full['Deck'] == 'G'), 'Part'] = 'S'



#T cabin was on the bow.

#Единственный пассажир на палубе T находился в носу

titanic_full.loc[titanic_full['Deck'] == 'T', 'Part'] = 'B'



#A

titanic_full.loc[(titanic_full['CabinNumber'] <= 35) & \

                 (titanic_full['Deck'] == 'A'), 'Part'] = 'B'



titanic_full.loc[(titanic_full['CabinNumber'] > 35) & \

                 (titanic_full['Deck'] == 'A'), 'Part'] = 'S'



#B

titanic_full.loc[(titanic_full['CabinNumber'] <= 52) & \

                 (titanic_full['Deck'] == 'B'), 'Part'] = 'B'



titanic_full.loc[(titanic_full['CabinNumber'] > 52) & \

                 (titanic_full['Deck'] == 'B'), 'Part'] = 'S'



#C

titanic_full.loc[(titanic_full['CabinNumber'] <= 40) & \

                 (titanic_full['Deck'] == 'C'), 'Part'] = 'B'



titanic_full.loc[(titanic_full['CabinNumber'] > 40) & \

                 (titanic_full['Deck'] == 'C'), 'Part'] = 'S'



#D

titanic_full.loc[(titanic_full['CabinNumber'] <= 35) & \

                 (titanic_full['Deck'] == 'D'), 'Part'] = 'B'



titanic_full.loc[(titanic_full['CabinNumber'] > 35) & \

                 (titanic_full['Deck'] == 'D'), 'Part'] = 'S'



#E

titanic_full.loc[(titanic_full['CabinNumber'] <= 31) & \

                 (titanic_full['Deck'] == 'E'), 'Part'] = 'B'



titanic_full.loc[(titanic_full['CabinNumber'] > 31) & \

                 (titanic_full['Deck'] == 'E'), 'Part'] = 'S'
#Let's look again at the data

#Давайте еще раз посмотрим на данные

titanic_full.describe()
titanic_full.describe(include=['O'])
#Now we can drop redundant features

#Теперь мы можем удалить избыточные свойства



drop_list = ['Name', 'Ticket', 'TNumber', 'Cabin', 'CabinNumber'] 

titanic_full.drop(drop_list, inplace=True, axis=1)
#Let's plot correlations between features.

#For the most precise picture we will convert string features into numerical



#Давайте построим график взаимосвязей между свойствами

#Для более точной картины преобразуем строковые свойства в численные



#Deck

deck_ordered = ['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G', '#']

titanic_full['DeckOrdered'] = titanic_full['Deck'].astype('category', ordered=True, categories=deck_ordered).cat.codes



#Embarked

titanic_full['EmbarkedOrdered'] = titanic_full['Embarked'].map( {'C': 0, 'Q': 1, 'S': 2} ).astype(int)



#Sex

titanic_full['SexOrdered'] = titanic_full['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



#Title

title_ordered = ['NobleFemale', 'Mrs', 'Miss', 'Master', 'Other', 'Military', 'NobleMale', 'Mr']

titanic_full['TitleOrdered'] = titanic_full['Title'].astype('category', ordered=True, categories=title_ordered).cat.codes



#Side

titanic_full['SideOrdered'] = titanic_full['Side'].map( {'L': 0, 'R': 1, '#': 2} ).astype(int)



#Part

titanic_full['PartOrdered'] = titanic_full['Part'].map( {'B': 0, 'S': 1, '#': 2} ).astype(int)



#Names of numeric features

#Имена численных свойств

cols = [key for key in dict(titanic_full.dtypes) if dict(titanic_full.dtypes)[key] not in ['object']]



fig = plt.figure(figsize = (12, 12))

ax = fig.add_subplot(111)



cax = ax.matshow(titanic_full[titanic_full['Survived'] != -1].corr())

fig.colorbar(cax)



ax.set_xticks(np.arange(len(cols)))

ax.set_xticklabels(cols, rotation=90)



ax.set_yticks(np.arange(len(cols)))

ax.set_yticklabels(cols)
titanic_full[titanic_full['Survived'] != -1].corr()['Survived'].sort_values(ascending=False)
#PassengerId has no correlation with the other features. HasFamily has higher correlation with Survived than Parch and SibSp. 

#So there is final list of features for prediction model.



#PassengerId не коррелирует практически ни с одним свойством. HasFamily имеет более высокую корреляцию с Survived, 

#чем Parch и SibSp. И мы можем подготовить набор свойства для модели.



drop_list = ['SibSp', 'Parch','CabinCount', 'FamilySize', 'SexOrdered', 'PassengerId', 'EmbarkedOrdered', \

            'SideOrdered', 'DeckOrdered', 'PartOrdered', 'TitleOrdered'] 

titanic_full.drop(drop_list, inplace=True, axis=1)
#Let's convert nominal features

#Преобразуем категории



nominal_list = ['Embarked', 'Title', 'Deck', 'Sex', 'Side', 'Part', 'Pclass']

titanic_full = pd.get_dummies(titanic_full, columns=nominal_list)
#Now combined dataset can be divided back into training and testing datasets.

#Теперь общий набор данных может быть разделен на тренировочный и тестовый наборы.



titanic_train = titanic_full[titanic_full['Survived'] != -1]

titanic_test = titanic_full[titanic_full['Survived'] == -1]

titanic_test = titanic_test.drop('Survived', axis=1)



labels_train = titanic_train.pop('Survived')
#Let's hide some data from classifier. We will use 70% of training data to train a model.

#Теперь скроем некоторые данные от классификатора. Мы будем использовать 70% тренировочных данных для тренировки модели.

data_train, data_test, label_train, label_test = train_test_split(titanic_train, labels_train, \

                                                                  test_size=0.3, random_state=753)



#The next values were obtained from GridSearchCV.  

#Следующие значения получены с помощью GridSearchCV

e = 100

f = None

n = 7

s = 2



model = RandomForestClassifier(n_estimators=e, \

                               max_features=f, \

                               min_samples_leaf=n, \

                               min_samples_split=s, \

                               oob_score=True, random_state=753)



model.fit(data_train, label_train)

score = model.score(data_test, label_test)
#The final score.

#Итоговый результат.

score