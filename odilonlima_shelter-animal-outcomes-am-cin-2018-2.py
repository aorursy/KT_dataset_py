import re
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
from collections import Counter

import warnings
warnings.filterwarnings('ignore')
# Data load
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Shape dos dados de treino: ", train.shape)
print("Shape dos dados de teste: ", test.shape)
# Colunas dados de treinamento
train.columns
# Colunas dados de teste
test.columns
train.head(10)
sns.countplot(x="AnimalType",data=train)
# Distribuicao dos dados da coluna OutcomeType
sns.countplot(x="OutcomeType",data=train)
# Distribuicao dos dados da coluna OutcomeSubtype
plt.figure(figsize=(17,6))
sns.countplot(x="OutcomeSubtype", data=train)
# Outcome pelo tipo do animal
plt.figure(figsize=(17,6))
sns.countplot(x="AnimalType",hue="OutcomeType",data=train)
# Relação entre o tipo e o sexo do animal
animal_sex = pd.crosstab(train['SexuponOutcome'], train['AnimalType'])
plt.show(animal_sex.plot(kind="bar", title = 'Tipo do animal x Sexo',figsize=(20,6)))
# Relação entre a saída e sexo do animal
sns.countplot(y="SexuponOutcome",data=train,hue="OutcomeType")
# Cria atributo AgeDays (idade em dias) a partir do atributo Age (idade)
def transform_age(df):
    df['Age'] = df['AgeuponOutcome'].str[0]
    df.Age.fillna('0',inplace=True)
    df.Age = df.Age.astype(np.int32)
    df["AgeFactor"] = 0
    df["AgeFactor"][df["AgeuponOutcome"].str[2]=='y'] = 365
    df["AgeFactor"][df["AgeuponOutcome"].str[2]=='w'] = 7
    df["AgeFactor"][df["AgeuponOutcome"].str[2]=='m'] = 30
    df["AgeDays"] = df['Age'].multiply(df['AgeFactor'])
    return df

train = transform_age(train)
#train.head()
outcome_age = pd.crosstab(train['AgeDays'], train['OutcomeType'])
plt.show(outcome_age.plot(kind="bar", title = 'Idade em dias x Outcome',figsize=(20,10)))
animal_age = pd.crosstab(train['AgeDays'], train['AnimalType'])
plt.show(animal_age.plot(kind="bar", title = 'Idade em dias x Tipo do animal',figsize=(20,6)))
age_sex = pd.crosstab(train['AgeDays'], train['SexuponOutcome'])
plt.show(age_sex.plot(kind="bar", title = 'Idade em dias x Sexo',figsize=(20,6)))
# Preenche dados ausentes
def fillna(df):
    df["SexuponOutcome"] = df["SexuponOutcome"].fillna("Unknown")    
    #df["AgeuponOutcome"] = df["AgeuponOutcome"].fillna(0)
    return df

train = fillna(train)
# OutcomeSubtype - atributo existente apenas nos dados de treinamento
train["OutcomeSubtype"] = train["OutcomeSubtype"].fillna("Unknown")
# Separa data em ano, mês, dia da semana, hora
def transform_date(df):
    date = pd.to_datetime(train.DateTime)
    df["hour"] = date.dt.hour
    df["weekday"] = date.dt.dayofweek
    df["month"] = date.dt.month
    df["year"] = date.dt.year    
    return df

train = transform_date(train)
#train.head()
# Tratamento dos dados da coluna Color
def reduce_colors(df, color_att):    
    df[color_att] = df[color_att].str.replace('.*Brown.*','Brown')
    df[color_att] = df[color_att].str.replace('.*Black.*','Black')
    df[color_att] = df[color_att].str.replace('.*White.*','White')    
    df[color_att] = df[color_att].str.replace('.*Grey.*','Grey')
    df[color_att] = df[color_att].str.replace('.*Gray.*','Grey')
    df[color_att] = df[color_att].str.replace('.*Silver.*','Grey')
    df[color_att] = df[color_att].str.replace('.*Tan.*','Brown')
    df[color_att] = df[color_att].str.replace('.*Chocolate.*','Brown')
    df[color_att] = df[color_att].str.replace('.*Blue.*','Blue')
    df[color_att] = df[color_att].str.replace('.*Yellow.*','Yellow')
    df[color_att] = df[color_att].str.replace('.*Gold.*','Yellow')
    df[color_att] = df[color_att].str.replace('.*Red.*','Orange')
    df[color_att] = df[color_att].str.replace('.*Orange.*','Orange')
    #df[color_att] = df[color_att].str.replace('.*Cream.*','Cream')
    # As substituicoes abaixo podem prejudicar a predicao, pois podem esconder preferencias
    #df[color_att] = df[color_att].str.replace('.*Calico.*','Tricolor')
    #df[color_att] = df[color_att].str.replace('.*Torbie.*','Tricolor')
    #df[color_att] = df[color_att].str.replace('.*Tortie.*','Tricolor')
    return df

def transform_color(df):
    colors = df['Color'].apply(lambda x : x.split('/'))
    df['Color_1'] = colors.apply(lambda x : x[0])
    df['Color_2'] = colors.apply(lambda x : x[1] if len(x) > 1 else x[0])
    df['Multiple_Colors'] = df['Color'].apply(lambda x : 1 if '/' in x else 0)
    df = reduce_colors(df, 'Color_1')
    df = reduce_colors(df, 'Color_2')    
    return df

train = transform_color(train)
#train.head()
# Agrupa raças de cachorro por tipos
def make_groups(df):
    feature_values_dog = df.loc[df['AnimalType'] == 'Dog', 'Breed']    
    breeds = ['Blue Lacy','Queensland Heeler','Rhod Ridgeback','Retriever','Chinese Sharpei','Black Mouth Cur','Catahoula','Staffordshire','Affenpinscher','Afghan Hound','Airedale Terrier','Akita','Australian Kelpie','Alaskan Malamute','English Bulldog','American Bulldog','American English Coonhound','American Eskimo Dog (Miniature)','American Eskimo Dog (Standard)','American Eskimo Dog (Toy)','American Foxhound','American Hairless Terrier','American Staffordshire Terrier','American Water Spaniel','Anatolian Shepherd Dog','Australian Cattle Dog','Australian Shepherd','Australian Terrier','Basenji','Basset Hound','Beagle','Bearded Collie','Beauceron','Bedlington Terrier','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Bernese Mountain Dog','Bichon Fris_','Black and Tan Coonhound','Black Russian Terrier','Bloodhound','Bluetick Coonhound','Boerboel','Border Collie','Border Terrier','Borzoi','Boston Terrier','Bouvier des Flandres','Boxer','Boykin Spaniel','Briard','Brittany','Brussels Griffon','Bull Terrier','Bull Terrier (Miniature)','Bulldog','Bullmastiff','Cairn Terrier','Canaan Dog','Cane Corso','Cardigan Welsh Corgi','Cavalier King Charles Spaniel','Cesky Terrier','Chesapeake Bay Retriever','Chihuahua','Chinese Crested Dog','Chinese Shar Pei','Chinook','Chow Chow',"Cirneco dell'Etna",'Clumber Spaniel','Cocker Spaniel','Collie','Coton de Tulear','Curly-Coated Retriever','Dachshund','Dalmatian','Dandie Dinmont Terrier','Doberman Pinsch','Doberman Pinscher','Dogue De Bordeaux','English Cocker Spaniel','English Foxhound','English Setter','English Springer Spaniel','English Toy Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Finnish Spitz','Flat-Coated Retriever','French Bulldog','German Pinscher','German Shepherd','German Shorthaired Pointer','German Wirehaired Pointer','Giant Schnauzer','Glen of Imaal Terrier','Golden Retriever','Gordon Setter','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Greyhound','Harrier','Havanese','Ibizan Hound','Icelandic Sheepdog','Irish Red and White Setter','Irish Setter','Irish Terrier','Irish Water Spaniel','Irish Wolfhound','Italian Greyhound','Japanese Chin','Keeshond','Kerry Blue Terrier','Komondor','Kuvasz','Labrador Retriever','Lagotto Romagnolo','Lakeland Terrier','Leonberger','Lhasa Apso','L_wchen','Maltese','Manchester Terrier','Mastiff','Miniature American Shepherd','Miniature Bull Terrier','Miniature Pinscher','Miniature Schnauzer','Neapolitan Mastiff','Newfoundland','Norfolk Terrier','Norwegian Buhund','Norwegian Elkhound','Norwegian Lundehund','Norwich Terrier','Nova Scotia Duck Tolling Retriever','Old English Sheepdog','Otterhound','Papillon','Parson Russell Terrier','Pekingese','Pembroke Welsh Corgi','Petit Basset Griffon Vend_en','Pharaoh Hound','Plott','Pointer','Polish Lowland Sheepdog','Pomeranian','Standard Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo Pequeno','Portuguese Water Dog','Pug','Puli','Pyrenean Shepherd','Rat Terrier','Redbone Coonhound','Rhodesian Ridgeback','Rottweiler','Russell Terrier','St. Bernard','Saluki','Samoyed','Schipperke','Scottish Deerhound','Scottish Terrier','Sealyham Terrier','Shetland Sheepdog','Shiba Inu','Shih Tzu','Siberian Husky','Silky Terrier','Skye Terrier','Sloughi','Smooth Fox Terrier','Soft-Coated Wheaten Terrier','Spanish Water Dog','Spinone Italiano','Staffordshire Bull Terrier','Standard Schnauzer','Sussex Spaniel','Swedish Vallhund','Tibetan Mastiff','Tibetan Spaniel','Tibetan Terrier','Toy Fox Terrier','Treeing Walker Coonhound','Vizsla','Weimaraner','Welsh Springer Spaniel','Welsh Terrier','West Highland White Terrier','Whippet','Wire Fox Terrier','Wirehaired Pointing Griffon','Wirehaired Vizsla','Xoloitzcuintli','Yorkshire Terrier']
    groups = ['Herding','Herding','Hound','Sporting','Non-Sporting','Herding','Herding','Terrier','Toy','Hound','Terrier','Working','Working','Working','Non-Sporting','Non-Sporting','Hound','Non-Sporting','Non-Sporting','Toy','Hound','Terrier','Terrier','Sporting','Working','Herding','Herding','Terrier','Hound','Hound','Hound','Herding','Herding','Terrier','Herding','Herding','Herding','Herding','Herding','Working','Non-Sporting','Hound','Working','Hound','Hound','Working','Herding','Terrier','Hound','Non-Sporting','Herding','Working','Sporting','Herding','Sporting','Toy','Terrier','Terrier','Non-Sporting','Working','Terrier','Working','Working','Herding','Toy','Terrier','Sporting','Toy','Toy','Non-Sporting','Working','Non-Sporting','Hound','Sporting','Sporting','Herding','Non-Sporting','Sporting','Hound','Non-Sporting','Terrier','Working','Working','Working','Sporting','Hound','Sporting','Sporting','Toy','Herding','Sporting','Herding','Non-Sporting','Sporting','Non-Sporting','Working','Herding','Sporting','Sporting','Working','Terrier','Sporting','Sporting','Working','Working','Working','Hound','Hound','Toy','Hound','Herding','Sporting','Sporting','Terrier','Sporting','Hound','Toy','Toy','Non-Sporting','Terrier','Working','Working','Sporting','Sporting','Terrier','Working','Non-Sporting','Non-Sporting','Toy','Terrier','Working','Herding','Terrier','Toy','Terrier','Working','Working','Terrier','Herding','Hound','Non-Sporting','Terrier','Sporting','Herding','Hound','Toy','Terrier','Toy','Herding','Hound','Hound','Hound','Sporting','Herding','Toy','Non-Sporting','Non-Sporting','Toy','Hound','Working','Toy','Herding','Herding','Terrier','Hound','Hound','Working','Terrier','Working','Hound','Working','Non-Sporting','Hound','Terrier','Terrier','Herding','Non-Sporting','Toy','Working','Toy','Terrier','Hound','Terrier','Terrier','Herding','Sporting','Terrier','Working','Sporting','Herding','Working','Non-Sporting','Non-Sporting','Toy','Hound','Sporting','Sporting','Sporting','Terrier','Terrier','Hound','Terrier','Sporting','Sporting','Non-Sporting','Toy']

    breeds_group = np.array([breeds,groups]).T
    dog_groups = np.unique(breeds_group[:,1])

    group_values_dog = []
    count = 0
    not_found = []
    for i in feature_values_dog:
        i = i.replace(' Shorthair','')
        i = i.replace(' Longhair','')
        i = i.replace(' Wirehair','')
        i = i.replace(' Rough','')
        i = i.replace(' Smooth Coat','')
        i = i.replace(' Smooth','')
        i = i.replace(' Black/Tan','')
        i = i.replace('Black/Tan ','')
        i = i.replace(' Flat Coat','')
        i = i.replace('Flat Coat ','')
        i = i.replace(' Coat','')

        groups = []
        if '/' in i:
            split_i = i.split('/')
            for j in split_i:
                if j[-3:] == 'Mix':
                    breed = j[:-4]               
                    if breed in breeds_group[:,0]:
                        indx = np.where(breeds_group[:,0] == breed)[0]
                        groups.append(breeds_group[indx,1][0])
                        groups.append('Mix')
                    elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                        find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                        groups.append(find_group[find_group != 'Unknown'])
                        groups.append('Mix')  
                    elif breed == 'Pit Bull':
                        groupd.append('Pit Bull')
                        groups.append('Mix')  
                    elif 'Shepherd' in breed:
                        groups.append('Herding')
                        groups.append('Mix')  
                    else:
                        not_found.append(breed)
                        groups.append('Unknown')
                        groups.append('Mix')
                else:
                    if j in breeds_group[:,0]:
                        indx = np.where(breeds_group[:,0] == j)[0]
                        groups.append(breeds_group[indx,1][0])
                    elif np.any([s.lower() in j.lower() for s in dog_groups]):
                        find_group = [s if s.lower() in j.lower() else 'Unknown' for s in dog_groups]                    
                        groups.append(find_group[find_group != 'Unknown'])
                    elif j == 'Pit Bull':
                        groups.append('Pit Bull')
                    elif 'Shepherd' in j:
                        groups.append('Herding')
                        groups.append('Mix')  
                    else:
                        not_found.append(j)
                        groups.append('Unknown')
        else:

            if i[-3:] == 'Mix':
                breed = i[:-4]
                if breed in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == breed)[0]
                    groups.append(breeds_group[indx,1][0])
                    groups.append('Mix')
                elif np.any([s.lower() in breed.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in breed.lower() else 'Unknown' for s in dog_groups]                    
                    groups.append(find_group[find_group != 'Unknown'])
                    groups.append('Mix') 
                elif breed == 'Pit Bull':
                    groups.append('Pit Bull')
                    groups.append('Mix') 
                elif 'Shepherd' in breed:
                    groups.append('Herding')
                    groups.append('Mix')  
                else:
                    groups.append('Unknown')
                    groups.append('Mix') 
                    not_found.append(breed)

            else:
                if i in breeds_group[:,0]:
                    indx = np.where(breeds_group[:,0] == i)[0]
                    groups.append(breeds_group[indx,1][0])
                elif np.any([s.lower() in i.lower() for s in dog_groups]):
                    find_group = [s if s.lower() in i.lower() else 'Unknown' for s in dog_groups]                    
                    groups.append(find_group[find_group != 'Unknown'])
                elif i == 'Pit Bull':
                    groups.append('Pit Bull')
                elif 'Shepherd' in i:
                    groups.append('Herding')
                    groups.append('Mix') 
                else:
                    groups.append('Unknown') 
                    not_found.append(i)
        group_values_dog.append(list(set(groups)))
    return np.array([feature_values_dog,group_values_dog]).T
# o tratamento considera tipos de raças de cães, necessário preencher valores ausentes para gatos
def clean_cats(df):
    df["Herding"] = df["Herding"].fillna(0) 
    df["Hound"] = df["Hound"].fillna(0)
    df["Mix"] = df["Mix"].fillna(0)
    df["Non-Sporting"] = df["Non-Sporting"].fillna(0)
    df["Pit Bull"] = df["Pit Bull"].fillna(0)
    df["Sporting"] = df["Sporting"].fillna(0)
    df["Terrier"] = df["Terrier"].fillna(0)
    df["Toy"] = df["Toy"].fillna(0)
    df["Unknown"] = df["Unknown"].fillna(0)
    df["Working"] = df["Working"].fillna(0)
    return df

# Usa resultado do agrupamento por tipo de raça e 'concatena' nos dados
def breed_groups(df):
    breed_dogs = make_groups(df)
    dataset = pd.DataFrame(breed_dogs, columns = ['Breed', 'Group'])
    group_dummies = dataset['Group'].str.join(sep='*').str.get_dummies(sep='*')
    data = pd.concat([dataset, group_dummies], axis=1)
    data.drop(['Group'], axis=1,inplace=True)
    df = df.join(data.set_index('Breed'), on='Breed')
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    df = clean_cats(df)
    return df
# Tratamento dos dados da coluna Breed
def transform_breed(df):
    #df['IsMix'] = df['Breed'].apply(lambda x : 1 if 'Mix' in x else 0)
    breeds = df['Breed'].apply(lambda x : x.split('/'))
    df['Breed_1'] = breeds.apply(lambda x : x[0])
    df['Breed_2'] = breeds.apply(lambda x : x[1] if len(x) > 1 else x[0])
    df['Multiple_Breeds'] = df['Breed'].apply(lambda x : 1 if '/' in x else 0) 
    df = breed_groups(df)
    return df
    
train = transform_breed(train)
#train.head()
# Converte dados categoricos para formato numerico
def label_encoder(df):
    le = preprocessing.LabelEncoder()
    df["AnimalType"] = le.fit_transform(df["AnimalType"])
    df["SexuponOutcome"] = le.fit_transform(df["SexuponOutcome"])
    df['Color_1'] = le.fit_transform(df['Color_1'])
    df['Color_2'] = le.fit_transform(df['Color_2'])
    
    '''
    feature_def = gen_features(columns=[df['Breed_1'], df['Breed_2']], 
                               classes=[preprocessing.LabelEncoder])
    mapper = DataFrameMapper(feature_def)
    mapper.fit_transform(df)
    '''
    #df['Breed_1'] = le.fit_transform(df['Breed_1'])
    #df['Breed_2'] = le.fit_transform(df['Breed_2'])
    
    return df

# ToDo: 
# corrigir erro do label encoder de Breed_1 e Breed_2: labels diferentes para mesmo valor
# https://github.com/scikit-learn-contrib/sklearn-pandas#same-transformer-for-the-multiple-columns

train = label_encoder(train)
# OutcomeType e OutcomeSubtype - atributos existentes apenas nos dados de treinamento
le = preprocessing.LabelEncoder()
train["OutcomeType"] = le.fit_transform(train["OutcomeType"])
train["OutcomeSubtype"] = le.fit_transform(train["OutcomeSubtype"])
train.head()
X_train = train.copy()

# Remove colunas redundantes
X_train.drop(["AnimalID","Breed","Color","DateTime","Name","AgeuponOutcome","OutcomeType",
              "OutcomeSubtype", "Age","AgeFactor"],axis=1,inplace=True)

#X_train = pd.get_dummies(X_train)
Y_train = train["OutcomeType"]
X_train.head()
#Y_train.head()
# Prepara dados de teste
test = fillna(test)
test = transform_date(test)
test = transform_age(test)
test = transform_color(test)
test = transform_breed(test)
test = label_encoder(test)

X_test = test.copy()
# Remove colunas redundantes
X_test.drop(["ID","Breed","Color","DateTime","Name","AgeuponOutcome", 
             "Age","AgeFactor"],axis=1,inplace=True)

X_test.head()
def get_score(X, y, use_svm, svm_kernel, svm_c,
              use_rf, rf_n_estimators, rf_max_depth,
              use_knn, knn_n,
              use_mlp, mlp_hl, mlp_act, mlp_lr, mlp_alpha):
    # combina classificadores em uma lista
    clf_list = []

    if use_svm:
        clf_svm = SVC(kernel=svm_kernel, C=svm_c)
        clf_list.append(('svm', clf_svm))

    if use_rf:
        clf_rf = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
        clf_list.append(('rf', clf_rf))

    if use_knn:
        clf_knn = KNeighborsClassifier(n_neighbors=knn_n)
        clf_list.append(('knn', clf_knn))

    if use_mlp:
        clf_mlp = MLPClassifier(hidden_layer_sizes=mlp_hl, activation=mlp_act, learning_rate_init=mlp_lr,
                                alpha=mlp_alpha, max_iter=1000)
        clf_list.append(('mlp', clf_mlp))

    if len(clf_list) == 0:
        return 0

    # resultado por votaçao (maioria)
    eclf = VotingClassifier(estimators=clf_list, voting='hard')

    scores = cross_val_score(eclf, X, y, cv=5)

    return scores.mean()
# definição do espaço de busca
space = {
    'use_svm': hp.choice('use_svm', [
        {'use': False,

        },
        {'use': True,
         'svm_kernel': hp.choice('svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
         'svm_c': hp.choice('svm_c', [1.0, 0.1, 0.01, 0.001])
         }
    ]),
    'use_rf': hp.choice('use_rf', [
        {'use': False,

        },
        {'use': True,
         'rf_n_estimators': hp.choice('rf_n_estimators', [10, 20, 50, 100, 200]),
         'rf_max_depth': hp.choice('rf_max_depth', [None, 10, 20, 50, 100, 200])
         }
    ]),
    'use_knn': hp.choice('use_knn', [
        {'use': False,

        },
        {'use': True,
         'knn_n': hp.choice('knn_n', [1, 3, 5, 7, 9]),
         }
    ]),
    'use_mlp': hp.choice('use_mlp', [
        {'use': False,

        },
        {'use': True,
         'mlp_hl': hp.choice('mlp_hl', [20, 50, 100, 200, 500]),
         'mlp_act': hp.choice('mlp_act', ['tanh', 'relu', 'logistic']),
         'mlp_lr': hp.choice('mlp_lr', [0.1, 0.01, 0.001, 0.0001]),
         'mlp_alpha': hp.choice('mlp_alpha', [0.1, 0.01, 0.001, 0.0001])
         }
    ]),
}
def f_nn(params):
    print(params)

    acc = get_score(X_train, Y_train,
                    use_svm=params['use_svm']['use'],
                    svm_kernel=0 if params['use_svm']['use'] is False else params['use_svm']['svm_kernel'],
                    svm_c=0 if params['use_svm']['use'] is False else params['use_svm']['svm_c'],
                    use_rf=params['use_rf']['use'],
                    rf_n_estimators=0 if params['use_rf']['use'] is False else params['use_rf']['rf_n_estimators'],
                    rf_max_depth=0 if params['use_rf']['use'] is False else params['use_rf']['rf_max_depth'],
                    use_knn=params['use_knn']['use'],
                    knn_n=0 if params['use_knn']['use'] is False else params['use_knn']['knn_n'],
                    use_mlp=params['use_mlp']['use'],
                    mlp_hl=0 if params['use_mlp']['use'] is False else (params['use_mlp']['mlp_hl'],),
                    mlp_act=0 if params['use_mlp']['use'] is False else params['use_mlp']['mlp_act'],
                    mlp_lr=0 if params['use_mlp']['use'] is False else params['use_mlp']['mlp_lr'],
                    mlp_alpha=0 if params['use_mlp']['use'] is False else params['use_mlp']['mlp_alpha'])

    print(acc)

    #text_file = open("hyperopt.csv", "a")
    #text_file.write(str(SortedDisplayDict(params)) + ',' + str(acc) + '\n')
    #text_file.close()

    return {'loss': -acc, 'status': STATUS_OK}
best = fmin(partial(f_nn), space, algo=tpe.suggest, max_evals=100)