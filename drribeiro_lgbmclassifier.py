from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics
from sklearn.metrics import classification_report

import pandas as pd

train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
colunas = ['SK_ID_CURR','TARGET','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','DAYS_BIRTH','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT_W_CITY','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT']
colunas2 = ['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','DAYS_BIRTH','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT_W_CITY','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT']

treino = train[colunas]
teste = test[colunas2]
def preenche_valores(df):
    anuidade_media = df['AMT_ANNUITY'].median()
    familia_media = df['CNT_FAM_MEMBERS'].median()
    
    df['AMT_ANNUITY'] = df['AMT_ANNUITY'].fillna(anuidade_media)
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Laborers')
    df['CODE_GENDER'] = df['CODE_GENDER'].fillna('M')
    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(familia_media)
    
    df['CREDIT'] = df['AMT_REQ_CREDIT_BUREAU_MON'] + df['AMT_REQ_CREDIT_BUREAU_QRT']
    credit_media = df['CREDIT'].median()
    df['CREDIT'] = df['CREDIT'].fillna(credit_media)
    
    df.drop(['AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT'], axis = 1, inplace=True)
    
    
    return df
def mapeia_valores(df):
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'] * (-1)
    df['CODE_GENDER'] = df['CODE_GENDER'].map({'M':0, 'F':1})
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'N':0, 'Y':1})
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'N':0, 'Y':1})
    df['NAME_CONTRACT_TYPE'] = df['NAME_CONTRACT_TYPE'].map({'Cash loans':0,
                                                             'Revolving loans':1})
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].map({'Working':0,
                                                         'Commercial associate':1,
                                                         'Pensioner':2,
                                                         'State servant':4,
                                                         'Unemployed':5,
                                                         'Student':6,
                                                         'Businessman':7,
                                                         'Maternity leave':8})
   
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map({'Secondary / secondary special':0,
                                                               'Higher education':1,
                                                               'Incomplete higher':2,
                                                               'Lower secondary':3,
                                                               'Academic degree':4})
     
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].map({'Laborers':0,
                                                       'Sales staff':1,
                                                       'Core staff':2,
                                                       'Managers':3,
                                                       'Drivers':4,
                                                       'High skill tech staff':5,
                                                       'Accountants':6,
                                                       'Medicine staff':7,
                                                       'Security staff':8,
                                                       'Cooking staff':9,
                                                       'Cleaning staff':10,
                                                       'Private service staff':11,
                                                       'Low-skill Laborers':12,
                                                       'Waiters/barmen staff':13,
                                                       'Secretaries':14,
                                                       'Realty agents':15,
                                                       'HR staff':16,
                                                       'IT staff':17})
    return df
treino_ajustado = mapeia_valores(preenche_valores(treino))
teste_ajustado = mapeia_valores(preenche_valores(teste))

treino_ajustado.dropna(inplace=True)
#SEPARAR EM TREINO E TESTE
y = treino_ajustado['TARGET']
X = treino_ajustado.drop('TARGET', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)
#AJUSTANDO AO MODELO
clf = LGBMClassifier(
        n_estimators=300,
        num_leaves=15,
        colsample_bytree=.8,
        subsample=.8,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01
    )

clf.fit(X_train,y_train)

pred = clf.predict_proba(teste_ajustado)[:, 1]

submission = pd.DataFrame({
        "SK_ID_CURR": teste_ajustado.SK_ID_CURR,
        "TARGET": pred
    })

submission.to_csv('submission.csv', index=False)