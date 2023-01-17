import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import preprocessing
df_treino = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

df_teste = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')
print ("Dataframe de treino: {} colunas x {} linhas".format(str(df_treino.shape[0]),str(df_treino.shape[1])))

print ("Dataframe de teste: {} colunas x {} linhas".format(str(df_teste.shape[0]),str(df_teste.shape[1])))
df_treino.head()
df_teste.head()
df_treino.target.value_counts()
df_train = df_treino.drop(['ID','target'], axis=1)

target_train = df_treino['target']
df_test = df_teste.drop(['ID'], axis=1)

id_test = df_teste['ID']
df_train.head()
df_test.head()
def missing_values_table(df_miss):

        mis_val = df_miss.isnull().sum()

                

        mis_val_percent = 100 * mis_val / len(df_miss)

        

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        print ("Seu dataframe tem " + str(df_miss.shape[1]) + " colunas.\n"

            "Há " + str(mis_val_table_ren_columns.shape[0]) +

              " colunas que possuem valores ausentes.")

        

        return mis_val_table_ren_columns
missing_values_table(df_train)
missing_values_table(df_test)
# Seleciona as features que possuem maior % de valores missing

def missing_values (df_missing):

    vlr_missing = 50 # 50 % de valores missing na feature

    df_missing = missing_values_table(df_missing)

    missing_columns = list(df_missing[df_missing['% of Total Values'] > vlr_missing].index)

    print ('Total de Colunas com valores missing > {}%: {}'.format(vlr_missing, len (missing_columns)))

    missing_columns
missing_values(df_train)
missing_values(df_test)
# Substituindo o tipo 'object' por zero

def object_zero (df_zero):

    for object_zero in df_zero.columns[df_zero.dtypes == 'object']:

        df_zero[object_zero] = df_zero[object_zero].factorize()[0]

    return
object_zero(df_train)
object_zero(df_test)
# Preenchendo os valores NaN com zero

df_train.fillna(0,inplace=True)

df_test.fillna(0,inplace=True)
df_train.head()
df_test.head()
def normalize_df (df_normalized):

    cols = df_normalized.columns

    min_max_scaler = preprocessing.MinMaxScaler()

    np_scaled = min_max_scaler.fit_transform(df_normalized)

    df_normalized = pd.DataFrame(np_scaled, columns = cols)

    return df_normalized
df_train = normalize_df(df_train)
df_test = normalize_df(df_test)
df_train.head()
df_test.head()
extclfa = ExtraTreesClassifier(n_estimators = 100, # número de árvores da floresta

                            criterion = 'entropy', # qualidade de uma divisão, 'entropy' ganho de informações e 'gini' para impurezas

                            max_depth = None, # produndidade máxima da árvore

                            min_samples_split = 2, # número mínimo de amostras necessárias para dividir um nó inteiro

                            min_samples_leaf = 1, # número mínimo de amostras necessárias para estar em um nó folha

                            min_weight_fraction_leaf = 0, # as amostras têm peso igual quando sample_weight não é fornecido

                            max_features = 'auto', # número de recursos a serem considerados ao procurar a melhor divisão

                            max_leaf_nodes = None, # Os melhores nós são definidos como redução relativa na impureza.

                            min_impurity_decrease = 0, # Um nó será dividido se essa divisão induzir uma diminuição da impureza maior ou igual a esse valor

                            bootstrap = False, # se False, o conjunto de dados inteiro será usado para criar cada árvore

                            n_jobs = None, # número de tarefas a serem executadas em paralelo

                            random_state = None, # Controla 3 fontes de aleatoriedade: bootstrapping, amostragem e sorteio das divisões

                            class_weight = None # pesos associados às classes, se não for dada, todas as classes terão o mesmo peso 

                           )
extclfa.fit(df_train,target_train)
predicted = extclfa.predict_proba(df_test)

predicted
predicted.max()
pd.DataFrame({"ID": id_test, "PredictedProb": predicted[:,1]}).to_csv('submission.csv',index=False)