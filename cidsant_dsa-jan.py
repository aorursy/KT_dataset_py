import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
def zero_to_mean(df, name, min_value = 10):

    mean = df[name].mean()

    df[name]  = df[name].apply(lambda x: mean if x < min_value else x )

    

def remove_outliers(df, names, low=0.1, high=0.9):

    df_filter = df[names]

    df_ret = df_filter.quantile([low, high])

    return df_filter.apply(lambda x: x[(x > df_ret.loc[low, x.name]) & (x < df_ret.loc[high, x.name])], axis=0)

    



# Print dos graficos de histograma

def print_graphics(df, drop_cols = ['id', 'classe'], figsize= (22,20)):

    df.drop(drop_cols, axis=1).hist(bins=50, figsize=figsize)

    plt.show()





# Print da Acurácia em testes

def print_test(model, X_test, y_test):

    score = model.evaluate(X_test, y_test, verbose = 0)

    accuracy = 100*score[1]



    print('Acurácia em Teste: %.4f%%' % accuracy)

    return accuracy



# prepara o arquivo para envio para o kaggle

def make_submition(model, weights_file_name, sc_X, CST_COLS, 

                   save_file_name = 'submission.csv', rem_zero=False, rem_outliers=False):

    if weights_file_name:

        model.load_weights(weights_file_name)

        

    df = pd.read_csv(test_file_name)

    

    if rem_zero:

        zero_to_mean(df, 'bmi', min_value = 10)

        zero_to_mean(df, 'grossura_pele', min_value = 10)

        zero_to_mean(df, 'glicose', min_value = 10)

        zero_to_mean(df, 'insulina', min_value = 10)

        zero_to_mean(df, 'pressao_sanguinea', min_value = 10)

        

    if rem_outliers:

        #df_quantile = remove_outliers(df, ['grossura_pele', 'insulina'], low=0.1, high=0.9)

        names = ['grossura_pele', 'insulina']

        df_filter = data[names]

        low=0.1 

        high=0.9

        df_ret = df_filter.quantile([low, high])

        df_quantile = df[names].apply(lambda x: x[(x > df_ret.loc[low, x.name]) & (x < df_ret.loc[high, x.name])], axis=0)



        df['grossura_pele'] = df_quantile['grossura_pele']

        df['insulina'] = df_quantile['insulina']

        df.dropna(how='any', axis=0, inplace=True)

        

    X = df[CST_COLS].values

    X = X.astype('float32')   

    X_test = df[CST_COLS].values



    X_test = X_test.astype('float32')



    X_test = sc_X.transform(X_test)

    

    pred = model.predict(X_test)

    

    df_submit = df[['id']]

    predict_class = np.argmax(pred, axis=1)

    predict_class = predict_class.tolist()

    df_submit['classe'] = predict_class

    

    df_submit.to_csv(save_file_name, sep=',',index=False)

    
#carregando o dataset para analise

dataset_file_name = '../input/competicao-dsa-machine-learning-jan-2019/dataset_treino.csv'

test_file_name = '../input/competicao-dsa-machine-learning-jan-2019/dataset_teste.csv'

data = pd.read_csv(dataset_file_name)
#vizualizando as primeiras cinco linhas

data.head()
#analizando registros faltantes e o tipo de cada coluna

data.info()

#não tem registro faltantes e o tipo de cada coluna é int64, vamos mudar depois para float32 para uso no Keras
#analize estatistica dos dados

data.describe()

# glicose, grossura_pele e bmi tem valores zerados, mais tarde vamos ter que tratar esses dados



# grossura_pele  e insulina parecem ter outliers será necessário analizar melhor dpois 
print_graphics(data, figsize=(22,20))
zero_to_mean(data, 'bmi', min_value = 10)

zero_to_mean(data, 'grossura_pele', min_value = 10)

zero_to_mean(data, 'glicose', min_value = 10)

zero_to_mean(data, 'insulina', min_value = 10)

zero_to_mean(data, 'pressao_sanguinea', min_value = 10)

print_graphics(data, figsize=(22,20))
df_quantile = remove_outliers(data, ['grossura_pele', 'insulina'], low=0.1, high=0.9)

data['grossura_pele'] = df_quantile['grossura_pele']

data['insulina'] = df_quantile['insulina']

data.dropna(how='any', axis=0, inplace=True)

print_graphics(data, figsize=(17,10))
data['classe'].value_counts().plot(kind='bar')

plt.show()
import seaborn as sns



correlation = data.drop(['id'], axis=1).corr()



plt.figure(figsize=(17,7))

sns.heatmap(correlation, annot = True)

plt.show()
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils





CST_COLS = ['num_gestacoes', 'glicose', 'bmi', 'pressao_sanguinea', 'indice_historico', 'idade', 'insulina', 'indice_historico']

#CST_COLS = ['glicose', 'bmi', 'idade']

#CST_COLS = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']



def normalize_data_values(data, shuffle=False):

    if shuffle:

        df = data.sample(frac=1)

    else:

        df = data.copy()



    X = df[CST_COLS].values

    X = X.astype('float32')

    size = len(X)



    sc_X = StandardScaler()



    sc_X.fit(X)



    X = sc_X.transform(X)

    y = np_utils.to_categorical(df['classe'].values, num_classes=2)

    

    return X, y, sc_X, df







def process_data(filename='dataset.csv', rem_zero=False, rem_outliers=False):    

    df = pd.read_csv(filename)

    

    if rem_zero:

        zero_to_mean(df, 'bmi', min_value = 10)

        zero_to_mean(df, 'grossura_pele', min_value = 10)

        zero_to_mean(df, 'glicose', min_value = 10)

        zero_to_mean(df, 'insulina', min_value = 10)

        zero_to_mean(df, 'pressao_sanguinea', min_value = 10)

        

    if rem_outliers:

        df_quantile = remove_outliers(df, ['grossura_pele', 'insulina'], low=0.1, high=0.9)

        df['grossura_pele'] = df_quantile['grossura_pele']

        df['insulina'] = df_quantile['insulina']

        df.dropna(how='any', axis=0, inplace=True)

        

        

    X, y, sc_X, df= normalize_data_values(df, shuffle=False)

    

    return X, y, sc_X, df



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam, SGD, RMSprop



def make_model(units, X, opt, dropout=True):    

    model = Sequential()

    model.add(Dense(units[0], input_dim=X.shape[1], activation='relu', kernel_initializer='uniform'))

    

    if dropout:

        model.add(Dropout(0.2))

        

    for i in range(1, len(units)):

        model.add(Dense(units[i], activation='relu', kernel_initializer='uniform'))

        if dropout:

            model.add(Dropout(0.2))

            

    model.add(Dense(2,kernel_initializer='uniform', activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



    

    return model
#CST_COLS = ['num_gestacoes', 'glicose', 'bmi', 'pressao_sanguinea', 'indice_historico', 'idade', 'insulina', 'indice_historico']

#CST_COLS = ['glicose', 'bmi', 'idade']

CST_COLS = ['num_gestacoes', 'glicose', 'pressao_sanguinea', 'grossura_pele', 'insulina', 'bmi', 'indice_historico', 'idade']

rem_zero=True

rem_outliers=False

X, y, sc_X, data = process_data(filename=dataset_file_name, rem_zero=rem_zero, rem_outliers=rem_outliers)
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split



verbose = 1



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 22,  stratify=data['classe'])



filepath_model_checkpoint="weights.best.hdf5"

#opt = RMSprop(lr=0.04, decay=1e-6)

#opt = RMSprop(lr=0.047, decay=1e-6)

#opt = make_opt('weights.best-adam.decay-0.017-12.8.hdf5')

opt = Adam(lr=0.007, decay=1e-6)

model = make_model([300,200,100,50], X, opt, dropout=True)

#model = make_model([200,100], X, opt, dropout=True)

#model = make_model([100], X, opt, dropout=True)

#model = make_model([12, 8], X, opt, dropout=True)

model_checkpoint = ModelCheckpoint(filepath=filepath_model_checkpoint, monitor='val_acc', verbose=verbose, mode='auto', save_best_only=True)

monitor = EarlyStopping(monitor = 'val_acc', min_delta = 1e-3, patience = 50, verbose = verbose, mode = 'auto')

callbacks_list = [model_checkpoint]



#model.fit(X, y, batch_size=100, epochs=1000, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=verbose)

model.fit(X, y, batch_size=100, epochs=1000, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=verbose)

model.load_weights(filepath_model_checkpoint)

accuracy = print_test(model,X_test, y_test)



save_file_name = 'sampleSubmission.csv'



make_submition( model, weights_file_name = None, sc_X=sc_X, 

               CST_COLS=CST_COLS, save_file_name = save_file_name,

              rem_zero=rem_zero, rem_outliers=rem_outliers)

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

import keras.backend as K





def do_train(X, y, verbose):

    #filepath_model_checkpoint="weights.best.{}"

    filepath_model_checkpoint="weights.best.hdf5"

    

    opts = []

    

    opts_list = np.arange(0.001, 0.051, 0.001)

    np.random.shuffle(opts_list)

    

    for lr in  opts_list:

        if lr > 0:

            opts.append({'opt': Adam(lr=lr, decay=0.00), 'name':'adam-' + str(lr), 'lr': lr})

            opts.append({'opt': SGD(lr=lr, decay=0.00, nesterov=True), 'name':'sgd-'+ str(lr), 'lr': lr})

            opts.append({'opt': RMSprop(lr=lr, decay=0.00), 'name':'RMSprop-'+ str(lr), 'lr': lr})

            opts.append({'opt': Adam(lr=lr, decay=1e-6), 'name':'adam.decay-' + str(lr), 'lr': lr})

            opts.append({'opt': RMSprop(lr=lr, decay=1e-6), 'name':'RMSprop.decay-'+ str(lr), 'lr': lr})

            opts.append({'opt': SGD(lr=lr, decay=1e-6, nesterov=True), 'name':'sgd.decay-'+ str(lr), 'lr': lr})

            

    units = [[100], [12,8], [12, 8, 4], [100,50], [200, 100], [200, 150, 100, 50], 

             [200, 200, 100, 50], [300, 200, 100, 50], [500,200,100,50], [250,100, 20]]

    np.random.shuffle(units)

    model_list = []

    max_list = []

    max_acc = ""

    max_acc_value = 0

    for ls in opts:

        print('###############', ls, '###############')

        max_value = 0

        max_name = ''

        for unit in units:        

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 22,  stratify=data['classe'])

            opt = ls['opt']

            model = make_model(units=unit, X = X_train, opt=opt, dropout=True)

            sunit ='.'.join(map(str, unit))

            file_name = filepath_model_checkpoint #.format(ls['name'] +'-' + sunit)

            print('-------', ls['name'] +'-' + sunit, '-------')

            print(file_name)

            

           

            model_checkpoint = ModelCheckpoint(filepath=file_name, monitor='val_acc', verbose=verbose, mode='auto', save_best_only=True)

            monitor = EarlyStopping(monitor = 'val_acc', min_delta = 1e-3, patience = 50, verbose = verbose, mode = 'auto')

            callbacks_list = [model_checkpoint, monitor]

            model.fit(X_train, y_train, batch_size=100, epochs=1000, validation_data=(X_test, y_test), callbacks=callbacks_list, verbose=verbose)

            model.load_weights(file_name)

            accuracy = print_test(model,X_test, y_test)

            if accuracy > max_value:

                if accuracy > max_acc_value:

                    max_acc_value =  accuracy

                    max_name = ls['name'] + ': ' + str(ls['lr']) +' -' + sunit

                    

                max_value = accuracy

                max_name = ls['name'] + ': ' + str(ls['lr']) +' -' + sunit

                max_list.append({'name':max_name, 'acc:': accuracy, 'neurons':sunit, 'lr':ls['lr']})

                model_list.append({'name':max_name, ' acc:': accuracy, 'model':model })

            print('------------------------------.: acc:', max_acc_value, ' name:',max_name) 

            

            

            

    print('***************BEST VALUES****************************')

    for value in max_list:

        print(value)

        

    return model_list
#model_list = do_train(X, y, verbose=0)


def make_opt(pattern):

    opts = pattern.split('-')

    name = opts[1].split('.')

    lr = round(float(opts[2]), 3)

    decay = 0

    if(len(name) > 1):

        decay = 1e-6

    

    name = name[0]

    

    print('name:',name, 'lr:', lr, 'decay:', decay )

    opt_dic = {

        'adam': Adam(lr=lr, decay=decay),

        'sgd':SGD(lr=lr, decay=decay, nesterov=True),

        'RMSprop':RMSprop(lr=lr, decay=1e-6)

    }

    

    

    return opt_dic[name]



def load_weights():

    #weights.best-adam.decay-0.017-12.8.hdf5

    from sklearn.model_selection import train_test_split

    file_name = '../input/weightsbestadamdecay0017128hdf5/weights.best-adam.decay-0.017-12.8.hdf5'



    save_file_name = 'submission.csv'







    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 22,  stratify=data['classe'])



    units = file_name.split('-')[3].split('.')

    units = [int(units[i]) for i in range(len(units) - 1)]

    print(units)

    opt = make_opt(file_name)



    model = make_model(units, X, opt, dropout=True)

    model.load_weights(file_name)

    make_submition( model, weights_file_name = None, sc_X=sc_X, CST_COLS=CST_COLS, save_file_name = save_file_name, rem_zero=True, rem_outliers=False)



    print_test(model, X_test, y_test)