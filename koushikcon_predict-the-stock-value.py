import numpy as np

import pandas as pd

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

from torch.optim import lr_scheduler



from sklearn.metrics import accuracy_score

import json

from sklearn.tree import DecisionTreeRegressor

from sklearn import tree

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer

from collections import Counter

from sklearn.preprocessing import LabelEncoder, scale

from sklearn.datasets import load_boston

from sklearn.metrics import r2_score

from sklearn.cross_decomposition import PLSRegression, PLSSVD

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from pandas.plotting import scatter_matrix



import torch.utils.data

from sklearn.model_selection import train_test_split



import torch

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
train_csv = pd.read_csv("../input/train.csv", keep_default_na=False)

test_csv = pd.read_csv("../input/test.csv", keep_default_na=False)



# train = train[0:30]

print(train_csv.columns)
test_csv.shape
def convertKgtoL(x):

    if x != "":

        l = x.split(" ") 

        if l[1] and l[1] == "km/kg":

            return float(l[0]) / 0.42

        return l[0]

    return x



def convertCrtoLak(x):

    if x != "":

        l = x.split(" ") 

        if l[1] and l[1] == "Cr":

            return float(l[0]) * 100

        return l[0]

    return x



def convertToBrand(x):

    if x != "":

        l = x.split(" ")

        return l[0].upper()

    return x



def convertYearDiff(x):

    if x != "":

        return 2019 - int(x)

    return x
def preprocess_data(dataset):

    #dataset['Power'] = dataset["Power"].replace("bhp", "", regex=True).replace("null", "", regex=True)

    #dataset['Power'] = pd.to_numeric(dataset["Power"].str.strip())

    #dataset['Engine'] = dataset["Engine"].replace("CC", "", regex=True)

    #dataset['Engine'] = pd.to_numeric(dataset["Engine"].str.strip())

    #dataset['Mileage'] = dataset["Mileage"].apply(convertKgtoL)

    #dataset['Mileage'] = pd.to_numeric(dataset["Mileage"])

    #dataset['New_Price'] = dataset["New_Price"].apply(convertKgtoL)

    #dataset['New_Price'] = pd.to_numeric(dataset["New_Price"])

    

    #dataset['Year_Old'] = dataset["Year"].apply(convertYearDiff)

    #dataset['Car_Brand'] = dataset["Name"].apply(convertToBrand)

    #dataset['Kilometers_Driven'] = dataset["Kilometers_Driven"]/dataset["Kilometers_Driven"].max()

    

    

    

    #dataset['Mileage'].fillna(dataset['Mileage'].mean(), inplace=True)

    #dataset['Power'].fillna(dataset['Power'].mean(), inplace=True)

    #dataset['Engine'].fillna(dataset['Engine'].mean(), inplace=True)

    #dataset['New_Price'].fillna(0.0, inplace=True)

    

    dataset = dataset.replace('NaN', '')

    for col in list(dataset.columns):

        if col != 'Company ' and col != 'Date':

            dataset[col] = pd.to_numeric(dataset[col])



     

    dataset = dataset.drop(['ID'],axis=1)

    #dataset = dataset.drop(['Year'],axis=1)

    return dataset
train = preprocess_data(train_csv)

test = preprocess_data(test_csv)



display(train.head())

#train.columns
train['Price'].describe()
sns.distplot(train['Price']);
print("Skewness: %f" % train['Price'].skew())

print("Kurtosis: %f" % train['Price'].kurt())
train.dtypes
combined_set = pd.concat([train, test], axis=0, ignore_index=True)
corcolm = ['SMA', 'WMA', 'MACD', 'MACD_Hist', 'FastK', 'RSI',

       'FatD', 'FatK', 'ADX', 'PPO', 'MOM', 'BOP',

       'ROC', 'ROCR', 'Aroon Down', 'Aroon Up', 

       'MFI', 'ULTOSC', 'DX', 'MINUS_DI', 'MINUS_DM',

       'MIDPOINT', 'MIDPRICE', 'ATR', 'Chaikin A/D',

       'ADOSC', 'LEAD SINE', 'SINE', 'TRENDMODE',

       'DCPERIOD', 'HT_DCPHASE', 'QUADRATURE', 'Company ', 'Price'];

#correlation matrix

corrmat = combined_set[corcolm].corr()

f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corrmat, vmax=.8, square=True);

combined_set = combined_set[corcolm]

#train = train[corcolm]



total = combined_set.isnull().sum().sort_values(ascending=False)

percent = (combined_set.isnull().sum()/combined_set.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)


combined_set = combined_set.interpolate(method ='linear', limit_direction ='both')
combined_set.isnull().sum().max() #just checking that there's no missing data missing...
#saleprice correlation matrix

k = 7 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Price')['Price'].index

cm = np.corrcoef(combined_set[:11997][cols].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(15, 15))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(combined_set[:11997]['Price'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
col_train_num = list(cols)

col_train_num.append("Company ")



col_train_num_bias = list(cols)

col_train_num_bias.append("Company ")

col_train_num_bias.remove('Price')



print(col_train_num_bias)



#test = test[col_train_num_bias]

#train = train[col_train_num]



combined_set = combined_set[col_train_num]



#test = test.interpolate(method ='linear', limit_direction ='both') 

#scatter_matrix(train[col_train_num], figsize=(25, 25))

#plt.show()

#test.isnull().sum().max()
from scipy import stats

#histogram and normal probability plot

sns.distplot(combined_set[:11997]['Price'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(combined_set[:11997]['Price'], plot=plt)
combined_set[:11997]['Price'] = np.log(combined_set[:11997]['Price'])

sns.distplot(combined_set[:11997]['Price'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(combined_set[:11997]['Price'], plot=plt)
#train_numerical = train.select_dtypes(exclude=['object'])



for col in col_train_num:

    print(col)

    try:

        plt.figure()

        sns.distplot(combined_set[:11997][col], fit=stats.norm);

    except TypeError:

        print("No graph for this {} column".format(col))

    #fig = plt.figure()

    #stats.probplot(train[col], plot=plt)
for col in col_train_num:

    print(col)

    try:

        data = pd.concat([combined_set[:11997]['Price'], train[col]], axis=1)

        data.plot.scatter(x=col, y='Price', ylim=(0,10));

    except ValueError:

        print("No graph for this {} column".format(col))
combined_set['SMA'] = np.log(combined_set['SMA'])

sns.distplot(combined_set['SMA'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(combined_set['SMA'], plot=plt)
combined_set['WMA'] = np.log(combined_set['WMA'])

sns.distplot(combined_set['WMA'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(combined_set['WMA'], plot=plt)
combined_set['MINUS_DM'] = np.log(combined_set['MINUS_DM'])

sns.distplot(combined_set['MINUS_DM'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(combined_set['MINUS_DM'], plot=plt)
"""

train['PLUS_DI'] = np.log(train['PLUS_DI'])

test['PLUS_DI'] = np.log(test['PLUS_DI'])

sns.distplot(train['PLUS_DI'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train['PLUS_DI'], plot=plt)

"""
"""

train.loc[train['WILLR']>0, 'WILLR'] = np.log(train[train['WILLR']>0]['WILLR'])

sns.distplot(train['WILLR'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(train['WILLR'], plot=plt)

"""
combined_set = pd.get_dummies(combined_set)

display(combined_set.head())
print(combined_set.shape)



train = combined_set[:11997]

test = combined_set[11997:]

test = test.drop("Price",axis = 1)



col_train = list(train.columns)

col_train_bis = list(train.columns)

col_train_bis.remove("Price")



mat_train = np.matrix(train)

mat_test  = np.matrix(test)

mat_new = np.matrix(train.drop('Price',axis = 1))



mat_y = np.array(train.Price).reshape((11997,1))



prepro_y = MinMaxScaler()

prepro_y.fit(mat_y)



prepro = MinMaxScaler()

prepro.fit(mat_train)



prepro_test = MinMaxScaler()

prepro_test.fit(mat_new)



# trimed_test.to_csv("output_final_3.csv")

train_set = pd.DataFrame(prepro.transform(train),columns = col_train)



# test = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)



test_set  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)



display(train_set.head())
COLUMNS = col_train

FEATURES = col_train_bis

LABEL = "Price"



#FEATURES.remove('Price')



# Training set and Prediction set with the features to predict

training_set = train_set[col_train]

prediction_set = training_set.Price



# print(prediction_set)



X_train, X_val, y_train, y_val = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.4)



train_set_tensor = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))

val_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values))



batch_size = 8

train_loader = torch.utils.data.DataLoader(train_set_tensor,batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)
"""

train_numerical = train.select_dtypes(exclude=['object'])

train_numerical.fillna(0,inplace = True)

train_categoric = train.select_dtypes(include=['object'])

train_categoric.fillna('NONE',inplace = True)

train_new = train_numerical.merge(train_categoric, left_index = True, right_index = True)



test_numerical = test.select_dtypes(exclude=['object'])

test_numerical.fillna(0,inplace = True)

test_categoric = test.select_dtypes(include=['object'])

test_categoric.fillna('NONE',inplace = True)

test_new = test_numerical.merge(test_categoric, left_index = True, right_index = True) 



train.dtypes

"""
"""

# Removie the outliers

from sklearn.ensemble import IsolationForest



clf = IsolationForest(max_samples = 100, random_state = 42)

clf.fit(train_numerical)

y_noano = clf.predict(train_numerical)

y_noano = pd.DataFrame(y_noano, columns = ['Top'])

y_noano[y_noano['Top'] == 1].index.values



train_numerical = train_numerical.iloc[y_noano[y_noano['Top'] == 1].index.values]

train_numerical.reset_index(drop = True, inplace = True)



train_categoric = train_categoric.iloc[y_noano[y_noano['Top'] == 1].index.values]

train_categoric.reset_index(drop = True, inplace = True)



train_new = train_new.iloc[y_noano[y_noano['Top'] == 1].index.values]

train_new.reset_index(drop = True, inplace = True)

display(train_new.head())

"""
"""

#col_train = list(train_new.columns)

col_train_num = list(train_numerical.columns)

col_train_num_bis = list(train_numerical.columns)



col_train_cat = list(train_categoric.columns)



col_train_num_bis.remove('Price')



mat_train = np.matrix(train_numerical)

mat_test  = np.matrix(test_numerical)

mat_new = np.matrix(train_numerical.drop('Price',axis = 1))

mat_y = np.array(train_new.Price)



print(mat_y.shape)



prepro_y = MinMaxScaler()

prepro_y.fit(mat_y.reshape(5417,1))



prepro = MinMaxScaler()

prepro.fit(mat_train)



prepro_test = MinMaxScaler()

prepro_test.fit(mat_new)



train_num_scale = pd.DataFrame(prepro.transform(mat_train),columns = col_train_num)

test_num_scale  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_num_bis)



def oneHotEncode(df,colNames):

    for col in colNames:

        if( df[col].dtype == np.dtype('object')):

            dummies = pd.get_dummies(df[col],prefix=col)

            df = pd.concat([df,dummies],axis=1)



            #drop the encoded column

            df.drop([col],axis = 1 , inplace=True)

    return df



train_new[col_train_num] = pd.DataFrame(prepro.transform(mat_train),columns = col_train_num)

test_new[col_train_num_bis]  = test_num_scale



prediction_set = train_new.Price

print(prediction_set.shape)

combined = train_new.drop('Price',axis = 1).append(test_new)

combined.reset_index(inplace=True)

print('There were {} columns before encoding categorical features'.format(combined.shape[1]))

combined = oneHotEncode(combined, col_train_cat)

print('There are {} columns after encoding categorical features'.format(combined.shape[1]))



train_new = combined[:5417]

test_new = combined[5417:]

"""
"""

# Train and Test 

X_train, X_val, y_train, y_val = train_test_split(train_new, prediction_set, test_size=0.33, random_state=42)

train_set_tensor = torch.utils.data.TensorDataset(torch.FloatTensor(X_train.values), torch.FloatTensor(y_train.values))

val_set = torch.utils.data.TensorDataset(torch.FloatTensor(X_val.values), torch.FloatTensor(y_val.values))



batch_size = 1

train_loader = torch.utils.data.DataLoader(train_set_tensor,batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size)

"""
# Hyperparameters

# batch_no = len(X_train) // batch_size  #batches

# cols=X_train.shape[1] #Number of columns in input matrix



# Sequence Length

#sequence_length = 6  # of words in a sequence 892110

# Batch Size

# batch_size = 128

# train_loader = batch_data(int_text, sequence_length, batch_size)

# Number of Epochs

num_epochs = 3000

# Learning Rate

learning_rate = 0.002

# Model parameters

# Input size

input_size = X_train.shape[1]

# Output size

output_size = 1

# Embedding Dimension

#embedding_dim = 128

# Hidden Dimension

hidden_dim = 64

# Number of RNN Layers

n_layers = 2



# Show stats for every n number of batches

show_every_n_batches = 50
import torch.nn as nn



class DNNClassifier(nn.Module):

    """

    This is the simple DNN model we will be using to perform Sentiment Analysis.

    """



    def __init__(self, hidden_dim, input_size, output_size, dropout=0.5):

        """

        Initialize the model by settingg up the various layers.

        """

        super(DNNClassifier, self).__init__()



        self.sig = nn.Sigmoid()        

        # self.word_dict = None

        

        self.fc1 = nn.Linear(input_size, hidden_dim * 4)

        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        #self.fc3 = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        self.fc4 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.out = nn.Linear(hidden_dim, output_size)

        self.dropout = nn.Dropout(p=0.5)

        self.init_weights()

        

    def init_weights(m):

        initrange = 0.08

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:

            # get the number of the inputs

            n = m.in_features

            y = 1.0/np.sqrt(n)

            m.weight.data.normal_(0.0, y)

            m.bias.data.fill_(0)

        

    def forward(self, x):

        """

        Perform a forward pass of our model on some input.

        """

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        #x = F.relu(self.fc3(x))

        #x = self.dropout(x)

        x = F.relu(self.fc4(x))

        x = self.dropout(x)

        out = self.out(x)

        return out
from torch.autograd import Variable



def forward_back_prop(rnn, optimizer, criterion, inputs, labels, clip=9):



    if(train_on_gpu):

        inputs, labels = inputs.cuda(), labels.cuda()



    hidden = {}

    # hidden = tuple([each.data for each in hidden_dim])

    

    rnn.zero_grad()

    optimizer.zero_grad()

    #print(inputs)

    try:

        # get the output from the model

        # output, hidden = rnn(inputs, hidden)

        output = rnn.forward(inputs)

        #output = rnn(inputs.unsqueeze(0))

        output = output.squeeze()

        #print(output)

    except RuntimeError:

        raise

    #print(labels)

    loss = criterion(output, labels)

    loss.backward()

    

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.

    # nn.utils.clip_grad_norm_(rnn.parameters(),  clip)

   

    optimizer.step()



    return loss.item()
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):

    batch_losses = []

    val_batch_losses = []

    valid_loss_min = np.Inf

    

    rnn.train()

    

    previousLoss = np.Inf

    minLoss = np.Inf



    print("Training for %d epoch(s)..." % n_epochs)

    for epoch_i in range(1, n_epochs + 1):

        

        # initialize hidden state

        # hidden = rnn.init_hidden(batch_size)

        # print("epoch ",epoch_i)

        rnn.train()

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # batch_last = batch_i

            # n_batches = len(train_loader.dataset) // batch_size

            loss = forward_back_prop(rnn, optimizer, criterion, inputs, labels, clip=5)

            #print(loss)

            # record loss

            batch_losses.append(loss)

            

        rnn.eval()

        for batch_i, (inputs, labels) in enumerate(val_loader, 1):

            # batch_last = batch_i

            # n_batches = len(val_loader.dataset) // batch_size

            if(train_on_gpu):

                inputs, labels = inputs.cuda(), labels.cuda()

            # if(batch_i > n_batches):

                # break

            try:

                output = rnn.forward(inputs)

                output = output.squeeze()

            except RuntimeError:

                raise

            # print(labels)

            loss = criterion(output, labels)



            val_batch_losses.append(loss.item())



        # printing loss stats

        if epoch_i%show_every_n_batches == 0:

            average_loss = np.average(batch_losses)

            val_average_loss = np.average(val_batch_losses)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch_i, average_loss, val_average_loss))



            ## TODO: save the model if validation loss has decreased

            # save model if validation loss has decreased

            if val_average_loss < valid_loss_min:

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                valid_loss_min,

                val_average_loss))

                with open('trained_rnn_new', 'wb') as pickle_file:

                    # print(pickle_file)

                    torch.save(rnn, pickle_file)

                valid_loss_min = val_average_loss



            batch_losses = []

            val_batch_losses = []

            

    return rnn
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:

    print('No GPU found. Please use a GPU to train your neural network.')
# create model and move to gpu if available

# rnn = RNN(input_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.25)

# rnn.apply(weight_init)

#rnn = LSTMClassifier(embedding_dim, hidden_dim, input_size, n_layers, output_size)

rnn = DNNClassifier(hidden_dim, input_size, output_size)



#rnn = torch.load("trained_rnn_new")



if train_on_gpu:

    rnn.cuda()



decay_rate = learning_rate / num_epochs



# print(decay_rate)

# defining loss and optimization functions for training

#optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=decay_rate)



# criterion = nn.CrossEntropyLoss()

criterion = nn.MSELoss()

#rnn = torch.load("trained_rnn_new")



# training the model

#trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)



# saving the trained model

# helper.save_model('./save/trained_rnn', trained_rnn)

print('Model Trained and Saved')
def predict(model, inputs):



    if(train_on_gpu):

        inputs = inputs.cuda()

    

    try:

        output = model.forward(inputs)

        output = output.squeeze()

        #print(output)

    except RuntimeError:

        raise

    

    # prediction = np.array(output).argmax(0)

    # p = F.softmax(output, dim=1).data

    # p = F.sigmoid(output)

    # p = F.logsigmoid(output)

    p = output.cpu().detach().numpy().flatten()

    #print(p[0])

    # prediction = np.argmax(p)

    # print(prediction)

    return p[0]
model_rnn = torch.load("trained_rnn_new")

model_rnn.eval()

display(X_train.head())
display(test_set.head())
Val_outputs = []

print(train[:500].shape)



pred_training_set = train_set[col_train][:500]

pred_training_set = pred_training_set.drop('Price',axis = 1)



for row in pred_training_set.values:

    valoutput = predict(model_rnn, torch.FloatTensor(row))

    Val_outputs.append(valoutput)



print(Val_outputs[:10])

print(y_val.values[:10])



s_out = pd.Series(prepro_y.inverse_transform(np.array(Val_outputs).reshape(500,1)).squeeze())

t_out =  np.exp(s_out)

print(t_out.values[:20])

print(train_csv['Price'].values[:20])

r2_score(train_csv['Price'].values[:500], t_out.values)
Test_outputs = []

for row in test_set.values:

    testoutput = predict(model_rnn, torch.FloatTensor(row))

    Test_outputs.append(testoutput)



print(Test_outputs[:30])

print(len(Test_outputs))
"""

test_input = torch.randn(3, 5, requires_grad=True)

test_target = torch.randn(3, 5)

X = Variable(torch.FloatTensor(X_train.values)) 

print(X)

pred = predict(model_rnn, X)

print(pred[:30])

print(pred.shape)

# pred= result

print(y_val.values[:30])

r2_score(y_train.values, pred)



loss = nn.L1Loss()

output_loss = loss(torch.FloatTensor(y_train.values),torch.FloatTensor(pred))

print(1 - output_loss)

"""

"""

test_X = Variable(torch.FloatTensor(test_set.values))

print(test_X)

test_pred = predict(model_rnn, test_X)

print(test_pred)

print(len(test_pred))

# print(np.array(test_p).reshape(9614,1))

"""
a = np.array([2,4,8,10,12,18, 100, 200, 400])

log_a = np.log(a)

exp_a = np.exp(log_a)

print(a)

print(log_a)

print(exp_a)
s_out = pd.Series(prepro_y.inverse_transform(np.array(Test_outputs).reshape(4161,1)).squeeze())

t_out =  np.exp(s_out)

predictions = pd.DataFrame(test_csv["ID"].values, columns = ["ID"])

# predictions = pd.DataFrame(np.array(test_pred).reshape(8037,1), columns = ["FORECLOSURE"])

# predictions["FORECLOSURE"] = predictions["FORECLOSURE"]

# predictions['SalePrice'] = predictions['SalePrice']

# predictions['FORECLOSURE'] = predictions['FORECLOSURE'].apply(lambda x: 0 if x < 0.01 else 1)

# predictions['FORECLOSURE'] = predictions['FORECLOSURE'].apply(lambda x: 1 if x > 0 else x)

# predictions = predictions.round(2)

# predictions["ID"] = test_csv["ID"]

predictions["Price"] = t_out

display(predictions.head())
predictions.to_csv("submission_3.csv", index=False)
# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename="submission_1.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(predictions)